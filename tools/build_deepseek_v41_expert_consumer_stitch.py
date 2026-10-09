# SPDX-License-Identifier: Apache-2.0
"""Private compiler role admission for C2-C6 expert BMM consumers."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


POLICY = r'''
// Admit only the C2-C6 horizontal expert matrix. All subsequent dependency,
// access-pattern, slicing and SRAM-capacity validators remain authoritative.
static bool dsparkExpertConsumer(const pBundle& bundle, const HabanaGraph& graph)
{
    const char* mode = std::getenv("VLLM_HPU_DSV41_DSPARK_EXPERT_CONSUMER_STITCH");
    if (!mode || mode[0] != '1' || mode[1] != '\0' || bundle->getNodes().empty()) return false;
    const auto& matrix = bundle->getNodes().front();
    if (!HabanaGraph::runsOnMME(matrix) || matrix->getInputs().size() < 2) return false;
    for (unsigned index = 0; index < 2; ++index)
    {
        auto tensor = matrix->getInput(index);
        auto producer = graph.getTensorProducer(tensor);
        for (unsigned depth = 0; producer && producer->isLogicalOperation() && depth < 4; ++depth)
        {
            if (producer->getInputs().size() != 1) break;
            tensor = producer->getInput(0);
            producer = graph.getTensorProducer(tensor);
        }
        if (!producer || !HabanaGraph::runsOnTPC(producer)) continue;
        const auto guid = std::static_pointer_cast<TPCNode>(producer)->getGUIDWithoutDtype();
        if (guid.find("custom_deepseek_v41_expert") != 0 || guid.find("sat") == std::string_view::npos) continue;
        const auto& activation = matrix->getInput(1 - index);
        if (activation && activation->getDim() == 3 && activation->getSizeInElements(0) >= 2560 &&
            activation->getSizeInElements(2) >= 6 && activation->getSizeInElements(2) <= 18)
            return true;
    }
    return false;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--runtime-source', type=Path, required=True)
    parser.add_argument('--compile-database', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    build, parent, runtime = args.build.resolve(), args.parent.resolve(), args.runtime_source.resolve()
    build.mkdir(parents=True, exist_ok=False)
    unit = 'pipeline_management/pipeline_bundlizer.cpp'
    source = runtime / 'synapse/src/graph_compiler/passes/sram_management' / unit
    original = source.read_text()
    candidate = '#include <cstdlib>\n#include <cstdio>\n' + original
    policy = POLICY.replace('const pBundle& bundle', 'const NodePtr& matrix')
    policy = policy.replace(" || bundle->getNodes().empty()", "")
    policy = policy.replace('    const auto& matrix = bundle->getNodes().front();\n', '')
    anchor = 'void MantaRayBundlizer::addConsumers('
    assert candidate.count(anchor) == 1
    candidate = candidate.replace(anchor, policy + '\n' + anchor)
    anchor = '    if (consumerChain.empty()) return;'
    assert candidate.count(anchor) == 1
    candidate = candidate.replace(anchor, r'''    if (dsparkExpertConsumer(mmeNode, m_graph))
        std::fprintf(stderr, "DSPARK_EXPERT_CONSUMER_PIPELINE_READY node=%s chain=%zu dims=%zu\n",
                     mmeNode->getNodeName().c_str(), consumerChain.size(), params.sharedOperandSlicingDims.size());
    if (consumerChain.empty()) return;''')
    anchor = '    for (auto output : lastConsumedNode->getOutputs())'
    assert candidate.count(anchor) == 1
    candidate = candidate.replace(anchor, '''    const bool expertConsumer =
        dsparkExpertConsumer(std::static_pointer_cast<Node>(*mmeNodes.begin()), m_graph);
    for (auto output : lastConsumedNode->getOutputs())''')
    anchor = '        if (consumerCanBeProducer(output, lastConsumedNode)) return false;'
    assert candidate.count(anchor) == 1
    candidate = candidate.replace(anchor,
        '''        if (consumerCanBeProducer(output, lastConsumedNode) && !expertConsumer) return false;''')
    private = build / source.name
    private.write_text(candidate)
    (build / 'source.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), candidate.splitlines(True),
        fromfile='a/' + unit, tofile='b/' + unit)))
    database = json.loads(args.compile_database.read_text())
    entry, = [row for row in database if row['file'].endswith('/sram_management/' + unit)]
    command = shlex.split(entry['command'])
    command = [item.replace('runtime-sources/Intel_Gaudi3_Software', 'runtime-sources/' + runtime.name)
               for item in command]
    command[1:1] = ['-I' + str(source.parent)]
    obj = build / 'dspark_expert_consumer_stitch.cpp.o'
    command[command.index('-o') + 1] = str(obj)
    command[command.index('-c') + 1] = str(private)
    archive_parent = next(Path(item) for item in json.loads((parent / 'commands.json').read_text())['link']
                          if item.endswith('/libGraphCompiler.a'))
    archive = build / 'libGraphCompiler.a'
    member = source.name + '.o'
    target = ' T MantaRayBundlizer::isBundleSupportedForConsumers('
    members = subprocess.check_output(['ar', 't', str(archive_parent)], text=True).splitlines()
    matches = []
    for occurrence in range(1, members.count(member) + 1):
        extracted = build / f'archive-member-{occurrence}'
        extracted.mkdir()
        subprocess.run(['ar', 'xN', str(occurrence), str(archive_parent), member],
                       cwd=extracted, check=True)
        symbols = subprocess.check_output(['nm', '-C', str(extracted / member)], text=True)
        if any(target in row
               for row in symbols.splitlines()):
            matches.append(occurrence)
    if len(matches) != 1:
        raise RuntimeError('Compiler archive must contain exactly one role-admission definition')
    occurrence, = matches
    link = list(json.loads((parent / 'commands.json').read_text())['link'])
    link[link.index(str(archive_parent))] = str(archive)
    link[link.index('-o') + 1] = str(build / 'libSynapse.so')
    (build / 'commands.json').write_text(json.dumps({'compile': command, 'link': link}, indent=2) + '\n')
    with (build / 'build.log').open('w') as log:
        subprocess.run(command, cwd=entry['directory'], stdout=log, stderr=subprocess.STDOUT, check=True)
        shutil.copy2(archive_parent, archive)
        subprocess.run(['ar', 'dN', str(occurrence), str(archive), member],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['ar', 'r', str(archive), str(obj)], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(link, stdout=log, stderr=subprocess.STDOUT, check=True)
    if source.read_text() != original:
        raise RuntimeError('Shared compiler source changed during the isolated build')
    proof = dict(status='built_unqualified', candidate_name='expert BMM output-consumer stitching',
                 default_enabled=False, parent_library=str(parent / 'libSynapse.so'),
                 parent_sha256=hashlib.sha256((parent / 'libSynapse.so').read_bytes()).hexdigest(),
                 library=str(build / 'libSynapse.so'),
                 sha256=hashlib.sha256((build / 'libSynapse.so').read_bytes()).hexdigest(),
                 shared_source=str(source),
                 shared_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 pipeline_bundlizer=True,
                 scope='role admission only; downstream validators unchanged; C1 and prefill geometry excluded',
                 archive_member_count=len(members), replacement_archive_member=member,
                 replaced_occurrence=occurrence, same_name_other_occurrences_unchanged=True)
    (build / 'RESULT.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
