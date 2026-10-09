# SPDX-License-Identifier: Apache-2.0
"""Apply only the additive record-copy capability to an isolated serving bridge."""
import argparse
from pathlib import Path
import shutil


def prepare(parent, output):
    output.mkdir(parents=True, exist_ok=False)
    for path in parent.iterdir():
        if path.is_file():
            shutil.copy2(path, output / path.name)
    header = output / 'tp2_native_decode_graph.h'
    text = header.read_text()
    text = text.replace('copyIntegerRowToHost(const at::Tensor& source, int64_t columns) {',
                        'copyIntegerRowToHost(const at::Tensor& source, int64_t columns, bool dspark_record = false) {')
    text = text.replace('source.device().type() == at::kHPU && source.sizes() == at::IntArrayRef({1, columns}) &&',
                        'source.device().type() == at::kHPU &&\n'
                        '                  (source.sizes() == at::IntArrayRef({1, columns}) ||\n'
                        '                   (dspark_record && source.sizes() == at::IntArrayRef({columns}))) &&')
    text = text.replace('(columns == 1 || columns == 4) && (bytes == 4 * columns || bytes == 8 * columns)',
                        '(columns == 1 || columns == 4 || (dspark_record && columns == 16)) &&\n'
                        '                  (bytes == 4 * columns || bytes == 8 * columns)')
    needle = ('inline std::pair<at::Tensor, std::shared_ptr<NativeCompletion>> '
              'copyIntegerRecordToHost(const at::Tensor& source) {\n'
              '  return copyIntegerRowToHost(source, 4);\n}\n')
    if needle not in text or text == header.read_text():
        raise ValueError('Parent does not expose the qualified integer-copy contract')
    text = text.replace(needle, needle + '\ninline std::pair<at::Tensor, std::shared_ptr<NativeCompletion>> '
                        'copyDSparkRecordToHost(const at::Tensor& source) {\n'
                        '  return copyIntegerRowToHost(source, 16, true);\n}\n')
    header.write_text(text)
    source = output / 'tp2_fused_ar_norm_bridge.cpp'
    text = source.read_text()
    needle = '  module.def("copy_integer_record_to_host", &tp2_native::copyIntegerRecordToHost);'
    if needle not in text:
        raise ValueError('Parent integer-copy registration is missing')
    source.write_text(text.replace(needle, needle + '\n'
                      '  module.def("copy_dspark_record_to_host", &tp2_native::copyDSparkRecordToHost);\n'
                      '  module.attr("dspark_record_readback_version") = 1;'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.parent, args.output)
