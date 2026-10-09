# SPDX-License-Identifier: Apache-2.0
"""Freeze a completed actual-input C1 oracle before its native timing gate."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    ranks = [json.loads((args.run / f"request-c6-rank{i}.json").read_text()) for i in range(4)]
    identity = {(row["candidate"], row["group"]) for row in ranks}
    if len(identity) != 1 or any(row["status"] != "diagnostic_completed" for row in ranks):
        raise ValueError("All four actual-input oracle owners must complete the same candidate/group")
    candidate, group = identity.pop()
    key = "selection_oracle" if candidate == "threshold_selection" else "projection_oracle"
    cases = [row[key] for row in ranks]
    if not all(len(rows) == 3 and all(case["passed"] for case in rows) for rows in cases):
        raise ValueError("Three real input oracles per rank must all pass before timing")
    profile = json.loads((args.run / "runtime-profile.json").read_text())
    native = Path(profile["environment"]["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"])
    proof = dict(
        kind=("mhc_epilogue" if candidate == "mhc_mme_epilogue" else
              "c1_selection" if key == "selection_oracle" else "c1_projection"),
        candidate=candidate,
        eligible_group=group,
        native_binaries={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(native.glob("*.so"))},
        fixtures_by_rank=[[fixture["sha256"] for fixture in row["fixtures"]] for row in ranks],
        oracle=cases,
        source_run=str(args.run.resolve()),
        scope="Three real request inputs × four ranks; accepted C1 primitive comparison; service quality pending",
    )
    with args.output.open("x") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
