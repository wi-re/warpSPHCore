#!/usr/bin/env python3
"""Production-query-layout build paths with capacity validation before collection."""
import hashlib
import json
from pathlib import Path
import sys

import torch
import build_paths
import layout as experiment
import validated_collect


def main():
    paths=[Path(__file__),Path(__file__).with_name('validated_collect.py')]
    hashes={str(p.relative_to(build_paths.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    original=experiment.commonInputs
    experiment.commonInputs=build_paths.productionQueryInputs
    try:
        with validated_collect.collectors():build_paths.main()
    finally:experiment.commonInputs=original
    output=Path(sys.argv[sys.argv.index('--output')+1])
    report=json.loads(output.read_text())
    report['collector_policy']='capacity validated from host count-pass total/max before launch; per-edge overflow branch removed; fixed geometry during collection'
    report['generated_sources']={kind:info for kind,(_,info) in validated_collect.KERNELS.items()}
    report['source_sha256'].update(hashes)
    assert all(hashlib.sha256((build_paths.ROOT/path).read_bytes()).hexdigest()==sha for path,sha in hashes.items())
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):main()
    stream.synchronize()
