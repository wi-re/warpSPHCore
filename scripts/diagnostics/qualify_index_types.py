#!/usr/bin/env python3
"""Qualify import-time integer roles in fresh precision/width processes."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import torch
import warp as wp

ROOT=Path(__file__).resolve().parents[2]
TESTS=['tests/test_type_config.py','tests/radiusSearch/test_indexDtypes.py',
       'tests/radiusSearch/test_indexStreams.py',
       'tests/operations/test_operations_core.py','tests/operations/test_grid_modes.py']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--precisions',nargs='+',choices=['float32','float64'],default=['float32','float64'])
    parser.add_argument('--index-dtypes',nargs='+',choices=['int32','int64'],default=['int32','int64'])
    parser.add_argument('--offset-dtypes',nargs='+',choices=['int32','int64'],default=['int32','int64'])
    parser.add_argument('--extra-tests',nargs='*',default=[])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('Accessible CUDA required; cannot qualify with skipped CUDA tests.')
    paths=list((ROOT/'src/warpSPHCore').rglob('*.py'))+[Path(__file__)]+[ROOT/p.split('::')[0] for p in TESTS+args.extra_tests]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report=dict(complete=False,torch=torch.__version__,warp=wp.__version__,device=args.device,
                gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
                source_sha256=hashes,records=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    with tempfile.TemporaryDirectory(prefix='warpsphcore-index-matrix-') as directory:
        for precision,ids,offsets in itertools.product(args.precisions,args.index_dtypes,args.offset_dtypes):
            env=dict(os.environ,warpSPHCore_PRECISION=precision,warpSPHCore_INDEX_DTYPE=ids,warpSPHCore_OFFSET_DTYPE=offsets)
            if args.device=='cpu':env['CUDA_VISIBLE_DEVICES']='-1'
            xml=Path(directory)/f'{precision}-{ids}-{offsets}.xml'
            command=[sys.executable,'-m','pytest','-q',*TESTS,*args.extra_tests,'-x',f'--junitxml={xml}']
            print(f'Qualifying {precision}, IDs={ids}, offsets={offsets}, {args.device}',flush=True)
            result=subprocess.run(command,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            record=dict(precision=precision,index_dtype=ids,offset_dtype=offsets,command=command[:-1],returncode=result.returncode)
            if xml.exists():
                suites=ET.parse(xml).getroot().iter('testsuite')
                record['tests']=[dict(suite.attrib) for suite in suites]
            report['records'].append(record);save()
            print(result.stdout,flush=True)
            if result.returncode:raise SystemExit(result.returncode)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in hashes.items())
    report['complete']=True;save()


if __name__=='__main__':main()
