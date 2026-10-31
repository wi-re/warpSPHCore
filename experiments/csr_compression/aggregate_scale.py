#!/usr/bin/env python3
"""Aggregate independent-process scaling fixtures; preserve individual provenance."""
import argparse
import itertools
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive',type=Path)
args=parser.parse_args()
for precision in ('float32','float64'):
    cases,activity,inputs=[],[],[]
    for dimension,nh,n in itertools.product((1,2,3),(4.,2.57),(1000000,2000000,4000000,8000000,16000000)):
        path=args.archive/f'{precision}-d{dimension}-h{nh:g}-n{n}.json'
        report=json.loads(path.read_text())
        assert report['complete'] and report['uncontended_process_snapshots'],path
        assert len(report['cases'])==1,path
        cases.extend(report['cases'])
        activity.extend(report['activity'])
        inputs.append(path.name)
    (args.archive/f'final-{precision}.json').write_text(json.dumps(dict(
        complete=True,precision=precision,cases=cases,activity=activity,
        uncontended_process_snapshots=True,individual_reports=inputs,
        note='Source/library hashes, calibration and environment are in each individual report; schedules remain fixed from the preceding 262K screen.'),indent=2)+'\n')
print('Aggregated 60 qualified fixtures')
