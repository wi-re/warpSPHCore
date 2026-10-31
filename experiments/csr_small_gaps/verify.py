#!/usr/bin/env python3
"""Independent scalar readers, exact GPU IDs and dense interpolation checks."""
import argparse
import itertools
import json
from pathlib import Path
import torch
from gaps_runtime import BAND, NIBBLE, RAW32, RAW64, Library, build_plan, descriptor, prefix
# Reuse the earlier independent dense math fixtures, not its codec readers.
import importlib.util
spec = importlib.util.spec_from_file_location('byte_verify', Path(__file__).parents[1]/'csr_compression/verify.py')
fixtures = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = fixtures
spec.loader.exec_module(fixtures)


def decode_cpu(plan):
    buffers = {key: value.cpu().tolist() for key, value in plan.buffers.items()}
    data, result = buffers['codes'], []
    def word(at):
        return sum(data[at + k] << (8 * k) for k in range(4))
    def nibble(at, index):
        return (data[at + index // 2] >> (4 * (index % 2))) & 15
    for row, count in enumerate(buffers['counts']):
        if not count:
            continue
        begin = buffers['row_blocks'][row]
        previous = word(begin)
        cursor = begin + 4
        if plan.encoding == BAND:
            cursor += (count + 3) // 4
            for edge in range(count):
                tag = (data[begin + 4 + edge // 4] >> (2 * (edge % 4))) & 3
                length = 1 if tag == 2 else 4 if tag == 3 else 0
                gap = tag + 1 if tag < 2 else sum(data[cursor+k] << (8*k) for k in range(length)) + 1
                cursor += length
                if edge:
                    previous += gap
                result.append(previous)
        else:
            for start in range(0, count, plan.chunk):
                unity = word(cursor)
                nonunit = [lane for lane in range(plan.chunk) if not unity & (1 << lane)]
                info = cursor + 4
                payload = info + (len(nonunit) + 1) // 2
                info_cursor = data_cursor = 0
                for lane in range(min(plan.chunk, count-start)):
                    gap = 1
                    if lane in nonunit:
                        tag = nibble(info, info_cursor)
                        info_cursor += 1
                        if tag >= 8:
                            gap = tag - 6
                        else:
                            gap = sum(nibble(payload, data_cursor+k) << (4*k) for k in range(tag+1))
                            data_cursor += tag + 1
                    if start+lane:
                        previous += gap
                    result.append(previous)
                cursor = payload + (data_cursor + 1) // 2
    return torch.tensor(result, dtype=torch.int64)


def check_ids(library, plan, offsets, indices):
    torch.testing.assert_close(decode_cpu(plan), indices.cpu(), rtol=0, atol=0)
    actual = torch.empty_like(indices)
    request = descriptor(dict(plan.buffers, row_offsets=offsets, decoded=actual), len(plan.buffers['counts']))
    for items in (1,2,4):
        actual.fill_(-1)
        library.codec(request, plan.encoding, plan.chunk, items)
        torch.testing.assert_close(actual, indices, rtol=0, atol=0)


def synthetic(library):
    rows = [[], [0], [2147483647]] + [list(range(n)) for n in (2,3,4,7,8,15,16,31,32,33,65,1025)]
    gaps = [1,2,3,8,9,10,15,16,17,254,255,256,257,65535,65536,16777215,16777216,268435456]
    rows += [[0]+list(itertools.accumulate(gaps)), list(range(0,2048,2)), list(range(100,3000,9))]
    for input_rows in (rows, [], [[],[],[]]):
        counts = torch.tensor([len(row) for row in input_rows], device='cuda', dtype=torch.int32)
        offsets = prefix(counts)
        indices = torch.tensor([j for row in input_rows for j in row], device='cuda', dtype=torch.int64)
        for encoding, chunk in itertools.product((BAND,NIBBLE),(4,8,16,32)):
            check_ids(library, build_plan(library,indices,counts,offsets,chunk,encoding),offsets,indices)
    return 24


def numerical(library,n,dimension,dtype,periodic,quick):
    state,values,_,flags = fixtures.fixture(n,dimension,dtype,periodic)
    expected,mask = fixtures.dense(state,values,flags)
    counts,indices = mask.sum(1).int(),mask.nonzero()[:,1].contiguous()
    offsets = prefix(counts)
    output = torch.empty_like(values)
    tensors = {name:getattr(state,name) for name in ('positions','supports','masses','densities')}
    tensors.update(values=values,lower=values.new_zeros(dimension),upper=values.new_ones(dimension),
                   periodic=flags,packed=values.new_empty((n,((2*dimension+3)//4)*4)),output=output)
    records = 0
    for encoding,chunk in itertools.product((BAND,NIBBLE),(8,32) if quick else (4,8,16,32)):
        plan = build_plan(library,indices,counts,offsets,chunk,encoding)
        check_ids(library,plan,offsets,indices)
        request = plan.request(tensors,n,dimension,int(dtype==torch.float64))
        assert not request.indices and not request.row_offsets
        for items,batch,threads in itertools.product((1,2,4),(1,2),(128,) if quick else (128,256)):
            library.run(request,encoding,chunk,items,batch,threads)
            torch.testing.assert_close(output.double(),expected,rtol=4e-5 if dtype==torch.float32 else 4e-10,
                                      atol=4e-5 if dtype==torch.float32 else 4e-10)
            for raw in (RAW32,RAW64):
                raw_ids = indices.int() if raw==RAW32 else indices
                raw_output = torch.empty_like(output)
                raw_request = descriptor(dict(tensors,counts=counts,row_offsets=offsets,indices=raw_ids,output=raw_output),
                                         n,dimension=dimension,precision=int(dtype==torch.float64))
                library.run(raw_request,raw,chunk,items,batch,threads)
                torch.testing.assert_close(output,raw_output,rtol=0,atol=0)
            records += 1
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    library=Library(args.library)
    torch.set_num_threads(8)
    count=synthetic(library)
    for n,d,dtype,periodic in itertools.product((33,257) if args.quick else (1,33,257,769),
                                               (3,) if args.quick else (1,2,3),
                                               (torch.float32,torch.float64),(True,False)):
        count += numerical(library,n,d,dtype,periodic,args.quick)
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        count += numerical(library,33,3,torch.float32,True,True)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    args.output.write_text(json.dumps(dict(complete=True,configurations=count,nondefault_stream=True),indent=2)+'\n')
    print(f'Passed {count} configurations, independent readers and nondefault stream',flush=True)

if __name__=='__main__': main()
