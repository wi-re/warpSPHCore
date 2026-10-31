"""Offline storage models; these are not implemented GPU consumers or timings.

Band's shifted deltas 0/1 correspond to positive neighbor-index gaps 1/2.
Models use a uint32 first ID and common uint64 row offset + int32 count metadata.
The nibble model streams warp-sized segments within each row; our implemented
chunk32 format instead restarts each chunk and stores its own payload directory.
"""
import numpy as np


def storage_models(ids, counts):
    rows, width = ids.shape
    columns = np.arange(width-1)[None,:]
    gap_valid = columns < np.maximum(counts-1,0)[:,None]
    gaps = np.diff(ids,axis=1)
    assert np.all(gaps[gap_valid] > 0)
    common = 12*rows+8
    first = 4*np.count_nonzero(counts)
    control = np.sum((np.maximum(counts-1,0)+3)//4)
    shifted = gaps-1
    # Stream VByte always uses >=1 data byte, with two control bits/value.
    stream_widths = np.where(shifted<=255,1,np.where(shifted<=65535,2,np.where(shifted<=16777215,3,4)))
    stream = common+first+control+np.sum(stream_widths[gap_valid])
    # Band et al., Section 3.1 / Table 1: 0/1 require no payload;
    # [2,256) requires one byte, and [256,2^32) requires four.
    band_widths = np.where(shifted<=1,0,np.where(shifted<=255,1,4))
    band = common+first+control+np.sum(band_widths[gap_valid])
    row_byte_escape = common+first+np.count_nonzero(gap_valid)+4*np.count_nonzero((gaps>=255)&gap_valid)

    # Nibble scheme from Thaler/Keller Section II-B, applied to full particle
    # lists, not to clusters. Partial masks occupy one uint32, like our existing
    # cluster codec. Info and payload nibble streams round independently.
    rounded = (width-1+31)//32*32
    padding = ((0,0),(0,rounded-(width-1)))
    valid_blocks = np.pad(gap_valid,padding).reshape(rows,-1,32)
    gap_blocks = np.pad(gaps,padding).reshape(rows,-1,32)
    non_unity = (gap_blocks!=1)&valid_blocks
    large = (gap_blocks>9)&valid_blocks
    nibble_lengths = np.floor(np.log2(np.maximum(gap_blocks,1))).astype(np.int64)//4+1
    info = (non_unity.sum(2)+1)//2
    payload = ((nibble_lengths*large).sum(2)+1)//2
    nibble = common+first+np.sum(4*np.any(valid_blocks,axis=2)+info+payload)

    # Actual independent chunk32 byte layout, including padding, bases, uint32
    # payload offsets and per-row int64 chunk offsets. No per-edge CPU scans are
    # added to any runtime operation; this is a standalone distribution model.
    chunks = int(np.sum((counts+31)//32))
    within_chunk = (columns+1)%32 != 0
    escape_count = int(np.count_nonzero((gaps>=255)&gap_valid&within_chunk))
    chunk32 = common+4+40*chunks+4*escape_count
    contiguous = np.all((gap_blocks==1)|~valid_blocks,axis=2)&np.any(valid_blocks,axis=2)
    edges = int(counts.sum())
    totals = dict(csr32=4*edges+common,csr64=8*edges+common,stream_vbyte_model=int(stream),
                  band_model=int(band),nibble_full_list_model=int(nibble),
                  row_byte_escape_model=int(row_byte_escape),implemented_chunk32_model=int(chunk32))
    return dict(sampled_rows=rows,sampled_edges=edges,bytes_per_edge={key:value/edges for key,value in totals.items()},
                bytes_per_particle={key:value/rows for key,value in totals.items()},
                successive_gap_one_fraction=float(np.mean(gaps[gap_valid]==1)),
                successive_gap_one_or_two_fraction=float(np.mean(gaps[gap_valid]<=2)),
                remaining_gap_segments_all_unity_fraction=float(contiguous.sum()/np.any(valid_blocks,axis=2).sum()),
                note='Storage estimates including 12 B/row metadata; models do not implement or measure GPU decoding. Nibble model is applied to full lists, not shared cluster lists.')


def verify_models():
    # Hand-counted all-consecutive row: first uint32 + eight control bytes for
    # Band; first uint32 + one uint32 mask for nibble; add 20 bytes row metadata.
    consecutive=np.arange(100,132,dtype=np.int64)[None,:]
    result=storage_models(consecutive,np.array([32]))
    assert result['bytes_per_edge']['band_model']*32 == 32
    assert result['bytes_per_edge']['nibble_full_list_model']*32 == 28
    # All 31 within-chunk gaps need escapes; ensure 254/255 and 256/257 differ
    # at the actual byte-escape and Band boundaries respectively.
    for gap,band_data,byte_extra in ((1,0,0),(2,0,0),(3,31,0),(254,31,0),(255,31,124),(256,31,124),(257,124,124)):
        ids=(np.arange(32,dtype=np.int64)*gap)[None,:]
        result=storage_models(ids,np.array([32]))
        assert result['bytes_per_edge']['band_model']*32 == 20+4+8+band_data
        assert result['bytes_per_edge']['implemented_chunk32_model']*32 == 20+4+40+byte_extra
