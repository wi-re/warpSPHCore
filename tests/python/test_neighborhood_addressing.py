"""Cooperative row increments must remain wide at the int32 count boundary."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_csr_lane_increment_past_int32_max(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    include = Path(__file__).resolve().parents[2] / 'src/warpSPHCore/native/csrc'
    source = tmp_path / 'wide.cpp'
    source.write_text(r'''
#include "execution/neighborhood.h"
#include <climits>
#include <cassert>
int main() {
    // Shift a final-lane iteration onto a one-entry physical allocation. This
    // isolates increment arithmetic without allocating billions of neighbors.
    const int64_t offset = -int64_t(INT_MAX - 2);
    const int32_t count = INT_MAX, reference = 17;
    sph_native::neighborhood::Csr<int32_t> csr{&offset, &count, &reference};
    int visits = 0;
    csr.each(0, INT_MAX - 2, 8, [&](auto edge) {
        assert(++visits == 1 && edge.edge == 0 && edge.reference == 17);
    });
    assert(visits == 1);
}
''')
    binary = tmp_path / 'wide'
    subprocess.run([compiler, '-std=c++17', '-fsanitize=undefined',
                    '-fno-sanitize-recover=all', '-I', str(include), str(source),
                    '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=5)
