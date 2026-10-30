#pragma once
#include "math/grid.h"

namespace sph_native::compact {
// Reusable candidate-cell traversal for radius search and frontend operators.
// A short periodic axis visits each cell once. Hash collisions are resolved
// using the full int64 Cartesian cell key, never by distance or hash equality.
template <class T, int D> class CellTraversal {
    static_assert(D >= 1 && D <= 3, "Cell traversal requires dimension 1, 2 or 3");
    View index_;
    int64_t first_[3]{}, lengths_[3]{1, 1, 1};
    int64_t cursor_ = 0, count_ = 1;
    bool valid_ = true;

  public:
    SPH_HD CellTraversal(const T *position, View index) : index_(index) {
        int64_t center[D];
        valid_ = coordinates<T, D>(position, index_.grid, center);
        if (!valid_) {
            count_ = 0;
            return;
        }
        for (int axis = 0; axis < D; ++axis) {
            const auto cells = index_.grid.cells[axis];
            if (index_.grid.periodic[axis]) {
                first_[axis] = cells <= 2 ? 0 : center[axis] - 1;
                lengths_[axis] = cells < 3 ? cells : 3;
            } else {
                first_[axis] = center[axis] > 0 ? center[axis] - 1 : 0;
                const auto last = center[axis] + 1 < cells ? center[axis] + 1 : cells - 1;
                lengths_[axis] = last - first_[axis] + 1;
            }
            count_ *= lengths_[axis];
        }
    }
    SPH_HD bool valid() const { return valid_; }
    SPH_HD bool next(Cell &entry) {
        while (cursor_ < count_) {
            int64_t cell[D], remainder = cursor_++;
            for (int axis = 0; axis < D; ++axis) {
                cell[axis] = first_[axis] + remainder % lengths_[axis];
                remainder /= lengths_[axis];
                if (cell[axis] < 0)
                    cell[axis] += index_.grid.cells[axis];
                if (cell[axis] >= index_.grid.cells[axis])
                    cell[axis] -= index_.grid.cells[axis];
            }
            const auto key = linear_key(cell, index_.grid);
            const auto bucket = index_.buckets[hash_key(cell, index_.grid)];
            int64_t begin = bucket.begin, end = bucket.end;
            while (begin < end) {
                const auto middle = begin + (end - begin) / 2;
                if (index_.cells[middle].key < key)
                    begin = middle + 1;
                else
                    end = middle;
            }
            if (begin != bucket.end && index_.cells[begin].key == key) {
                entry = index_.cells[begin];
                return true;
            }
        }
        return false;
    }
};
} // namespace sph_native::compact
