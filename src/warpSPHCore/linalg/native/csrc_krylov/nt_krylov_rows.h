#pragma once
#include "torch/types.h"
#include "math/distance.h"
namespace sph_native::krylov {
// The state machines match the readable Warp kernels. Each row has one writer;
// only exceptional vector flags need integer atomic maxima.
template<class T> SPH_HD bool finite(T value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}
template<class T> SPH_HD T absolute(T value) { return value<T(0) ? -value:value; }
template<class T> SPH_HD T safe_divide(T a,T b) { return finite(b) && b!=T(0) ? a/b:a; }
SPH_HD inline void atomic_max(int32_t* pointer,int32_t value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    atomicMax(pointer,value);
#else
    int32_t previous=__atomic_load_n(pointer,__ATOMIC_RELAXED);
    while (previous<value && !__atomic_compare_exchange_n(pointer,&previous,value,false,__ATOMIC_RELAXED,__ATOMIC_RELAXED)) {}
#endif
}

template<class T> struct cg_initialize {
    T* rr;
    T* target;
    int32_t* state;
    T* norms;
    bool* valid;
    bool* initialThreshold;
    bool* zeroRhs;
    bool history;
    SPH_HD void operator()(int64_t thread) const {
        auto norm = scalar_sqrt(rr[0]);
        auto conv = (finite(norm) && (norm <= target[0]));
        auto live = ((!conv) && (!initialThreshold[0]));
        state[0] = int32_t(live);
        state[1] = (-14);
        if (conv) {
            state[1] = 0;
        }
        if (initialThreshold[0]) {
            state[1] = (-12);
        }
        state[2] = 0;
        state[3] = 0;
        state[4] = 0;
        if (history) {
            norms[0] = norm;
            valid[0] = (!(zeroRhs[0] && conv));
        }
    }
};

template<class T> struct cg_candidate {
    T* x;
    T* r;
    T* p;
    T* ap;
    T* rz;
    T* pap;
    T* xn;
    T* rn;
    int32_t* flags;
    T threshold;
    bool hasThreshold;
    int dim;
    SPH_HD void operator()(int64_t thread) const {
        auto i = thread;
        auto alpha = T(0);
        if ((finite(pap[0]) && (pap[0] != T(0)))) {
            alpha = (rz[0] / pap[0]);
        }
        xn[i] = (x[i] + (alpha * p[i]));
        rn[i] = (r[i] - (alpha * ap[i]));
        if (((!finite(xn[i])) || (!finite(rn[i])))) {
            atomic_max(flags+0, 1);
        }
        if ((hasThreshold && ((i % dim) == 0))) {
            auto length2 = T(0);
            for (int j=0; j<dim; ++j) {
                auto value = (x[(i + j)] + (alpha * p[(i + j)]));
                length2 += (value * value);
            }
            if ((scalar_sqrt(length2) > T(threshold))) {
                atomic_max(flags+1, 1);
            }
        }
    }
};

template<class T> struct cg_decide {
    T* rz;
    T* pap;
    T* rr;
    T* target;
    int32_t* flags;
    int32_t* state;
    T* norms;
    bool* valid;
    bool history;
    SPH_HD void operator()(int64_t thread) const {
        auto norm = scalar_sqrt(rr[0]);
        auto live = (state[0] != 0);
        auto ok = (finite(rz[0]) && (rz[0] > T(0)) && finite(pap[0]) && (pap[0] > T(0)) && finite(norm) && (flags[0] == 0));
        auto go = (live && ok);
        auto conv = (go && (norm <= target[0]) && (flags[1] == 0));
        if ((live && (!ok))) {
            state[1] = (-16);
        } else {
            if ((go && (flags[1] != 0))) {
                state[1] = (-12);
            } else {
                if (conv) {
                    state[1] = state[3];
                }
            }
        }
        state[0] = int32_t((go && (!conv) && (flags[1] == 0)));
        state[4] = int32_t(go);
        state[2] += int32_t(go);
        state[3] += 1;
        if (history) {
            norms[state[3]] = norm;
            valid[state[3]] = go;
        }
    }
};

template<class T> struct cg_commit {
    T* x;
    T* r;
    T* xn;
    T* rn;
    int32_t* state;
    SPH_HD void operator()(int64_t thread) const {
        auto i = thread;
        if ((state[4] != 0)) {
            x[i] = xn[i];
            r[i] = rn[i];
        }
    }
};

template<class T> struct cg_diagonal {
    T* r;
    T* inverse;
    T* z;
    int64_t inverseStride;
    SPH_HD void operator()(int64_t thread) const {
        auto i = thread;
        z[i] = (r[i] * inverse[i*inverseStride]);
    }
};

template<class T> struct cg_direction {
    T* z;
    T* p;
    T* rz;
    T* rzn;
    int32_t* state;
    SPH_HD void operator()(int64_t thread) const {
        auto i = thread;
        if ((state[0] != 0)) {
            auto beta = T(0);
            if ((finite(rz[0]) && (rz[0] != T(0)))) {
                beta = (rzn[0] / rz[0]);
            }
            p[i] = (z[i] + (beta * p[i]));
        }
    }
};

template<class T> struct bicgstab_initialize {
    T* rr;
    T* target;
    T* r0norm;
    T* rnorm;
    int32_t* state;
    bool* threshold;
    SPH_HD void operator()(int64_t thread) const {
        auto norm = scalar_sqrt(rr[0]);
        r0norm[0] = norm;
        rnorm[0] = norm;
        auto conv = (finite(norm) && (norm <= target[0]));
        state[0] = int32_t(((!conv) && (!threshold[0])));
        state[1] = (-14);
        if (conv) {
            state[1] = 0;
        }
        if (threshold[0]) {
            state[1] = (-12);
        }
        for (int i=2; i<9; ++i) {
            state[i] = 0;
        }
    }
};

template<class T> struct bicgstab_alpha {
    T* rho;
    T* rv;
    T* aa;
    T* r0norm;
    T* rnorm;
    T* alpha;
    int32_t* state;
    SPH_HD void operator()(int64_t thread) const {
        auto eps = T(1e-06);
        auto badRho = ((!finite(rho[0])) || (rho[0] == T(0)) || (absolute(rho[0]) < ((eps * r0norm[0]) * rnorm[0])));
        auto badRv = ((!finite(rv[0])) || (rv[0] == T(0)) || (absolute(rv[0]) < ((eps * r0norm[0]) * scalar_sqrt(aa[0]))));
        alpha[0] = safe_divide(rho[0], rv[0]);
        state[4] = int32_t(((state[0] != 0) && (!badRho) && (!badRv)));
        if ((state[0] != 0)) {
            if (badRho) {
                state[1] = (-10);
            } else {
                if (badRv) {
                    state[1] = (-11);
                }
            }
        }
    }
};

template<class T> struct bicgstab_halfCandidate {
    T* x;
    T* r;
    T* z;
    T* ap;
    T* alpha;
    T* candidate;
    T* s;
    int32_t* flags;
    T threshold;
    bool hasThreshold;
    int dim;
    SPH_HD void operator()(int64_t thread) const {
        auto i = thread;
        auto value = (x[i] + (alpha[0] * z[i]));
        candidate[i] = value;
        s[i] = (r[i] - (alpha[0] * ap[i]));
        if ((!finite(value))) {
            atomic_max(flags+0, 1);
        }
        if ((hasThreshold && ((i % dim) == 0))) {
            auto length2 = T(0);
            for (int k=0; k<dim; ++k) {
                auto v = (x[(i + k)] + (alpha[0] * z[(i + k)]));
                length2 += (v * v);
            }
            if ((scalar_sqrt(length2) > T(threshold))) {
                atomic_max(flags+1, 1);
            }
        }
    }
};

template<class T> struct bicgstab_halfDecision {
    T* ss;
    T* target;
    int32_t* flags;
    int32_t* state;
    SPH_HD void operator()(int64_t thread) const {
        auto go = ((state[4] != 0) && (flags[0] == 0));
        auto norm = scalar_sqrt(ss[0]);
        auto half = (go && finite(norm) && (norm <= target[0]));
        state[5] = int32_t(half);
        state[6] = int32_t((go && (!half)));
        if (((state[4] != 0) && (flags[0] != 0))) {
            state[1] = (-11);
        }
        if (half) {
            state[1] = state[3];
            if ((flags[1] != 0)) {
                state[1] = (-12);
            }
        }
    }
};

template<class T> struct bicgstab_omega {
    T* ts;
    T* tt;
    T* omega;
    SPH_HD void operator()(int64_t thread) const {
        omega[0] = safe_divide(ts[0], tt[0]);
    }
};

template<class T> struct bicgstab_fullCandidate {
    T* x;
    T* z;
    T* zs;
    T* ast;
    T* s;
    T* candidate;
    T* alpha;
    T* omega;
    int32_t* flags;
    int32_t* state;
    T threshold;
    bool hasThreshold;
    int dim;
    SPH_HD void operator()(int64_t thread) const {
        auto i = thread;
        auto value = ((x[i] + (alpha[0] * z[i])) + (omega[0] * zs[i]));
        if ((state[6] != 0)) {
            candidate[i] = value;
        }
        s[i] = (s[i] - (omega[0] * ast[i]));
        if ((!finite(value))) {
            atomic_max(flags+0, 1);
        }
        if ((hasThreshold && ((i % dim) == 0))) {
            auto length2 = T(0);
            for (int k=0; k<dim; ++k) {
                auto v = ((x[(i + k)] + (alpha[0] * z[(i + k)])) + (omega[0] * zs[(i + k)]));
                length2 += (v * v);
            }
            if ((scalar_sqrt(length2) > T(threshold))) {
                atomic_max(flags+1, 1);
            }
        }
    }
};

template<class T> struct bicgstab_fullDecision {
    T* rr;
    T* rhon;
    T* rho;
    T* alpha;
    T* omega;
    T* tt;
    T* beta;
    T* target;
    int32_t* flags;
    int32_t* state;
    SPH_HD void operator()(int64_t thread) const {
        auto norm = scalar_sqrt(rr[0]);
        auto bad = ((!finite(omega[0])) || (omega[0] == T(0)) || (!finite(tt[0])) || (tt[0] == T(0)));
        bad = (bad || (flags[0] != 0) || (!finite(norm)));
        auto full = ((state[6] != 0) && (!bad));
        auto conv = (full && (norm <= target[0]));
        state[7] = int32_t(full);
        state[8] = int32_t((full && (flags[1] == 0) && (!conv)));
        beta[0] = (safe_divide(rhon[0], rho[0]) * safe_divide(alpha[0], omega[0]));
        if (((state[6] != 0) && bad)) {
            state[1] = (-11);
        }
        if (full) {
            if ((flags[1] != 0)) {
                state[1] = (-12);
            } else {
                if (conv) {
                    state[1] = (state[3] + 1);
                }
            }
        }
    }
};

template<class T> struct bicgstab_direction {
    T* x;
    T* candidate;
    T* s;
    T* p;
    T* ap;
    T* pn;
    T* beta;
    T* omega;
    int32_t* flags;
    int32_t* state;
    SPH_HD void operator()(int64_t thread) const {
        auto i = thread;
        auto value = (s[i] + (beta[0] * (p[i] - (omega[0] * ap[i]))));
        pn[i] = value;
        if (((state[8] != 0) && ((!finite(value)) || (!finite(beta[0]))))) {
            atomic_max(flags+2, 1);
        }
        if (((state[5] != 0) || (state[7] != 0))) {
            x[i] = candidate[i];
        }
    }
};

template<class T> struct bicgstab_finish {
    T* rr;
    T* rnorm;
    int32_t* flags;
    int32_t* state;
    T* norms;
    bool* valid;
    bool history;
    SPH_HD void operator()(int64_t thread) const {
        if (((state[8] != 0) && (flags[2] != 0))) {
            state[1] = (-11);
        }
        state[0] = int32_t(((state[8] != 0) && (flags[2] == 0)));
        state[2] += int32_t(((state[5] != 0) || (state[7] != 0)));
        if (history) {
            norms[state[3]] = scalar_sqrt(rr[0]);
            valid[state[3]] = (state[7] != 0);
        }
        if ((state[0] != 0)) {
            rnorm[0] = scalar_sqrt(rr[0]);
        }
        state[3] += 1;
    }
};

template<class T> struct bicgstab_commit {
    T* r;
    T* p;
    T* s;
    T* pn;
    int32_t* state;
    SPH_HD void operator()(int64_t thread) const {
        auto i = thread;
        if ((state[0] != 0)) {
            r[i] = s[i];
            p[i] = pn[i];
        }
    }
};

template<class T> struct bicgstab_diagonal {
    T* v;
    T* inverse;
    T* out;
    int64_t inverseStride;
    SPH_HD void operator()(int64_t thread) const {
        auto i = thread;
        out[i] = (v[i] * inverse[i*inverseStride]);
    }
};

template<class T,class Launch> void execute_typed(const Tensors& arrays,int task,int64_t size,
        T threshold,bool hasThreshold,int dim,bool history,const Launch& launch) {
    switch(task) {
    case 0:
        launch(size,cg_initialize<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<int32_t>(), arrays[3].mutable_data_ptr<T>(), arrays[4].mutable_data_ptr<bool>(), arrays[5].mutable_data_ptr<bool>(), arrays[6].mutable_data_ptr<bool>(), history});
        break;
    case 1:
        launch(size,cg_candidate<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>(), arrays[3].mutable_data_ptr<T>(), arrays[4].mutable_data_ptr<T>(), arrays[5].mutable_data_ptr<T>(), arrays[6].mutable_data_ptr<T>(), arrays[7].mutable_data_ptr<T>(), arrays[8].mutable_data_ptr<int32_t>(), threshold, hasThreshold, dim});
        break;
    case 2:
        launch(size,cg_decide<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>(), arrays[3].mutable_data_ptr<T>(), arrays[4].mutable_data_ptr<int32_t>(), arrays[5].mutable_data_ptr<int32_t>(), arrays[6].mutable_data_ptr<T>(), arrays[7].mutable_data_ptr<bool>(), history});
        break;
    case 3:
        launch(size,cg_commit<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>(), arrays[3].mutable_data_ptr<T>(), arrays[4].mutable_data_ptr<int32_t>()});
        break;
    case 4:
        launch(size,cg_diagonal<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>(), arrays[1].stride(0)});
        break;
    case 5:
        launch(size,cg_direction<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>(), arrays[3].mutable_data_ptr<T>(), arrays[4].mutable_data_ptr<int32_t>()});
        break;
    case 6:
        launch(size,bicgstab_initialize<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>(), arrays[3].mutable_data_ptr<T>(), arrays[4].mutable_data_ptr<int32_t>(), arrays[5].mutable_data_ptr<bool>()});
        break;
    case 7:
        launch(size,bicgstab_alpha<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>(), arrays[3].mutable_data_ptr<T>(), arrays[4].mutable_data_ptr<T>(), arrays[5].mutable_data_ptr<T>(), arrays[6].mutable_data_ptr<int32_t>()});
        break;
    case 8:
        launch(size,bicgstab_halfCandidate<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>(), arrays[3].mutable_data_ptr<T>(), arrays[4].mutable_data_ptr<T>(), arrays[5].mutable_data_ptr<T>(), arrays[6].mutable_data_ptr<T>(), arrays[7].mutable_data_ptr<int32_t>(), threshold, hasThreshold, dim});
        break;
    case 9:
        launch(size,bicgstab_halfDecision<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<int32_t>(), arrays[3].mutable_data_ptr<int32_t>()});
        break;
    case 10:
        launch(size,bicgstab_omega<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>()});
        break;
    case 11:
        launch(size,bicgstab_fullCandidate<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>(), arrays[3].mutable_data_ptr<T>(), arrays[4].mutable_data_ptr<T>(), arrays[5].mutable_data_ptr<T>(), arrays[6].mutable_data_ptr<T>(), arrays[7].mutable_data_ptr<T>(), arrays[8].mutable_data_ptr<int32_t>(), arrays[9].mutable_data_ptr<int32_t>(), threshold, hasThreshold, dim});
        break;
    case 12:
        launch(size,bicgstab_fullDecision<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>(), arrays[3].mutable_data_ptr<T>(), arrays[4].mutable_data_ptr<T>(), arrays[5].mutable_data_ptr<T>(), arrays[6].mutable_data_ptr<T>(), arrays[7].mutable_data_ptr<T>(), arrays[8].mutable_data_ptr<int32_t>(), arrays[9].mutable_data_ptr<int32_t>()});
        break;
    case 13:
        launch(size,bicgstab_direction<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>(), arrays[3].mutable_data_ptr<T>(), arrays[4].mutable_data_ptr<T>(), arrays[5].mutable_data_ptr<T>(), arrays[6].mutable_data_ptr<T>(), arrays[7].mutable_data_ptr<T>(), arrays[8].mutable_data_ptr<int32_t>(), arrays[9].mutable_data_ptr<int32_t>()});
        break;
    case 14:
        launch(size,bicgstab_finish<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<int32_t>(), arrays[3].mutable_data_ptr<int32_t>(), arrays[4].mutable_data_ptr<T>(), arrays[5].mutable_data_ptr<bool>(), history});
        break;
    case 15:
        launch(size,bicgstab_commit<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>(), arrays[3].mutable_data_ptr<T>(), arrays[4].mutable_data_ptr<int32_t>()});
        break;
    case 16:
        launch(size,bicgstab_diagonal<T>{arrays[0].mutable_data_ptr<T>(), arrays[1].mutable_data_ptr<T>(), arrays[2].mutable_data_ptr<T>(), arrays[1].stride(0)});
        break;
    }
}
template<class Launch> void execute(const Tensors& arrays,int task,int64_t size,double threshold,
        bool hasThreshold,int dimension,bool history,const Launch& launch) {
    if (arrays[0].scalar_type()==ScalarType::Float)
        execute_typed<float>(arrays,task,size,float(threshold),hasThreshold,dimension,history,launch);
    else execute_typed<double>(arrays,task,size,threshold,hasThreshold,dimension,history,launch);
}
void update_gpu(const Tensors&,int,int64_t,double,bool,int,bool);
} // namespace sph_native::krylov
