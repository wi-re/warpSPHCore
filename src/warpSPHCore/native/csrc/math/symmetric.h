#pragma once
#include "math/distance.h"

namespace sph_native {
// Jacobi rotations on a padded 3x3 matrix. This is shared numerical code:
// module wrappers choose their own rank cutoffs and low-neighbor policies.
template<class T> struct SymmetricSpectrum {
    T values[3]={}, vectors[9]={};
};
template<class T,int D> SPH_HD SymmetricSpectrum<T> symmetric_spectrum(const T* input) {
    T matrix[9]={};
    SymmetricSpectrum<T> result;
    for (int row=0;row<3;++row) {
        result.vectors[row*3+row]=T(1);
        for (int col=0;col<3;++col)
            if (row<D && col<D) matrix[row*3+col]=(input[row*D+col]+input[col*D+row])/T(2);
    }
    for (int sweep=0;sweep<8;++sweep) for (int pair=0;pair<3;++pair) {
        const int p=pair==2 ? 1:0, q=pair==0 ? 1:2;
        const T off=matrix[p*3+q];
        if (off==T(0)) continue;
        const T theta=(matrix[q*3+q]-matrix[p*3+p])/(T(2)*off);
        const T magnitude=theta<T(0) ? -theta:theta;
        const T tangent=(theta<T(0) ? T(-1):T(1))/(magnitude+scalar_sqrt(theta*theta+T(1)));
        const T cosine=T(1)/scalar_sqrt(T(1)+tangent*tangent), sine=tangent*cosine;
        T rotation[9]={},product[9]={},next[9]={},vectors[9]={};
        for (int i=0;i<3;++i) rotation[i*3+i]=T(1);
        rotation[p*3+p]=rotation[q*3+q]=cosine;
        rotation[p*3+q]=sine; rotation[q*3+p]=-sine;
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) for (int k=0;k<3;++k) {
            product[i*3+j]+=matrix[i*3+k]*rotation[k*3+j];
            vectors[i*3+j]+=result.vectors[i*3+k]*rotation[k*3+j];
        }
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) for (int k=0;k<3;++k)
            next[i*3+j]+=rotation[k*3+i]*product[k*3+j];
        for (int i=0;i<9;++i) {matrix[i]=next[i];result.vectors[i]=vectors[i];}
    }
    for (int i=0;i<3;++i) result.values[i]=matrix[i*3+i];
    return result;
}
template<class T,int D> SPH_HD void symmetric_inverse(const T* input,T rtol,T* output) {
    const auto spectrum=symmetric_spectrum<T,D>(input);
    T largest=0;
    for (T value:spectrum.values) { const T magnitude=value<T(0) ? -value:value; if (magnitude>largest) largest=magnitude; }
    for (int i=0;i<D*D;++i) output[i]=T(0);
    for (int k=0;k<3;++k) {
        const T value=spectrum.values[k],magnitude=value<T(0) ? -value:value;
        if (magnitude<=rtol*largest || magnitude==T(0)) continue;
        for (int i=0;i<D;++i) for (int j=0;j<D;++j)
            output[i*D+j]+=spectrum.vectors[i*3+k]*spectrum.vectors[j*3+k]/value;
    }
}
} // namespace sph_native
