#pragma once
#include "config.h"
#include <cmath>

namespace sph_native {
// One-sided Jacobi SVD for real square matrices of order <= 3. Scaling before
// the rotations avoids squaring the condition number as A^T A would do.
template<class T,int D> SPH_HD void small_pinv(const T* input,T rtol,T* output) {
    T scale=0;
    for(int k=0;k<D*D;++k)if(std::abs(input[k])>scale)scale=std::abs(input[k]);
    for(int k=0;k<D*D;++k)output[k]=T(0);
    if(scale==T(0))return;
    T columns[D*D],vectors[D*D]={};
    for(int k=0;k<D*D;++k)columns[k]=input[k]/scale;
    for(int k=0;k<D;++k)vectors[k*D+k]=T(1);
    for(int sweep=0;sweep<12;++sweep)for(int p=0;p<D;++p)for(int q=p+1;q<D;++q) {
        T a=0,b=0,c=0;
        for(int k=0;k<D;++k){a+=columns[k*D+p]*columns[k*D+p];b+=columns[k*D+q]*columns[k*D+q];c+=columns[k*D+p]*columns[k*D+q];}
        if(c==T(0))continue;
        const T theta=(b-a)/(T(2)*c);
        const T tangent=(theta<T(0)?T(-1):T(1))/(std::abs(theta)+std::hypot(theta,T(1)));
        const T cosine=T(1)/std::sqrt(T(1)+tangent*tangent),sine=tangent*cosine;
        for(int k=0;k<D;++k) {
            const T x=columns[k*D+p],y=columns[k*D+q];
            columns[k*D+p]=cosine*x-sine*y;columns[k*D+q]=sine*x+cosine*y;
            const T v=vectors[k*D+p],w=vectors[k*D+q];
            vectors[k*D+p]=cosine*v-sine*w;vectors[k*D+q]=sine*v+cosine*w;
        }
    }
    T norms[D]={},largest=0;
    for(int k=0;k<D;++k) {
        for(int i=0;i<D;++i)norms[k]+=columns[i*D+k]*columns[i*D+k];
        norms[k]=std::sqrt(norms[k]);if(norms[k]>largest)largest=norms[k];
    }
    for(int k=0;k<D;++k) {
        if(norms[k]<=rtol*largest||norms[k]==T(0))continue;
        // Divide in stages: tiny singular values need not be squared in the
        // original units, and the final inverse can still be representable.
        for(int i=0;i<D;++i)for(int j=0;j<D;++j)
            output[i*D+j]+=(vectors[i*D+k]/norms[k])*(columns[j*D+k]/norms[k])/scale;
    }
}
// Moore-Penrose product rules used by Torch's real pinv AD contract. They
// differentiate the selected rank, not the Jacobi rotations or rank decision.
template<class T,int D> SPH_HD void small_pinv_jvp(const T* a,const T* p,const T* da,T* dp) {
    T u[D*D]={},v[D*D]={},w[D*D]={},z[D*D]={};
    for(int i=0;i<D;++i)for(int j=0;j<D;++j) {
        v[i*D+j]=w[i*D+j]=i==j?T(1):T(0);
        for(int k=0;k<D;++k) {
            u[i*D+j]+=p[i*D+k]*p[j*D+k];z[i*D+j]+=p[k*D+i]*p[k*D+j];
            v[i*D+j]-=a[i*D+k]*p[k*D+j];w[i*D+j]-=p[i*D+k]*a[k*D+j];
        }
    }
    for(int i=0;i<D;++i)for(int j=0;j<D;++j) {
        T value=0;
        for(int k=0;k<D;++k)for(int l=0;l<D;++l)
            value+=-p[i*D+k]*da[k*D+l]*p[l*D+j]+u[i*D+k]*da[l*D+k]*v[l*D+j]+w[i*D+k]*da[l*D+k]*z[l*D+j];
        dp[i*D+j]=value;
    }
}
template<class T,int D> SPH_HD void small_pinv_vjp(const T* a,const T* p,const T* bar,T* out) {
    // Transpose the tiny Jacobian without a shared reduction or heap storage.
    for(int k=0;k<D*D;++k) {
        T basis[D*D]={},column[D*D];basis[k]=T(1);small_pinv_jvp<T,D>(a,p,basis,column);
        T value=0;for(int l=0;l<D*D;++l)value+=bar[l]*column[l];out[k]=value;
    }
}
}
