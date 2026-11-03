#pragma once
#include "math/distance.h"
namespace sph_native {
// Fixed-size value types for native pair algebra. Storage is an aggregate;
// numerical kernels own the tensor views and perform explicit loads/stores.
template<class T,int N> struct SmallVector {
    T values[N]={};
    SPH_HD T& operator[](int i) {return values[i];}
    SPH_HD T operator[](int i) const {return values[i];}
    SPH_HD SmallVector& operator+=(const SmallVector& other) {
        for (int i=0;i<N;++i) values[i]+=other[i];return *this;
    }
};
template<class T,int N> SPH_HD SmallVector<T,N> operator+(SmallVector<T,N> a,const SmallVector<T,N>& b) {a+=b;return a;}
template<class T,int N> SPH_HD SmallVector<T,N> operator-(SmallVector<T,N> a,const SmallVector<T,N>& b) {for(int i=0;i<N;++i)a[i]-=b[i];return a;}
template<class T,int N> SPH_HD SmallVector<T,N> operator-(SmallVector<T,N> a) {for(int i=0;i<N;++i)a[i]=-a[i];return a;}
template<class T,int N> SPH_HD SmallVector<T,N> operator*(SmallVector<T,N> a,T b) {for(int i=0;i<N;++i)a[i]*=b;return a;}
template<class T,int N> SPH_HD SmallVector<T,N> operator*(T b,SmallVector<T,N> a) {return a*b;}
template<class T,int N> SPH_HD SmallVector<T,N> operator/(SmallVector<T,N> a,T b) {for(int i=0;i<N;++i)a[i]/=b;return a;}
template<class T,int N> SPH_HD T dot(const SmallVector<T,N>& a,const SmallVector<T,N>& b) {T s=0;for(int i=0;i<N;++i)s+=a[i]*b[i];return s;}
template<class T,int N> SPH_HD T length(const SmallVector<T,N>& a) {return scalar_sqrt(dot(a,a));}
template<class T,int N> SPH_HD SmallVector<T,N> load_vector(const T* values,int64_t row) {
    SmallVector<T,N> result{};for(int i=0;i<N;++i)result[i]=values[row*N+i];return result;
}
template<class T,int N> SPH_HD void store_vector(T* values,int64_t row,const SmallVector<T,N>& value) {for(int i=0;i<N;++i)values[row*N+i]=value[i];}
template<class T,int D> struct SmallMatrix {
    T values[D*D]={};
    SPH_HD T& operator()(int i,int j) {return values[i*D+j];}
    SPH_HD T operator()(int i,int j) const {return values[i*D+j];}
    SPH_HD SmallMatrix& operator+=(const SmallMatrix& other) {for(int i=0;i<D*D;++i)values[i]+=other.values[i];return *this;}
};
template<class T,int D> SPH_HD SmallMatrix<T,D> load_matrix(const T* values,int64_t row) {SmallMatrix<T,D> m{};for(int i=0;i<D*D;++i)m.values[i]=values[row*D*D+i];return m;}
template<class T,int D> SPH_HD void store_matrix(T* values,int64_t row,const SmallMatrix<T,D>& m) {for(int i=0;i<D*D;++i)values[row*D*D+i]=m.values[i];}
template<class T,int D> SPH_HD SmallMatrix<T,D> operator*(SmallMatrix<T,D> m,T scale) {for(int i=0;i<D*D;++i)m.values[i]*=scale;return m;}
template<class T,int D> SPH_HD SmallVector<T,D> operator*(const SmallMatrix<T,D>& m,const SmallVector<T,D>& v) {SmallVector<T,D> out{};for(int i=0;i<D;++i)for(int j=0;j<D;++j)out[i]+=m(i,j)*v[j];return out;}
template<class T,int D> SPH_HD SmallMatrix<T,D> operator*(const SmallMatrix<T,D>& a,const SmallMatrix<T,D>& b) {SmallMatrix<T,D> out{};for(int i=0;i<D;++i)for(int j=0;j<D;++j)for(int k=0;k<D;++k)out(i,j)+=a(i,k)*b(k,j);return out;}
template<class T,int D> SPH_HD SmallMatrix<T,D> outer(const SmallVector<T,D>& a,const SmallVector<T,D>& b) {SmallMatrix<T,D> out{};for(int i=0;i<D;++i)for(int j=0;j<D;++j)out(i,j)=a[i]*b[j];return out;}
} // namespace sph_native
