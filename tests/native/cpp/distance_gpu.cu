#include "gpu_storage.h"
#include "math/distance.h"
#include "math/queryGeometry.h"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

template<class T,int D> __global__ void distances(const T* x,const T* y,T* out){
    const int i=threadIdx.x;T lo[D],hi[D],direct[D],prepared[D];bool periodic[D];
    for(int d=0;d<D;++d){lo[d]=T(-.5-d);hi[d]=lo[d]+(d==2?T(.75):T(1));periodic[d]=d!=1;}
    out[i*(2*D+2)]=sph_native::minimum_image_displacement<T,D>(x+i*D,y+i*D,lo,hi,periodic,direct);
    sph_native::QueryGeometry<T,D> geometry(x+i*D,lo,hi,periodic);
    out[i*(2*D+2)+1]=geometry.distance(y+i*D,prepared);
    for(int d=0;d<D;++d){out[i*(2*D+2)+2+d]=direct[d];out[i*(2*D+2)+2+D+d]=prepared[d];}
}
template<class T,int D>void cases(){
    const T a=T(.25),b=std::nextafter(a,std::numeric_limits<T>::infinity()),below=std::nextafter(T(.5),T(0));
    const std::array<T,14> xs={b,a,T(1e-12),T(-1e-12),T(.99),T(.01),T(.5),T(-.5),below,-below,T(4.25),T(-7.75),T(0x1p55),T(-0x1p55)};
    const std::array<T,14> ys={a,b,T(0),T(0),T(.01),T(.99),T(0),T(0),T(0),T(0),T(-.125),T(.125),T(.25),T(.25)};
    std::vector<T> x(xs.size()*D),y(x.size()),out(xs.size()*(2*D+2));
    for(size_t i=0;i<xs.size();++i)for(int d=0;d<D;++d){x[i*D+d]=xs[i];y[i*D+d]=ys[i];}
    Storage storage;const auto dx=storage.upload(x.data(),x.size()),dy=storage.upload(y.data(),y.size());
    auto* result=static_cast<T*>(storage.allocate(out.size()*sizeof(T)));
    distances<T,D><<<1,xs.size(),0,storage.stream>>>(static_cast<const T*>(dx.data),static_cast<const T*>(dy.data),result);
    check(GPU(GetLastError)());storage.download(out,{result,int64_t(out.size())});
    for(size_t i=0;i<xs.size();++i){
        const auto offset=i*(2*D+2);if(out[offset]!=out[offset+1])throw std::runtime_error("GPU direct/prepared norms disagree");
        for(int d=0;d<D;++d){
            long double expected=(long double)xs[i]-(long double)ys[i];
            if(d!=1){const long double span=d==2?.75L:1.L,half=span/2;expected=std::fmod(expected,span);if(expected>=half)expected-=span;else if(expected< -half)expected+=span;}
            const T actual=out[offset+2+d];
            if(actual!=out[offset+2+D+d])throw std::runtime_error("GPU direct/prepared vectors disagree");
            if(std::abs((long double)actual-(long double)T(expected))>2*std::numeric_limits<T>::epsilon())throw std::runtime_error("GPU displacement disagrees with independent oracle");
            if(i<4&&actual!=xs[i]-ys[i])throw std::runtime_error("GPU local displacement bits lost");
        }
    }
}
int main(){int devices=0;if(GPU(GetDeviceCount)(&devices)!=GPU(Success)||!devices)return 77;
    try{cases<float,1>();cases<float,2>();cases<float,3>();cases<double,1>();cases<double,2>();cases<double,3>();}
    catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
    std::cout<<"GPU direct/prepared geometry: local ulps, seams, ties and large images in FP32/FP64, 1D-3D passed\n";
}
