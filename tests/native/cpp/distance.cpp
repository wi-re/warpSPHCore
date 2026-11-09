#include "math/distance.h"
#include "math/queryGeometry.h"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

// No Torch/Warp dependency: exercise both native traversal geometries against
// long-double arithmetic on the exact same stored float/double coordinates.
template <class T, int D> void check() {
    const T a=T(.25), b=std::nextafter(a,std::numeric_limits<T>::infinity());
    const T below=std::nextafter(T(.5),T(0));
    const std::array<T,14> xs={b,a,T(1e-12),T(-1e-12),T(.99),T(.01),T(.5),T(-.5),
                              below,-below,T(4.25),T(-7.75),T(0x1p55),T(-0x1p55)};
    const std::array<T,14> ys={a,b,T(0),T(0),T(.01),T(.99),T(0),T(0),T(0),T(0),
                              T(-.125),T(.125),T(.25),T(.25)};
    T lo[D], hi[D], x[D], y[D], direct[D], prepared[D];bool periodic[D];
    for(int d=0;d<D;++d){lo[d]=T(-.5-d);hi[d]=lo[d]+(d==2?T(.75):T(1));periodic[d]=d!=1;}
    for(size_t i=0;i<xs.size();++i){
        for(int d=0;d<D;++d){x[d]=xs[i];y[d]=ys[i];}
        T rn=sph_native::minimum_image_displacement<T,D>(x,y,lo,hi,periodic,direct);
        sph_native::QueryGeometry<T,D> geometry(x,lo,hi,periodic);
        T pn=geometry.distance(y,prepared);
        if(rn!=pn)throw std::runtime_error("Direct/prepared norms disagree");
        for(int d=0;d<D;++d){
            long double expected=(long double)x[d]-(long double)y[d];
            if(periodic[d]){const long double span=d==2?.75L:1.L,half=span/2;expected=std::fmod(expected,span);if(expected>=half)expected-=span;else if(expected< -half)expected+=span;}
            if(direct[d]!=prepared[d])throw std::runtime_error("Direct/prepared vectors disagree");
            if(std::abs((long double)direct[d]-(long double)T(expected))>2*std::numeric_limits<T>::epsilon())
                throw std::runtime_error("Native displacement disagrees with independent oracle");
            if(i<4 && direct[d] != x[d]-y[d])throw std::runtime_error("Local displacement bits lost");
        }
    }
}
int main(){try{check<float,1>();check<float,2>();check<float,3>();check<double,1>();check<double,2>();check<double,3>();}
catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
std::cout<<"Native direct/prepared periodic geometry passed in 1D/2D/3D, FP32/FP64\n";}
