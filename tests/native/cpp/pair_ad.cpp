// A shared input receives cancelling bars from independently seeded outputs.
#include "math/pair_ad.h"
#include <stdexcept>
#include <iostream>
using namespace sph_native;
static float pullback(bool widen){
    pair_ad::Trace<float,32> trace;
    using S=pair_ad::Number<float,true,32>;
    const S x=S::leaf(1.f,0.f,&trace,0,0,true);
    const S positive=x*S(33554432.f),unit=x*S(1.f),negative=x*S(-33554432.f);
    if(positive.v!=33554432.f||unit.v!=1.f||negative.v!=-33554432.f)
        throw std::runtime_error("Accumulator changed primal values");
    if(widen){
        double bars[32]={};bars[positive.node]=bars[unit.node]=bars[negative.node]=1.;
        trace.reverse(bars);return float(bars[x.node]);
    }
    positive.seed(1.f);unit.seed(1.f);negative.seed(1.f);trace.reverse();
    return trace.nodes[x.node].bar;
}
int main(){
    // d[(2^25*x), x, (-2^25*x)]^T [1,1,1] / dx = 1.
    if(pullback(false)!=0.f||pullback(true)!=1.f)
        throw std::runtime_error("Pair AD cancellation check failed");
    std::cout<<"Pair AD: default arithmetic preserved; wider bars recover unit sensitivity\n";
}
