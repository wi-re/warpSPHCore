// Independently reduced one-dimensional force-kernel differentials.
#include "kernels/warp_derivatives.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace sph_native;
int main(){
 for(float h:{.65f,1.1f,3.f})for(float q:{.002f,.015f,.08f,.18f}){
  float x[1]={q*h},w=0,g[1]={},wx[1]={},wh=0,H[1]={},gh[1]={};
  warp_compat::kernel_derivatives<float,1>(x,x[0],h,40,w,g,wx,wh,H,gh);
  const double r=x[0],hd=h,z=r/hd,C=15/(2*3.14159265358979323846);
  const double expected=-C*6*z*(1-z)/(hd*hd*hd);
  if(std::abs(double(gh[0])-expected)>2e-6*std::abs(expected))
   throw std::runtime_error("Viscosity mixed partial lost singular cancellation");
  w=wh=0;g[0]=wx[0]=H[0]=gh[0]=0;
  warp_compat::kernel_derivatives<float,1>(x,x[0],h,35,w,g,wx,wh,H,gh);
  const double eps=float(1e-7),e2h=eps*eps*hd;
  // Retain the declared regularized Hessian, rather than replace it by zero.
  const double B=e2h*(eps*hd*hd-r*r)/((r*r*r+eps*eps*eps*hd*hd*hd)*(r+e2h));
  const double target=B*double(float(2.0684349363))*double(float(-2.154228492776))/(hd*hd);
  if(std::abs(double(H[0])-target)>2e-6*std::abs(target))
   throw std::runtime_error("Linear-core Hessian acquired subtractive noise");
 }
 std::cout<<"Conditioned force-kernel partials preserve regularization and reduced equations\n";
}
