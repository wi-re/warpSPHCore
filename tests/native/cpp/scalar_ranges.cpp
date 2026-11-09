#include "scalar_ranges.h"
#include <iostream>
int main(){const int a=scalar_range_cases<float>(),b=scalar_range_cases<double>();if(a||b){std::cerr<<"Scalar range check failed: "<<a<<" "<<b<<"\n";return 1;}std::cout<<"Finite quotient products and scaled symmetric inverses passed FP32/FP64\n";}
