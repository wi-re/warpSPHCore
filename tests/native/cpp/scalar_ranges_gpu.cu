#include "gpu_storage.h"
#include "scalar_ranges.h"
#include <iostream>
template<class T>__global__ void scalar_ranges(int* out){*out=scalar_range_cases<T>();}
int main(){int devices=0;if(GPU(GetDeviceCount)(&devices)!=GPU(Success)||!devices)return 77;Storage storage;
 auto* result=static_cast<int*>(storage.allocate(2*sizeof(int)));scalar_ranges<float><<<1,1,0,storage.stream>>>(result);scalar_ranges<double><<<1,1,0,storage.stream>>>(result+1);check(GPU(GetLastError)());std::vector<int> out(2);storage.download(out,{result,2});
 if(out[0]||out[1]){std::cerr<<"GPU scalar range check failed: "<<out[0]<<" "<<out[1]<<"\n";return 1;}std::cout<<"GPU finite quotient products and scaled symmetric inverses passed FP32/FP64\n";
}
