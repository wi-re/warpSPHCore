// Real reference/edge addresses above int32, without resident multi-GB arrays.
#include "api.h"
#include "execution/neighborhood.h"
#include <cstdint>
#include <cmath>
#include <iostream>
#include <stdexcept>
#ifdef __linux__
#include <sys/mman.h>
template<class T>struct Sparse {
    int64_t count;T* data;
    explicit Sparse(int64_t n):count(n),data(static_cast<T*>(mmap(nullptr,size_t(n)*sizeof(T),
        PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0))) {
        if(data==MAP_FAILED)throw std::bad_alloc();
    }
    ~Sparse(){munmap(data,size_t(count)*sizeof(T));}
    sph_native::ConstBuffer view()const{return {data,count};}
};
#endif
int main(){
#ifndef __linux__
    return 77;
#else
    if(sizeof(void*)<8)return 77;
    using namespace sph_native;
    try {
        const int64_t n=int64_t(INT32_MAX)+4,begin=int64_t(INT32_MAX)+8,high=n-1;
        constexpr int D=3;
        Sparse<float> positions(n*D),supports(n),mass(n),density(n),values(n);
        Sparse<int32_t> kinds(n);Sparse<int64_t> ids(begin+2);
        for(auto j:{int64_t(1),high}){
            for(int d=0;d<D;++d)positions.data[j*D+d]=.5f;
            positions.data[j*D]=j==high?.4f:.6f;
            supports.data[j]=mass.data[j]=density.data[j]=1.f;
            values.data[j]=j==high?2.f:.5f;
        }
        ids.data[begin]=high;ids.data[begin+1]=1;
        const float qx[]={.5f,.5f,.5f},one[]={1.f},lo[]={0.f,0.f,0.f},hi[]={1.f,1.f,1.f};
        const bool periodic[]={false,false,false};const int32_t qkind[]={0},counts[]={2};
        const int64_t offsets[]={begin};float result=0.f;
        Problem p{};p.precision=Precision::Float32;p.dimension=D;
        p.query={1,{qx,D},{one,1},{one,1},{one,1},{qkind,1}};
        p.reference={n,positions.view(),supports.view(),mass.view(),density.view(),kinds.view()};
        p.domain={{lo,D},{hi,D},{periodic,D}};
        p.reference_values=values.view();p.adjacency={{offsets,1},{counts,1},ids.view()};
        p.properties.operation=Operation::Interpolate;
        Context context{};context.cpu_threads=1;forward(context,p,{&result,1});
        const float expected=2.5f*float(21./(2*3.141592653589793))*std::pow(.9f,4)*1.4f;
        if(std::abs(result-expected)>3e-5f+3e-5f*std::abs(expected))
            throw std::runtime_error("Wide standalone interpolation differs from independent radial equation");
        // Fixed CSR admission must not silently remove search's separate
        // generated-count limit. It rejects before touching unmapped rows.
        int32_t generated=-7;bool rejected=false;
        try{radius_count(context,p,{&generated,1});}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected||generated!=-7)throw std::runtime_error("Wide search count admission changed");
        neighborhood::Maximum<float> maximum{};
        maximum.consider(2.f,high);maximum.consider(.5f,1);
        if(maximum.index!=high||maximum.value!=2.f)throw std::runtime_error("Narrowed maximum reference ID");
        maximum.consider(2.f,1);
        if(maximum.index!=1)throw std::runtime_error("Maximum tie must retain the smaller wide ID");
        std::cout<<"Wide fixed CSR: real int64 references, offsets, 3D pointer products and maximum IDs passed\n";
        return 0;
    }catch(const std::bad_alloc&){std::cerr<<"Insufficient virtual address space for sparse wide test\n";return 77;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
#endif
}
