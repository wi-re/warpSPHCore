#pragma once
#include "execution/dispatch.h"
#include "execution/numeric.h"
#include "execution/neighborhood.h"
#include "math/queryGeometry.h"
#include "math/explicit_ad.h"
#include "kernels/explicit_value_derivatives.h"
#include "nt_polyfit_kernel.h"
#include "util/directionality.h"
#include <new>

namespace sph_native::polyfit {
struct Buffers {
    Tensor exponents,entries,volumes,values,output,counts,cotangent;
    Tensors tangents,bars;
    IndexView transposeOffsets,transposeQueries;
    const int32_t* transposeCounts=nullptr;
    int basis,components,sector,sectors;
    bool rhs,subtract,unit;
    int64_t rowStart=0;
    double scale=1;
};
template<class T>SPH_HD T integer_power(T x,int n){T y=1;for(int k=0;k<n;++k)y*=x;return y;}
template<class S>SPH_HD S polynomial_power(S x,int n){
    if(n<=0)return S(1);
    // One analytic primitive keeps arbitrary polynomial order independent of
    // trace capacity. Its value still uses the reference integer multiply loop.
    using T=decltype(x.v);
    return S::unary(integer_power(x.v,n),x,T(n)*integer_power(x.v,n-1));
}
template<class S>struct Monomial{S first,second;};
template<class S>SPH_HD auto monomial(const int32_t* exponent,const S* xi,const S* dxi,int dimension){
    S value=polynomial_power(xi[0],exponent[0]);
    auto derivative=[&](int c){return exponent[c]==0?S(0):S(exponent[c])*polynomial_power(xi[c],exponent[c]-1);};
    S tangent=dxi?derivative(0)*dxi[0]:S(0);
    for(int c=1;c<dimension;++c){const S next=polynomial_power(xi[c],exponent[c]);
        if(dxi)tangent=tangent*next+value*(derivative(c)*dxi[c]);value=value*next;}
    return Monomial<S>{value,tangent};
}
// Pair-local slots distinguish source/centre field reads. The two owned row
// passes accumulate those roles into the same physical shared field buffer.
template<class T,int D>struct Contribution{
    T value=0,weight=0,bar[20][D]={};int component=0;
};
template<class T,int D>struct Pair{
    RowArgs<T,D> args;const int32_t *exponents,*entries;const T *volumes,*values;
    const T* tangents[9]={};int basis,components,sector,sectors;bool rhs,subtract,unit;
    int64_t rowStart;T scale;
    SPH_HD int width()const{return rhs?basis*components:basis*(basis+1)/2;}
    SPH_HD T tangent(int k,int64_t at)const{return tangents[k]?tangents[k][at]:T(0);}
    SPH_HD bool geometry()const{return tangents[0]||tangents[1]||tangents[2]||tangents[3]||tangents[8];}
    SPH_HD bool allowed(int64_t i,int64_t j,const T(&dx)[D])const{
        if(sector<0||i+rowStart==j)return true;
        if constexpr(D==1)return sector==0?dx[0]>T(0):dx[0]<T(0);
        else{T angle=::atan2(-dx[1],-dx[0]);if(angle<T(0))angle+=T(6.283185307179586);
            const T size=T(6.283185307179586)/T(sectors),low=T(sector)*size;
            return angle>=low&&angle<low+size;}
    }
    SPH_HD T kernel_product(const T(&x)[D],T r,T hi,T hj,const T(&dt)[D],T ti,T tj,T& w)const{
        auto one=[&](T h,T dh,T& value){LiteralKernel<T,D> k{x,r,h,args.kernel};value=k.value();return k.tangent(dt,dh);};
        if(args.support==14||args.support==15){T wi,wj;const T di=one(hi,ti,wi),dj=one(hj,tj,wj);w=(wi+wj)/T(2);return(di+dj)/T(2);}
        return one(args.support==11?hi:args.support==12?hj:args.support==13?(hi+hj)/T(2):maximum_support(hi,hj),
            args.support==11?ti:args.support==12?tj:args.support==13?(ti+tj)/T(2):hi>=hj?ti:tj,w);
    }
    SPH_HD T kernel_value_pair(const T(&x)[D],T r,T hi,T hj)const{
        auto one=[&](T h){return LiteralKernel<T,D>{x,r,h,args.kernel}.value();};
        if(args.support==14||args.support==15)return(one(hi)+one(hj))/T(2);
        return one(args.support==11?hi:args.support==12?hj:args.support==13?(hi+hj)/T(2):maximum_support(hi,hj));
    }
    SPH_HD void kernel_reverse(const T(&x)[D],T r,T hi,T hj,const T(&dt)[D],T ti,T tj,T bw,T bd,bool product,interpolate_jvp::Bars<T,D>& bars)const{
        T bx[D]={},bt[D]={},bhi=0,bhj=0,bti=0,btj=0;
        auto one=[&](T h,T dh,T wb,T db,T& bh,T& bth){LiteralKernel<T,D> k{x,r,h,args.kernel};
            if(product)k.backward(dt,dh,wb,db,bx,bh,bt,bth);else k.value_reverse(wb,bx,bh);};
        if(args.support==14||args.support==15){one(hj,tj,bw/T(2),bd/T(2),bhj,btj);one(hi,ti,bw/T(2),bd/T(2),bhi,bti);}
        else{T bh=0,bth=0;one(args.support==11?hi:args.support==12?hj:args.support==13?(hi+hj)/T(2):maximum_support(hi,hj),
                args.support==11?ti:args.support==12?tj:args.support==13?(ti+tj)/T(2):hi>=hj?ti:tj,bw,bd,bh,bth);
            if(args.support==11){bhi=bh;bti=bth;}else if(args.support==12){bhj=bh;btj=bth;}
            else if(args.support==13){bhi=bhj=bh/T(2);bti=btj=bth/T(2);}else{if(maximum_support_query(hi,hj))bhi=bh;else bhj=bh;if(hi>=hj)bti=bth;else btj=bth;}}
        for(int c=0;c<D;++c){bars.add(0,c,bx[c]);bars.add(1,c,-bx[c]);bars.add(10,c,bt[c]);bars.add(11,c,-bt[c]);}
        bars.add(2,0,bhi);bars.add(3,0,bhj);bars.add(12,0,bti);bars.add(13,0,btj);
    }
    SPH_HD void value_reverse(const T(&x)[D],T r,T hi,T hj,T bw,T(&bx)[D],T& bhi,T& bhj)const{
        const T dt[D]={};interpolate_jvp::Bars<T,D> bars{};kernel_reverse(x,r,hi,hj,dt,0,0,bw,0,false,bars);
        for(int c=0;c<D;++c)bx[c]+=bars.q[c];bhi+=bars.q[D];bhj+=bars.r[D];
    }
    template<bool Reverse,bool Product>SPH_HD Contribution<T,D> evaluate(int64_t i,int64_t j,int channel,T seed=T(0))const{
        Contribution<T,D> result{};const auto& a=args;
        if(!query_allowed(a.qkind[i],a.direction)||!reference_allowed(a.rkind[j],a.direction))return result;
        T x[D],dt[D],wind[D];const T r=QueryGeometry<T,D>(a.qx+i*D,a.lo,a.hi,a.periodic).distance(a.rx+j*D,x);
        for(int c=0;c<D;++c){wind[c]=a.periodic[c]?T(round((a.qx[i*D+c]-a.rx[j*D+c]-x[c])/(a.hi[c]-a.lo[c]))):T(0);
            dt[c]=tangent(0,i*D+c)-tangent(1,j*D+c);if(wind[c]!=T(0))dt[c]-=wind[c]*tangent(8,c);}
        const T hi=a.qh[i],hj=a.rh[j],ti=tangent(2,i),tj=tangent(3,j);
        T raw=unit?T(0):kernel_value_pair(x,r,hi,hj),draw=0;
        if constexpr(Product)if(!unit&&geometry()){
            draw=kernel_product(x,r,hi,hj,dt,ti,tj,raw);
            if((hi==T(0)||hj==T(0))&&r==T(0))raw=kernel_value_pair(x,r,hi,hj);
        }
        using S=pair_ad::Number<T,Reverse,512>;
        using Storage=std::conditional_t<Reverse,explicit_ad::Trace<T,512>,char>;
        Storage storage;explicit_ad::Trace<T,512>* trace=nullptr;
        if constexpr(Reverse)trace=new(&storage)explicit_ad::Trace<T,512>;
        auto leaf=[&](T value,int slot,int at=0,bool active=true){return S::leaf(value,0,trace,slot,at,Reverse&&active);};
        S xx[D],dd[D],xi[D],dxi[D];
        for(int c=0;c<D;++c){
            xx[c]=leaf(a.qx[i*D+c],0,c)-leaf(a.rx[j*D+c],1,c);
            if(wind[c]!=T(0))xx[c]-=leaf(a.hi[c]-a.lo[c],9,c)*S(wind[c]);xx[c].v=x[c];
            dd[c]=Product?leaf(tangent(0,i*D+c),10,c,tangents[0])-leaf(tangent(1,j*D+c),11,c,tangents[1]):S(0);
            if constexpr(Product)if(wind[c]!=T(0))dd[c]-=leaf(tangent(8,c),19,c,tangents[8])*S(wind[c]);
        }
        const S H=leaf(hi,2),DH=Product?leaf(ti,12,0,tangents[2]):S(0);
        for(int c=0;c<D;++c){xi[c]=explicit_ad::matrix_divide(-xx[c],H);
            if constexpr(Product)dxi[c]=explicit_ad::matrix_divide(-dd[c],H)+xx[c]*explicit_ad::divide(DH,H*H);}
        S w=unit?S(r<hi?T(1):T(0)):S(0),dw(0);
        if(!unit){
            S V(0),DV(0);
            if(volumes){V=leaf(volumes[j],6);if constexpr(Product)DV=leaf(tangent(6,j),16,0,tangents[6]);}
            else{const S M=leaf(a.mass[j],4),R=leaf(a.density[j],5);V=explicit_ad::divide(M,R);
                if constexpr(Product)DV=explicit_ad::divide(leaf(tangent(4,j),14,0,tangents[4]),R)
                    -explicit_ad::divide(M*leaf(tangent(5,j),15,0,tangents[5]),R*R);}
            const S W=leaf(raw,40)*S(scale),DW=Product&&geometry()?leaf(draw,41)*S(scale):S(0);
            w=V*W;if constexpr(Product)dw=DV*W+V*DW;
        }
        if(!allowed(i,j,x)){w=S(0);dw=S(0);}
        result.weight=w.v;
        const int aa=rhs?channel/components:entries[channel*2],bb=rhs?-1:entries[channel*2+1];
        const auto pa=monomial(exponents+aa*D,xi,Product?dxi:nullptr,D);
        S output;
        if(rhs){result.component=channel%components;
            const S field=leaf(values[j*components+result.component],7)
                -(subtract?leaf(values[(i+rowStart)*components+result.component],8):S(0));
            if constexpr(Product){const S df=leaf(tangent(7,j*components+result.component),17,0,tangents[7])
                    -(subtract?leaf(tangent(7,(i+rowStart)*components+result.component),18,0,tangents[7]):S(0));
                output=(dw*pa.first+w*pa.second)*field+w*pa.first*df;
            }else output=w*pa.first*field;
        }else{const auto pb=monomial(exponents+bb*D,xi,Product?dxi:nullptr,D);
            if constexpr(Product)output=dw*pa.first*pb.first+w*(pa.second*pb.first+pa.first*pb.second);
            else output=w*pa.first*pb.first;}
        result.value=output.v;
        if constexpr(Reverse){output.seed(seed);trace->reverse();T bw=0,bdw=0;
            for(int k=0;k<trace->inputs;++k){const auto l=trace->leaves[k];const T b=trace->nodes[l.node].bar;
                if(l.slot==40)bw+=b;else if(l.slot==41)bdw+=b;else result.bar[l.slot][l.at]+=b;}
            if(!unit){
                if constexpr(Product){if(geometry()){
                    interpolate_jvp::Bars<T,D> bars{};kernel_reverse(x,r,hi,hj,dt,ti,tj,bw,bdw,true,bars);
                    for(int c=0;c<D;++c){result.bar[0][c]+=bars.q[c];result.bar[1][c]+=bars.r[c];
                        result.bar[10][c]+=bars.q[interpolate_jvp::Layout<D>::Q+c];result.bar[11][c]+=bars.r[interpolate_jvp::Layout<D>::R+c];
                        if(wind[c]!=T(0)){result.bar[9][c]-=wind[c]*bars.q[c];result.bar[19][c]-=wind[c]*bars.q[interpolate_jvp::Layout<D>::Q+c];}}
                    result.bar[2][0]+=bars.q[D];result.bar[3][0]+=bars.r[D];
                    result.bar[12][0]+=bars.q[interpolate_jvp::Layout<D>::Q+D];result.bar[13][0]+=bars.r[interpolate_jvp::Layout<D>::R+D];
                }else{T bx[D]={},bhi=0,bhj=0;value_reverse(x,r,hi,hj,bw,bx,bhi,bhj);
                    for(int c=0;c<D;++c){result.bar[0][c]+=bx[c];result.bar[1][c]-=bx[c];if(wind[c]!=T(0))result.bar[9][c]-=wind[c]*bx[c];}
                    result.bar[2][0]+=bhi;result.bar[3][0]+=bhj;}
                }else{T bx[D]={},bhi=0,bhj=0;value_reverse(x,r,hi,hj,bw,bx,bhi,bhj);
                    for(int c=0;c<D;++c){result.bar[0][c]+=bx[c];result.bar[1][c]-=bx[c];if(wind[c]!=T(0))result.bar[9][c]-=wind[c]*bx[c];}
                    result.bar[2][0]+=bhi;result.bar[3][0]+=bhj;}
            }
        }
        return result;
    }
};
template<class T,int D>struct Rows{
    Pair<T,D> pair;T* output;int32_t* counts;const T* cotangent;
    T* bars[18]={};IndexView transposeOffsets,transposeQueries;const int32_t* transposeCounts;int role;
    template<bool Product>SPH_HD void reduce(int64_t owner)const{
        const bool query=role==2||role==4;const auto& a=pair.args;const int width=pair.width();
        const int64_t count=query?a.counts[owner]:transposeCounts[owner];
        for(int64_t local=0;local<count;++local){const int64_t i=query?owner:transposeQueries[transposeOffsets[owner]+local];
            const int64_t j=query?a.neighbors[a.offsets[i]+local]:owner;
            for(int channel=0;channel<width;++channel){const auto result=pair.template evaluate<true,Product>(i,j,channel,cotangent[i*width+channel]);
                for(int half=0;half<(Product?2:1);++half){const int b=half*9,s=half*10;
                    if(query){for(int c=0;c<D;++c){bars[b][i*D+c]+=result.bar[s][c];bars[b+8][i*D+c]+=result.bar[s+9][c];}
                        bars[b+2][i]+=result.bar[s+2][0];
                        if(pair.rhs&&pair.subtract)bars[b+7][(i+pair.rowStart)*pair.components+result.component]+=result.bar[s+8][0];
                    }else{for(int c=0;c<D;++c)bars[b+1][j*D+c]+=result.bar[s+1][c];
                        bars[b+3][j]+=result.bar[s+3][0];bars[b+4][j]+=result.bar[s+4][0];bars[b+5][j]+=result.bar[s+5][0];
                        if(bars[b+6])bars[b+6][j]+=result.bar[s+6][0];
                        if(pair.rhs)bars[b+7][j*pair.components+result.component]+=result.bar[s+7][0];}
                }
            }
        }
    }
    SPH_HD void operator()(int64_t thread)const{
        const int width=pair.width();
        if(role<=1){const int64_t i=thread/width;const int channel=int(thread%width);T sum=0;int32_t count=0;
            for(int32_t k=0;k<pair.args.counts[i];++k){const auto j=pair.args.neighbors[pair.args.offsets[i]+k];
                const auto value=role==0?pair.template evaluate<false,false>(i,j,channel):pair.template evaluate<false,true>(i,j,channel);
                sum+=value.value;if(role==0&&!pair.rhs&&channel==0&&value.weight!=T(0))++count;}
            output[thread]=sum;if(role==0&&!pair.rhs&&channel==0)counts[i]=count;
        }else if(role<4)reduce<false>(thread);else reduce<true>(thread);
    }
};
template<class Launch>struct Dispatch{
    const Buffers& buffers;int role;
    template<class T,int D>void operator()(int64_t n,RowArgs<T,D>a,const Launch& launch)const{
        Rows<T,D> rows{};
        rows.pair.args=a;rows.pair.exponents=buffers.exponents.const_data_ptr<int32_t>();rows.pair.entries=buffers.entries.const_data_ptr<int32_t>();
        rows.pair.volumes=ptr<T>(buffers.volumes);rows.pair.values=ptr<T>(buffers.values);rows.pair.basis=buffers.basis;rows.pair.components=buffers.components;
        rows.pair.sector=buffers.sector;rows.pair.sectors=buffers.sectors;rows.pair.rhs=buffers.rhs;rows.pair.subtract=buffers.subtract;
        rows.pair.unit=buffers.unit;rows.pair.rowStart=buffers.rowStart;rows.pair.scale=T(buffers.scale);
        for(int k=0;k<9;++k)if(!buffers.tangents.empty())rows.pair.tangents[k]=ptr<T>(buffers.tangents[k]);
        rows.output=ptr<T>(buffers.output)?buffers.output.mutable_data_ptr<T>():nullptr;rows.counts=buffers.counts.defined()?buffers.counts.mutable_data_ptr<int32_t>():nullptr;
        rows.cotangent=ptr<T>(buffers.cotangent);rows.role=role;rows.transposeOffsets=buffers.transposeOffsets;rows.transposeQueries=buffers.transposeQueries;rows.transposeCounts=buffers.transposeCounts;
        for(size_t k=0;k<buffers.bars.size();++k)if(buffers.bars[k].numel())rows.bars[k]=buffers.bars[k].mutable_data_ptr<T>();
        launch(role<=1?n*rows.pair.width():(role==2||role==4)?n:a.references,rows);
    }
};
void run_gpu(const Tensors&,const Buffers&,int,int,int,int);
} // namespace sph_native::polyfit
