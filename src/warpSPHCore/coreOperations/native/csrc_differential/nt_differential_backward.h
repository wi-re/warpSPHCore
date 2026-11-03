#pragma once
#include "nt_differential_pair.h"
namespace sph_native::differential {
// Compact per-pair bars. Shared scalars belong to their physical population;
// domain bounds are query-owned partial sums, reduced after the native gather.
template <int D> struct Layout {
    static constexpr int F=D*D;
    static constexpr int qh=D, qd=qh+1, qf=qd+1, ca=qf+F, cb=ca+1,
        ga=cb+D, gb=ga+D, qo=gb+D*D, rm=qo+1, lo=rm+D*D, hi=lo+D, Q=hi+D;
    static constexpr int rh=D, mass=rh+1, rho=mass+1, rf=rho+1,
        volume=rf+F, ro=volume+1, Ref=ro+1;
    SPH_HD static int offset(int slot) {
        switch(slot) {
        case QX: case RX:return 0;
        case QH:return qh; case RH:return rh; case QDENSITY:return qd;
        case QVALUE:return qf; case RVALUE:return rf; case MASS:return mass; case DENSITY:return rho;
        case A:return ca; case B:return cb; case GA:return ga; case GB:return gb;
        case QO:return qo; case RO:return ro; case VOLUME:return volume; case R:return rm;
        case LO:return lo; case HI:return hi; default:return -1;
        }
    }
    SPH_HD static bool reference(int slot) {
        return slot==RX || slot==RH || slot==MASS || slot==DENSITY || slot==RVALUE || slot==VOLUME || slot==RO;
    }
    SPH_HD static int width(int slot,int fields) {
        switch(slot) {
        case QX:case RX:case LO:case HI:case B:case GA:return D;
        case GB:case R:return D*D;
        case QVALUE:case RVALUE:return fields;
        default:return 1;
        }
    }
};
template <class T,int D> struct PairBars {
    T query[Layout<D>::Q]={},reference[Layout<D>::Ref]={};
    SPH_HD void add(int slot,int component,T value) {
        const int at=Layout<D>::offset(slot)+component;
        if (Layout<D>::reference(slot)) reference[at]+=value;
        else query[at]+=value;
    }
};

template <class T,int D>
SPH_HD PairBars<T,D> backward(const RowArgs<T,D>& a,const Arrays<T>& arrays,
        int64_t i,int64_t j,const Pair<T,D>& p,const T* bar,bool consistent,bool geometry) {
    PairBars<T,D> result{};
    T clipped_bar[MaxOutputs]={};
    if (a.operation==5 && a.positive_divergence) {
        T gate=0;
        for (int d=0;d<D;++d) gate+=p.dx[d]*p.weighted[d];
        if (gate<0) return result;
        for (int d=0;d<D;++d) clipped_bar[d]=bar[d];
        bar=clipped_bar;
    }
    T bw=0,bg[D]={},bx[D]={},br=0,bh=0,blap=0,bv=0,bz[MaxFields]={};
    T raw_bw=0,raw_bg[D]={};
    if (a.operation==6) {
        raw_bw=bar[0]*p.mass;
        result.add(MASS,0,bar[0]*p.raw_w);
    } else if (a.operation==Volume) raw_bw=bar[0];
    else if (a.operation==Consistency) {
        bw=bar[0]*p.mass*p.volume+bar[1]*p.volume*p.volume;
        bv=bar[0]*p.mass*p.w+T(2)*bar[1]*p.volume*p.w;
        result.add(MASS,0,bar[0]*p.volume*p.w);
    } else if (a.operation==Moments) {
        constexpr int m1=1,m2=m1+D,dm0=m2+D*D,dm1=dm0+D,dm2=dm1+D*D;
        raw_bw+=bar[0]*p.volume; bv+=bar[0]*p.raw_w;
        for (int d=0; d<D; ++d) {
            T b=bar[m1+d];
            raw_bw+=b*p.volume*p.dx[d]; bx[d]+=b*p.volume*p.raw_w; bv+=b*p.raw_w*p.dx[d];
            b=bar[dm0+d]; raw_bg[d]+=b*p.volume; bv+=b*p.raw_g[d];
            for (int c=0; c<D; ++c) {
                b=bar[m2+d*D+c];
                raw_bw+=b*p.volume*p.dx[d]*p.dx[c]; bv+=b*p.raw_w*p.dx[d]*p.dx[c];
                bx[d]+=b*p.volume*p.raw_w*p.dx[c]; bx[c]+=b*p.volume*p.raw_w*p.dx[d];
                b=bar[dm1+d*D+c];
                bv+=b*(p.dx[d]*p.raw_g[c]+T(d==c)*p.raw_w);
                bx[d]+=b*p.volume*p.raw_g[c]; raw_bg[c]+=b*p.volume*p.dx[d]; raw_bw+=b*p.volume*T(d==c);
                for (int e=0; e<D; ++e) {
                    b=bar[dm2+e*D*D+d*D+c];
                    const T z=p.raw_g[e]*p.dx[d]*p.dx[c]+p.raw_w*(T(d==e)*p.dx[c]+T(c==e)*p.dx[d]);
                    bv+=b*z;
                    raw_bg[e]+=b*p.volume*p.dx[d]*p.dx[c];
                    raw_bw+=b*p.volume*(T(d==e)*p.dx[c]+T(c==e)*p.dx[d]);
                    bx[d]+=b*p.volume*(p.raw_g[e]*p.dx[c]+T(c==e)*p.raw_w);
                    bx[c]+=b*p.volume*(p.raw_g[e]*p.dx[d]+T(d==e)*p.raw_w);
                }
            }
        }
    } else if (a.operation==7) {
        for (int d=0; d<D; ++d) for (int c=0; c<D; ++c) {
            const T b=bar[d*D+c];
            bv-=b*p.dx[d]*p.gradient[c]; bx[d]-=b*p.volume*p.gradient[c]; bg[c]-=b*p.volume*p.dx[d];
        }
    } else if (a.operation==1) {
        for (int c=0; c<a.fields; ++c) { bz[c]=bar[c]*p.w; bw+=bar[c]*p.weighted[c]; }
    } else if (a.operation==2) {
        for (int c=0; c<a.fields; ++c) for (int d=0; d<D; ++d) {
            bz[c]+=bar[c*D+d]*p.gradient[d]; bg[d]+=bar[c*D+d]*p.weighted[c];
        }
    } else if (a.operation==3) {
        const int components=a.fields/D;
        for (int c=0; c<components; ++c) for (int d=0; d<D; ++d) {
            const int field=a.dot_mode ? d*components+c : c*D+d;
            bz[field]+=bar[c]*p.gradient[d]; bg[d]+=bar[c]*p.weighted[field];
        }
    } else if (a.operation==4) {
        if constexpr (D==2) {
            const int stride=a.fields/2;
            for (int c=0; c<stride; ++c) {
                bz[stride+c]+=bar[c]*p.gradient[0]; bz[c]-=bar[c]*p.gradient[1];
                bg[0]+=bar[c]*p.weighted[stride+c]; bg[1]-=bar[c]*p.weighted[c];
            }
        } else if constexpr (D==3) {
            const int stride=a.fields/3;
            for (int c=0; c<stride; ++c) for (int d=0; d<3; ++d) {
                const T b=bar[d*stride+c]; const int u=(d+1)%3,v=(d+2)%3;
                bz[v*stride+c]+=b*p.gradient[u]; bz[u*stride+c]-=b*p.gradient[v];
                bg[u]+=b*p.weighted[v*stride+c]; bg[v]-=b*p.weighted[u*stride+c];
            }
        }
    } else if (a.operation==5) {
        if (a.laplacian_mode==1) {
            for (int c=0; c<a.fields; ++c) { bz[c]=bar[c]*p.lap; blap+=bar[c]*p.weighted[c]; }
        } else if (p.distance>0) {
            const T denominator=p.distance+T(1e-8)*p.h;
            T normal[D],bn[D]={},projection=0,bden=0;
            for (int d=0; d<D; ++d) { normal[d]=p.dx[d]/denominator; projection+=p.gradient[d]*normal[d]; }
            T bprojection=0;
            if (a.laplacian_mode==3) {
                const T factor=projection/denominator;
                for (int base=0; base<a.fields; base+=D) {
                    T z=0,bzproj=0;
                    for (int d=0; d<D; ++d) { z+=p.weighted[base+d]*normal[d]; bzproj+=bar[base+d]*normal[d]; }
                    for (int d=0; d<D; ++d) {
                        bz[base+d]+=factor*(bar[base+d]-T(D+2)*bzproj*normal[d]);
                        bn[d]-=T(D+2)*factor*(z*bar[base+d]+bzproj*p.weighted[base+d]);
                        bprojection+=bar[base+d]*(p.weighted[base+d]-T(D+2)*z*normal[d])/denominator;
                    }
                }
                bden-=bprojection*projection/denominator;
            } else {
                const T eps=a.laplacian_mode==4 ? T(1e-12):T(1e-8);
                const T divisor=p.distance+eps*p.h,factor=-T(2)*projection/divisor;
                T bf=0;
                for (int c=0; c<a.fields; ++c) { bz[c]=bar[c]*factor; bf+=bar[c]*p.weighted[c]; }
                bprojection=-T(2)*bf/divisor;
                const T bdivisor=-bf*factor/divisor;
                br+=bdivisor; bh+=eps*bdivisor;
            }
            for (int d=0; d<D; ++d) {
                bg[d]+=bprojection*normal[d]; bn[d]+=bprojection*p.gradient[d];
                bx[d]+=bn[d]/denominator; bden-=bn[d]*p.dx[d]/(denominator*denominator);
            }
            br+=bden; bh+=T(1e-8)*bden;
        }
    }
    // Pull back the operator's density/volume weighting independently of the
    // kernel algebra. grad-h divisions belong to their original field roles.
    if (a.operation<=5) for (int c=0; c<a.fields; ++c) {
        T bfi=0,bfj=0,b=bz[c];
        if (consistent && a.operation==3 && a.scheme!=2) {
            result.add(QDENSITY,0,-b*p.weighted[c]/p.rhoi);
            result.add(DENSITY,0,b*p.weighted[c]/p.rhoj);
            b*=p.rhoj/p.rhoi;
        }
        if (a.operation==1) { bfj=b*p.volume; bv+=b*p.fj[c]; }
        else if (a.scheme==2 && a.operation!=5) {
            bfi=b*p.mass/p.rhoi; bfj=b*p.mass*p.rhoi/(p.rhoj*p.rhoj);
            result.add(MASS,0,b*(p.fi[c]/p.rhoi+p.rhoi*p.fj[c]/(p.rhoj*p.rhoj)));
            result.add(QDENSITY,0,b*p.mass*(-p.fi[c]/(p.rhoi*p.rhoi)+p.fj[c]/(p.rhoj*p.rhoj)));
            result.add(DENSITY,0,-T(2)*b*p.mass*p.rhoi*p.fj[c]/(p.rhoj*p.rhoj*p.rhoj));
        } else {
            const bool difference=a.operation==5 || a.scheme==3;
            const T combination=difference ? p.fj[c]-p.fi[c] : a.scheme==1 ? p.fj[c] : p.fj[c]+p.fi[c];
            T scale=p.volume;
            if (a.operation==5 && a.scheme==2) {
                scale=p.mass*p.rhoi/(p.rhoj*p.rhoj);
                result.add(MASS,0,b*combination*p.rhoi/(p.rhoj*p.rhoj));
                result.add(QDENSITY,0,b*combination*p.mass/(p.rhoj*p.rhoj));
                result.add(DENSITY,0,-T(2)*b*combination*scale/p.rhoj);
            } else bv+=b*combination;
            bfj=b*scale;
            if (difference || a.scheme!=1) bfi=difference ? -bfj:bfj;
        }
        const bool qo=a.operation>=3 && a.operation<=5;
        result.add(QVALUE,c,bfi/(qo ? p.query_omega:T(1)));
        result.add(RVALUE,c,bfj/(a.operation!=1 ? p.reference_omega:T(1)));
        if (qo) result.add(QO,0,-bfi*p.fi[c]/p.query_omega);
        if (a.operation!=1) result.add(RO,0,-bfj*p.fj[c]/p.reference_omega);
    }
    if (arrays.values[VOLUME]) result.add(VOLUME,0,bv);
    else if (a.operation!=6) {
        result.add(MASS,0,bv/p.rhoj);
        result.add(DENSITY,0,-bv*p.mass/(p.rhoj*p.rhoj));
    }
    // Reverse query renormalization and the CRK polynomial/gradient. gradB's
    // first axis is the derivative axis, not the component axis.
    T pre_bar[D]={};
    for (int d=0; d<D; ++d) for (int c=0; c<D; ++c) {
        pre_bar[c]+=p.matrix[d*D+c]*bg[d];
        result.add(R,d*D+c,bg[d]*p.before_R[c]);
    }
    T bp=0;
    if (p.crk) {
        result.add(A,0,bw*p.polynomial*p.raw_w);
        bp+=bw*p.coefficient_A*p.raw_w; raw_bw+=bw*p.coefficient_A*p.polynomial;
        for (int d=0; d<D; ++d) {
            T product=0; for (int c=0; c<D; ++c) product+=p.grad_B[d*D+c]*p.dx[c];
            const T b=pre_bar[d];
            result.add(A,0,b*(p.coefficient_B[d]*p.raw_w+p.polynomial*p.raw_g[d]+p.raw_w*product));
            result.add(B,d,b*p.coefficient_A*p.raw_w);
            raw_bw+=b*(p.coefficient_A*p.coefficient_B[d]+p.polynomial*p.grad_A[d]+p.coefficient_A*product);
            raw_bg[d]+=b*p.coefficient_A*p.polynomial;
            bp+=b*(p.coefficient_A*p.raw_g[d]+p.raw_w*p.grad_A[d]);
            result.add(GA,d,b*p.polynomial*p.raw_w);
            for (int c=0; c<D; ++c) {
                result.add(GB,d*D+c,b*p.coefficient_A*p.raw_w*p.dx[c]);
                bx[c]+=b*p.coefficient_A*p.raw_w*p.grad_B[d*D+c];
            }
        }
        for (int d=0; d<D; ++d) { result.add(B,d,bp*p.dx[d]); bx[d]+=bp*p.coefficient_B[d]; }
    } else { raw_bw+=bw; for (int d=0; d<D; ++d) raw_bg[d]+=pre_bar[d]; }
    if (geometry) {
        T bhi=0,bhj=0;
        if (blap!=0) {
            if (a.support==15) {
                laplacian_vjp<T,D>(p.distance,p.hi,a.kernel,blap/T(2),br,bhi);
                laplacian_vjp<T,D>(p.distance,p.hj,a.kernel,blap/T(2),br,bhj);
            } else laplacian_vjp<T,D>(p.distance,p.h,a.kernel,blap,br,bh);
        }
        if (a.support==14 || a.support==15) {
            T half[D]; for (int d=0; d<D; ++d) half[d]=raw_bg[d]/T(2);
            kernel_vjp<T,D>(p.dx,p.hi,a.kernel,raw_bw/T(2),half,bx,bhi);
            kernel_vjp<T,D>(p.dx,p.hj,a.kernel,raw_bw/T(2),half,bx,bhj);
        } else kernel_vjp<T,D>(p.dx,p.h,a.kernel,raw_bw,raw_bg,bx,bh);
        if (a.support==11) bhi+=bh;
        else if (a.support==12) bhj+=bh;
        else if (a.support==13) { bhi+=bh/T(2); bhj+=bh/T(2); }
        else if (p.hi>=p.hj) bhi+=bh; else bhj+=bh;
        result.add(QH,0,bhi); result.add(RH,0,bhj);
        for (int d=0; d<D; ++d) {
            const T b=bx[d]+(p.distance>0 ? br*p.dx[d]/p.distance:T(0));
            result.add(QX,d,b); result.add(RX,d,-b);
            if (a.periodic[d]) {
                const T winding=T(round((a.qx[i*D+d]-a.rx[j*D+d]-p.dx[d])/(a.hi[d]-a.lo[d])));
                result.add(LO,d,b*winding); result.add(HI,d,-b*winding);
            }
        }
    }
    return result;
}
} // namespace sph_native::differential
