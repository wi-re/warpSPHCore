#pragma once
#include "math/symmetric.h"
#include "execution/numeric.h"
namespace sph_native::dense {
template<class T> SPH_HD T absolute(T value) { return value<T(0) ? -value:value; }
template<class T,int D> struct InverseRow {
    const T *matrix,*values;
    const int32_t* counts;
    T *output,*eigenvalues;
    int mode,rows;
    T rtol;
    SPH_HD void operator()(int64_t particle) const {
        const T* input=matrix+particle*D*D;
        T inverse[D*D]={};
        if (mode==1) {
            if constexpr(D==1) {
                inverse[0]=input[0]>T(1e-10) ? T(1)/input[0]:T(0);
                eigenvalues[particle]=input[0];
            } else if constexpr(D==2) {
                T* spectrum=eigenvalues+particle*2;
                if (counts[particle]<4) {
                    inverse[0]=inverse[3]=T(1); spectrum[0]=spectrum[1]=T(1);
                } else {
                    const T a=input[0], b=(input[1]+input[2])/T(2), d=input[3];
                    const T discriminant=scalar_sqrt((a-d)*(a-d)+T(4)*b*b);
                    T big=(a+d+discriminant)/T(2),small=(a+d-discriminant)/T(2);
                    if (absolute(small)>absolute(big)) {const T swap=big;big=small;small=swap;}
                    spectrum[0]=big;spectrum[1]=small;
                    const T threshold=T(1e-6)*absolute(big);
                    if (absolute(big)>T(1e-12) && absolute(small)>threshold) {
                        const T determinant=a*d-b*b;
                        inverse[0]=d/determinant;inverse[1]=inverse[2]=-b/determinant;inverse[3]=a/determinant;
                    } else {
                        // Preserve the reference's separate dominant absolute
                        // cutoff and relative cutoff for the smaller direction.
                        const T angle=T(.5)*::atan2(T(2)*b,a-d),c=::cos(angle),s=::sin(angle);
                        T l1=a*c*c+T(2)*b*c*s+d*s*s,l2=a+d-l1;
                        T vx=c,vy=s,wx=-s,wy=c;
                        if (absolute(l2)>absolute(l1)) {
                            T swap=l1;l1=l2;l2=swap;swap=vx;vx=wx;wx=swap;swap=vy;vy=wy;wy=swap;
                        }
                        const T f=absolute(l1)<=T(1e-12) ? T(0):T(1)/l1;
                        const T g=absolute(l2)<=threshold ? T(0):T(1)/l2;
                        inverse[0]=f*vx*vx+g*wx*wx;inverse[1]=inverse[2]=f*vx*vy+g*wx*wy;inverse[3]=f*vy*vy+g*wy*wy;
                    }
                }
            }
        } else symmetric_inverse<T,D>(input,rtol,inverse);
        if (mode==2) {
            for (int row=0;row<rows;++row) for (int col=0;col<D;++col) {
                T sum=0;
                for (int k=0;k<D;++k) sum+=values[(particle*rows+row)*D+k]*inverse[k*D+col];
                output[(particle*rows+row)*D+col]=sum;
            }
        } else for (int i=0;i<D*D;++i) output[particle*D*D+i]=inverse[i];
    }
};
struct Buffers { Tensor matrix,values,counts,output,eigenvalues; };
template<class Launch> void execute(const Buffers& buffers,int mode,double rtol,const Launch& launch) {
    dispatch_numeric(buffers.matrix.scalar_type(),int(buffers.matrix.size(1)),[&](auto tag) {
        using T=typename decltype(tag)::scalar;
        constexpr int D=decltype(tag)::dim;
        launch(buffers.matrix.size(0),InverseRow<T,D>{buffers.matrix.const_data_ptr<T>(),
            buffers.values.defined() ? buffers.values.const_data_ptr<T>():nullptr,
            buffers.counts.defined() ? buffers.counts.const_data_ptr<int32_t>():nullptr,
            buffers.output.mutable_data_ptr<T>(),buffers.eigenvalues.defined() ? buffers.eigenvalues.mutable_data_ptr<T>():nullptr,mode,
            buffers.values.defined() ? int(buffers.values.size(1)):0,T(rtol)});
    });
}
void inverse_gpu(const Buffers&,int,double);
} // namespace sph_native::dense
