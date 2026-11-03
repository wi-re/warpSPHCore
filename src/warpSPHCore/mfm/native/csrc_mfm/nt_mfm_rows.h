#pragma once
#include "nt_mfm_pair.h"
#include "execution/rowNeighborhood.h"
#include "execution/dispatch.h"
namespace sph_native::mfm {
enum Field { Rho,Velocity,Pressure,GradientRho,GradientVelocity,GradientPressure,AlphaRho,AlphaVelocity,AlphaPressure,
             Volume,Inverse,Potential,Bad,Beta,Kappa,Vector,Cell,FieldCount };
enum Pass { Moments,Gradients,Limiter,Residual,Weights,Matvec,Coarse,Flux,Timestep };
struct Options {
    double gamma=1.4,dt=0,psi1=.5,psi2=.25,power=1,massLimit=.1,closureCap=0,cfl=.2;
    int mode=0,order=2,timeCentred=1,centred=1,areaCap=0,retry=1,faceGuard=0;
    int cells[3]={1,1,1},periodic[3]={};
    int64_t rowStart=0;
};
struct Buffers {Tensors values,output;Options options;};
template<class T,int D,int Task> struct Row {
    RowArgs<T,D> args;
    const T* fields[FieldCount]={};
    const int32_t *bad=nullptr,*cells=nullptr;
    T* output[3]={};
    int32_t* counts=nullptr;
    Options options;
    using Vec=SmallVector<T,D>;using Mat=SmallMatrix<T,D>;
    SPH_HD T scalar(int field,int64_t i) const {return fields[field][i];}
    SPH_HD Vec vector(int field,int64_t i) const {return load_vector<T,D>(fields[field],i);}
    SPH_HD Mat matrix(int field,int64_t i) const {return load_matrix<T,D>(fields[field],i);}
    SPH_HD Vec displacement(int64_t i,int64_t j) const {
        Vec d{};QueryGeometry<T,D>(args.rx+i*D,args.lo,args.hi,args.periodic).distance(args.rx+j*D,d.values);return -d;
    }
    SPH_HD bool admitted(int64_t i,int64_t j,const Vec& d) const {return i!=j && length(d)<=maximum(args.rh[i],args.rh[j]);}
    SPH_HD Vec face(int64_t i,int64_t j,const Vec& d) const {
        return guarded_face(d,scalar(Volume,i),scalar(Volume,j),matrix(Inverse,i),matrix(Inverse,j),
            args.rh[i],args.rh[j],bad[i]!=0,bad[j]!=0,args.kernel,options.centred,options.areaCap);
    }
    template<class Visit> SPH_HD void each(int64_t row,Visit visit) const {
        const auto begin=args.offsets[row];
        for(int32_t k=0;k<args.counts[row];++k)visit(neighborhood::Neighbor{args.neighbors[begin+k],begin+k});
    }
    SPH_HD FaceFlux<T,D> riemann(int64_t a,int64_t b,const Vec& normal,const Vec& frame,Primitive<T,D> left,Primitive<T,D> right) const {
        left.density=maximum(left.density,T(1e-30));right.density=maximum(right.density,T(1e-30));
        left.pressure=maximum(left.pressure,T(1e-30));right.pressure=maximum(right.pressure,T(1e-30));
        const T uL=dot(left.velocity,normal),uR=dot(right.velocity,normal);
        return face_flux(left.density,uL,left.velocity-uL*normal,left.pressure,right.density,uR,right.velocity-uR*normal,right.pressure,
            T(options.gamma),options.mode,normal,frame);
    }
    SPH_HD FaceFlux<T,D> flux(int64_t a,int64_t b,Vec d,Vec face,T area) const {
        const Vec normal=face/area,va=vector(Velocity,a),vb=vector(Velocity,b);
        const T fraction=args.rh[a]/(args.rh[a]+args.rh[b]);
        Vec halfA=va,halfB=vb;
        if(options.order>=2 && options.dt>0 && options.timeCentred) {
            const auto da=time_derivative(scalar(Rho,a),va,scalar(Pressure,a),scalar(AlphaRho,a)*vector(GradientRho,a),
                limited_gradient(matrix(GradientVelocity,a),vector(AlphaVelocity,a)),scalar(AlphaPressure,a)*vector(GradientPressure,a),T(options.gamma));
            const auto db=time_derivative(scalar(Rho,b),vb,scalar(Pressure,b),scalar(AlphaRho,b)*vector(GradientRho,b),
                limited_gradient(matrix(GradientVelocity,b),vector(AlphaVelocity,b)),scalar(AlphaPressure,b)*vector(GradientPressure,b),T(options.gamma));
            halfA=va+T(.5)*T(options.dt)*da.velocity;halfB=vb+T(.5)*T(options.dt)*db.velocity;
        }
        const Vec frame=halfA+fraction*(halfB-halfA);
        auto left=reconstruct(scalar(Rho,a),va,scalar(Pressure,a),scalar(Rho,b),vb,scalar(Pressure,b),vector(GradientRho,a),matrix(GradientVelocity,a),vector(GradientPressure,a),
            scalar(AlphaRho,a),vector(AlphaVelocity,a),scalar(AlphaPressure,a),fraction*d,fraction,frame,T(options.dt),T(options.gamma),options.order,T(options.psi1),T(options.psi2));
        auto right=reconstruct(scalar(Rho,b),vb,scalar(Pressure,b),scalar(Rho,a),va,scalar(Pressure,a),vector(GradientRho,b),matrix(GradientVelocity,b),vector(GradientPressure,b),
            scalar(AlphaRho,b),vector(AlphaVelocity,b),scalar(AlphaPressure,b),-(T(1)-fraction)*d,T(1)-fraction,frame,T(options.dt),T(options.gamma),options.order,T(options.psi1),T(options.psi2));
        auto result=riemann(a,b,normal,frame,left,right);
        if(options.retry) {
            const T r=length(d),s1=maximum(T(0),-dot(vb-va,d)/maximum(r,T(1e-30))),s2=maximum(T(0),dot(va-vb,normal));
            const T speed=maximum(s1,s2),v2=speed*speed;
            T limit=T(1.1)*maximum(scalar(Pressure,a)+scalar(Rho,a)*v2,scalar(Pressure,b)+scalar(Rho,b)*v2);
            if(options.mode==1)limit*=T(2);
            const bool valid=result.pressure>T(0) && finite(result.pressure) && finite(result.contact) && result.pressure<=T(1.4)*limit && finite(result.mass) && finite(result.energy);
            if(!valid) {
                left={maximum(scalar(Rho,a),T(1e-30)),va-frame,maximum(scalar(Pressure,a),T(1e-30))};
                right={maximum(scalar(Rho,b),T(1e-30)),vb-frame,maximum(scalar(Pressure,b),T(1e-30))};
                result=riemann(a,b,normal,frame,left,right);
                if(!(result.pressure>T(0) && finite(result.pressure) && finite(result.contact) && finite(result.mass) && finite(result.energy))) {
                    left.velocity={};right.velocity={};result=riemann(a,b,normal,frame,left,right);
                }
            }
        }
        if(options.mode==1 && options.massLimit>0 && options.dt>0) {
            const T mass=area*result.mass*T(options.dt);
            const auto donor=mass<T(0) ? b:a;
            const T cap=T(options.massLimit)*scalar(Rho,donor)*scalar(Volume,donor);
            result.mass=maximum(minimum(mass,cap),-cap)/(area*T(options.dt));
        }
        return result;
    }
    SPH_HD void operator()(int64_t row) const {
        const int64_t i=row+options.rowStart;
        if constexpr(Task==Moments) {
            Vec zero{};T omega=weight(zero,args.rh[i],args.kernel);Mat moment{};int32_t count=1;
            each(row,[&](auto edge){const auto j=edge.reference;const auto d=displacement(i,j);if(!admitted(i,j,d))return;
                const T w=weight(d,args.rh[i],args.kernel);omega+=w;moment+=outer(d,d)*w;if(w>T(0))++count;});
            output[0][row]=omega;store_matrix(output[1],row,moment);counts[row]=count;
        } else if constexpr(Task==Gradients) {
            Vec rho{},pressure{};Mat velocity{};
            if(!bad[i]) each(row,[&](auto edge){const auto j=edge.reference;const auto d=displacement(i,j);if(!admitted(i,j,d))return;
                const T w=weight(d,args.rh[i],args.kernel);if(w<=T(0))return;
                rho+=w*(scalar(Rho,j)-scalar(Rho,i))*d;pressure+=w*(scalar(Pressure,j)-scalar(Pressure,i))*d;
                velocity+=outer(vector(Velocity,j)-vector(Velocity,i),d)*w;});
            const auto inverse=bad[i] ? Mat{}:matrix(Inverse,i);
            store_vector(output[0],row,inverse*rho);store_matrix(output[1],row,velocity*inverse);store_vector(output[2],row,inverse*pressure);
        } else if constexpr(Task==Limiter) {
            const T density=scalar(Rho,i),pressure=scalar(Pressure,i);const auto velocity=vector(Velocity,i);
            T highR=density,lowR=density,midHighR=density,midLowR=density,highP=pressure,lowP=pressure,midHighP=pressure,midLowP=pressure;
            Vec highV=velocity,lowV=velocity,midHighV=velocity,midLowV=velocity;
            each(row,[&](auto edge){const auto j=edge.reference;const auto d=displacement(i,j);if(!admitted(i,j,d))return;
                const auto disp=(args.rh[i]/(args.rh[i]+args.rh[j]))*d;
                highR=maximum(highR,scalar(Rho,j));lowR=minimum(lowR,scalar(Rho,j));
                highP=maximum(highP,scalar(Pressure,j));lowP=minimum(lowP,scalar(Pressure,j));
                const T midR=density+dot(vector(GradientRho,i),disp),midP=pressure+dot(vector(GradientPressure,i),disp);
                midHighR=maximum(midHighR,midR);midLowR=minimum(midLowR,midR);midHighP=maximum(midHighP,midP);midLowP=minimum(midLowP,midP);
                const auto neighbor=vector(Velocity,j),mid=velocity+matrix(GradientVelocity,i)*disp;
                for(int c=0;c<D;++c){highV[c]=maximum(highV[c],neighbor[c]);lowV[c]=minimum(lowV[c],neighbor[c]);midHighV[c]=maximum(midHighV[c],mid[c]);midLowV[c]=minimum(midLowV[c],mid[c]);}});
            const T beta=scalar(Beta,i);Vec limited{};
            for(int c=0;c<D;++c)limited[c]=alpha(velocity[c],highV[c],lowV[c],midHighV[c],midLowV[c],beta);
            output[0][row]=alpha(density,highR,lowR,midHighR,midLowR,beta);store_vector(output[1],row,limited);output[2][row]=alpha(pressure,highP,lowP,midHighP,midLowP,beta);
        } else if constexpr(Task==Residual || Task==Weights) {
            Vec sum{};T norm=0;
            each(row,[&](auto edge){const auto j=edge.reference;const auto d=displacement(i,j);T value=0;
                if(admitted(i,j,d)) {
                    const auto area=face(i,j,d);
                    if constexpr(Task==Weights)value=scalar_power(length(area),T(options.power));
                    else {sum+=area;norm+=dot(area,area);}
                }
                if constexpr(Task==Weights)output[0][edge.edge]=value;});
            if constexpr(Task==Residual){store_vector(output[0],row,sum);output[1][row]=norm;}
        } else if constexpr(Task==Matvec) {
            Vec sum{};T diagonal=0;const auto value=vector(Vector,i);
            each(row,[&](auto edge){const T k=scalar(Kappa,edge.edge);if(k>T(0)){sum+=k*(value-vector(Vector,edge.reference));diagonal+=k;}});
            store_vector(output[0],row,sum);output[1][row]=diagonal;
        } else if constexpr(Task==Coarse) {
            constexpr int width=D==1 ? 3:D==2 ? 9:27;
            T stencil[width]={};const int n0=options.cells[0],n1=options.cells[1],n2=options.cells[2],ci=cells[i];
            int origin[3]={ci/(n2*n1),(ci/n2)%n1,ci%n2};
            each(row,[&](auto edge){const auto j=edge.reference;const auto d=displacement(i,j);if(!admitted(i,j,d))return;
                const int cj=cells[j];int target[3]={cj/(n2*n1),(cj/n2)%n1,cj%n2},delta[3];
                for(int c=0;c<3;++c){delta[c]=target[c]-origin[c];if(options.periodic[c]){const int n=options.cells[c];if(delta[c]>1)delta[c]-=n;if(delta[c]<-1)delta[c]+=n;}if(delta[c]<-1 || delta[c]>1)return;}
                if(delta[0]==0 && delta[1]==0 && delta[2]==0)return;
                const int slot=D==1 ? delta[0]+1:D==2 ? (delta[0]+1)*3+delta[1]+1:(delta[0]+1)*9+(delta[1]+1)*3+delta[2]+1;
                stencil[slot]+=scalar(Kappa,edge.edge);});
            for(int c=0;c<width;++c)output[0][row*width+c]=stencil[c];
        } else if constexpr(Task==Flux) {
            T mass=0,energy=0;Vec momentum{};
            each(row,[&](auto edge){const auto j=edge.reference;if(i==j)return;
                const auto a=i<j ? i:j,b=i<j ? j:i;const auto d=displacement(a,b);if(!admitted(a,b,d))return;
                const auto original=face(a,b,d);const T k=scalar_power(length(original),T(options.power));
                auto difference=vector(Potential,b)-vector(Potential,a);const T magnitude=length(difference);
                if(options.closureCap>0 && magnitude>T(options.closureCap))difference=difference*(T(options.closureCap)/magnitude);
                auto corrected=original+k*difference;
                if(options.faceGuard) {
                    const T before=dot(original,d),after=dot(corrected,d);
                    if(options.faceGuard==1) {if(after<=T(0))corrected=original;}
                    else if(options.faceGuard>=3) {const auto change=corrected-original;const T norm=length(change),cap=(options.faceGuard==4 ? T(.25):T(.5))*length(original);if(norm>cap)corrected=original+change*(cap/norm);}
                    else if(after<T(.25)*before){const T change=after-before,scale=change<T(0) ? maximum(T(0),(T(.25)*before-before)/change):T(0);corrected=original+scale*(corrected-original);}
                }
                const T area=length(corrected);if(area<=T(0))return;
                const auto value=flux(a,b,d,corrected,area);const T sign=i==a ? T(-1):T(1);
                mass+=sign*area*value.mass;momentum+=sign*area*value.momentum;energy+=sign*area*value.energy;});
            output[0][row]=mass;store_vector(output[1],row,momentum);output[2][row]=energy;
        } else if constexpr(Task==Timestep) {
            const T sound=scalar_sqrt(T(options.gamma)*scalar(Pressure,i)/scalar(Rho,i));T signal=T(2)*sound;
            each(row,[&](auto edge){const auto j=edge.reference;const auto d=displacement(i,j);if(!admitted(i,j,d))return;
                const T neighbor=scalar_sqrt(T(options.gamma)*scalar(Pressure,j)/scalar(Rho,j));
                const T approach=minimum(T(0),dot(vector(Velocity,j)-vector(Velocity,i),d)/maximum(length(d),T(1e-30)));
                signal=maximum(signal,sound+neighbor-approach);});
            output[0][row]=T(2)*T(options.cfl)*args.rh[i]/signal;
        }
    }
};
template<int Task> struct Dispatch {
    const Buffers& buffers;
    template<class T,int D,class Launch> void operator()(int64_t n,RowArgs<T,D> args,const Launch& launch) const {
        Row<T,D,Task> row{};row.args=args;row.options=buffers.options;
        for(int field=0;field<FieldCount;++field)if(buffers.values[field].defined()) {
            if(field==Bad)row.bad=buffers.values[field].const_data_ptr<int32_t>();
            else if(field==Cell)row.cells=buffers.values[field].const_data_ptr<int32_t>();
            else row.fields[field]=buffers.values[field].const_data_ptr<T>();
        }
        for(size_t slot=0;slot<buffers.output.size();++slot) {
            if constexpr(Task==Moments){if(slot==2){row.counts=buffers.output[slot].mutable_data_ptr<int32_t>();continue;}}
            row.output[slot]=buffers.output[slot].mutable_data_ptr<T>();
        }
        launch(n,row);
    }
};
template<class Launch> void run(const Tensors& t,const Buffers& buffers,int task,int kernel,const Launch& launch) {
#define MFM_PASS(Name) case Name:execute(Dispatch<Name>{buffers},t,11,0,1,launch,kernel);break
    switch(task){MFM_PASS(Moments);MFM_PASS(Gradients);MFM_PASS(Limiter);MFM_PASS(Residual);MFM_PASS(Weights);MFM_PASS(Matvec);MFM_PASS(Coarse);MFM_PASS(Flux);MFM_PASS(Timestep);}
#undef MFM_PASS
}
void run_gpu(const Tensors&,const Buffers&,int,int);
} // namespace sph_native::mfm
