#include "nt_mfm_rows.h"
#include "torch/validation.h"
#include "execution/cpu.h"
namespace sph_native::mfm {
Tensors evaluate(const OptionalTensors& inputs,const OptionalTensors& values,const std::vector<double>& scalars,
                 const std::vector<int64_t>& integers,int64_t task,int64_t kernel) {
    STD_TORCH_CHECK(inputs.size()==OUTPUT && values.size()==FieldCount && scalars.size()==8 && integers.size()==14,"Invalid meshless argument layout");
    STD_TORCH_CHECK(task>=Moments && task<=Timestep,"Unknown meshless pass");
    Tensors flat;for(const auto& value:inputs)flat.push_back(value ? *value:Tensor{});
    auto t=prepare_pair_geometry(flat,11,0);const auto& x=t[QX];
    const int64_t n=x.size(0),nr=t[RX].size(0),d=x.size(1);
    Buffers b{};auto& o=b.options;
    o.gamma=scalars[0];o.dt=scalars[1];o.psi1=scalars[2];o.psi2=scalars[3];o.power=scalars[4];o.massLimit=scalars[5];o.closureCap=scalars[6];o.cfl=scalars[7];
    o.mode=int(integers[0]);o.order=int(integers[1]);o.timeCentred=int(integers[2]);o.centred=int(integers[3]);o.areaCap=int(integers[4]);o.retry=int(integers[5]);o.faceGuard=int(integers[6]);
    for(int c=0;c<3;++c){o.cells[c]=int(integers[7+c]);o.periodic[c]=int(integers[10+c]);}
    o.rowStart=integers[13];
    STD_TORCH_CHECK(o.rowStart>=0 && o.rowStart+n<=nr,"Meshless rows must belong to the source population");
    constexpr uint32_t required[]={
        0,
        (1<<Rho)|(1<<Velocity)|(1<<Pressure)|(1<<Inverse)|(1<<Bad),
        (1<<Rho)|(1<<Velocity)|(1<<Pressure)|(1<<GradientRho)|(1<<GradientVelocity)|(1<<GradientPressure)|(1<<Beta),
        (1<<Volume)|(1<<Inverse)|(1<<Bad),
        (1<<Volume)|(1<<Inverse)|(1<<Bad),
        (1<<Kappa)|(1<<Vector),
        (1<<Kappa)|(1<<Cell),
        (1<<Rho)|(1<<Velocity)|(1<<Pressure)|(1<<GradientRho)|(1<<GradientVelocity)|(1<<GradientPressure)|
            (1<<AlphaRho)|(1<<AlphaVelocity)|(1<<AlphaPressure)|(1<<Volume)|(1<<Inverse)|(1<<Potential)|(1<<Bad),
        (1<<Rho)|(1<<Velocity)|(1<<Pressure)
    };
    for(int field=0;field<FieldCount;++field) {
        STD_TORCH_CHECK(values[field] || !(required[task]&(1<<field)),"Missing meshless input field");
        const Tensor value=values[field] ? *values[field]:Tensor{};
        b.values.push_back(value);if(!value.defined())continue;
        tensor(value,x,field==Bad || field==Cell ? ScalarType::Int:x.scalar_type(),"meshless field");
        std::vector<int64_t> shape={field==Kappa ? t[NEIGHBORS].numel():nr};
        if(field==Velocity || field==GradientRho || field==GradientPressure || field==AlphaVelocity || field==Potential || field==Vector)shape.push_back(d);
        if(field==Inverse || field==GradientVelocity){shape.push_back(d);shape.push_back(d);}
        STD_TORCH_CHECK(value.sizes().vec()==shape,"Meshless field shape mismatch");
    }
    auto scalar=[&](){return stable::new_empty(x,{n});};
    auto vector=[&](){return stable::new_empty(x,{n,d});};
    auto matrix=[&](){return stable::new_empty(x,{n,d,d});};
    switch(task) {
    case Moments:b.output={scalar(),matrix(),stable::new_empty(x,{n},ScalarType::Int)};break;
    case Gradients:b.output={vector(),matrix(),vector()};break;
    case Limiter:b.output={scalar(),vector(),scalar()};break;
    case Residual:case Matvec:b.output={vector(),scalar()};break;
    case Weights:b.output={stable::new_zeros(x,{t[NEIGHBORS].numel()})};break;
    case Coarse:b.output={stable::new_empty(x,{n,d==1 ? 3:d==2 ? 9:27})};break;
    case Flux:b.output={scalar(),vector(),scalar()};break;
    case Timestep:b.output={scalar()};break;
    }
    if(x.is_cuda()) {
#ifdef WITH_CUDA
        run_gpu(t,b,int(task),int(kernel));
#else
        STD_TORCH_CHECK(false,"Build native GPU support to use CUDA/HIP");
#endif
    } else run(t,b,int(task),int(kernel),CpuLauncher{});
    return b.output;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m) {
    m.def("meshless(Tensor?[] inputs, Tensor?[] values, float[] scalars, int[] integers, int task, int kernel) -> Tensor[]");
    m.impl("meshless",TORCH_BOX(&evaluate));
}
} // namespace sph_native::mfm
