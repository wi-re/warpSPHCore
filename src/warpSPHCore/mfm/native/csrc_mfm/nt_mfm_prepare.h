#pragma once
#include "nt_mfm_rows.h"
#include "torch/validation.h"
#include <limits>
namespace sph_native::mfm {
inline Tensors prepare_mfm(const OptionalTensors& inputs,const OptionalTensors& values,const std::vector<double>& scalars,
                          const std::vector<int64_t>& integers,int64_t task,Buffers& b){
    STD_TORCH_CHECK(inputs.size()==OUTPUT && (values.size()==FieldCount || values.size()==17) && scalars.size()==8 && integers.size()==14,"Invalid meshless argument layout");
    STD_TORCH_CHECK(task>=Moments && task<=Timestep,"Unknown meshless pass");
    Tensors flat;for(const auto& value:inputs)flat.push_back(value ? *value:Tensor{});
    auto t=prepare_pair_geometry(flat,11,0);const auto& x=t[QX];
    const int64_t n=x.size(0),nr=t[RX].size(0),d=x.size(1);
    auto& o=b.options;
    o.gamma=scalars[0];o.dt=scalars[1];o.psi1=scalars[2];o.psi2=scalars[3];o.power=scalars[4];o.massLimit=scalars[5];o.closureCap=scalars[6];o.cfl=scalars[7];
    o.mode=int(integers[0]);o.order=int(integers[1]);o.timeCentred=int(integers[2]);o.centred=int(integers[3]);o.areaCap=int(integers[4]);o.retry=int(integers[5]);o.faceGuard=int(integers[6]);
    int64_t cellCount=1;
    for(int c=0;c<3;++c) {
        const int64_t cells=integers[7+c];
        STD_TORCH_CHECK(cells>=1 && cells<=std::numeric_limits<int32_t>::max(),"Invalid coarse grid extent");
        if(task==Coarse) {
            STD_TORCH_CHECK(cellCount<=std::numeric_limits<int32_t>::max()/cells,"Coarse cell IDs must fit int32");
            cellCount*=cells;
        }
        o.cells[c]=int(cells);o.periodic[c]=int(integers[10+c]);
    }
    o.rowStart=integers[13];
    STD_TORCH_CHECK(o.rowStart>=0 && o.rowStart<=nr && n<=nr-o.rowStart,"Meshless rows must belong to the source population");
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
        STD_TORCH_CHECK((field<int(values.size()) && values[field]) || !(required[task]&(1<<field)),"Missing meshless input field");
        const Tensor value=field<int(values.size()) && values[field] ? *values[field]:Tensor{};
        b.values.push_back(value);if(!value.defined())continue;
        tensor(value,x,field==Bad || field==Cell || field==Material ? ScalarType::Int:x.scalar_type(),"meshless field");
        std::vector<int64_t> shape={field==Kappa ? t[NEIGHBORS].numel():field==DeviceTimestep?1:nr};
        if(field==Velocity || field==GradientRho || field==GradientPressure || field==AlphaVelocity || field==Potential || field==Vector)shape.push_back(d);
        if(field==Inverse || field==GradientVelocity){shape.push_back(d);shape.push_back(d);}
        STD_TORCH_CHECK(value.sizes().vec()==shape,"Meshless field shape mismatch");
    }
    return t;
}
}
