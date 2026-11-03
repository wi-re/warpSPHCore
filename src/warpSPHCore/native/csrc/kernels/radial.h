#pragma once
#include "kernels/kernelFunctions/wendland2.h"
#include "kernels/kernelFunctions/wendland4.h"
#include "kernels/kernelFunctions/wendland6.h"
#include "kernels/kernelFunctions/cubicSpline.h"
#include "kernels/kernelFunctions/quarticSpline.h"
#include "kernels/kernelFunctions/quinticSpline.h"
#include "kernels/kernelFunctions/B7.h"
#include "kernels/kernelFunctions/B8.h"
#include "kernels/kernelFunctions/hoct4.h"
#include "kernels/kernelFunctions/gaussian.h"
#include "kernels/kernelFunctions/poly6.h"
#include "kernels/kernelFunctions/spiky.h"
#include "kernels/kernelFunctions/viscosityKernel.h"
#include "kernels/kernelFunctions/cohesionKernel.h"
#include "kernels/kernelFunctions/adhesionKernel.h"
namespace sph_native {
inline bool valid_kernel(int k) {
    switch (k) {
    case 0:
    case 1:
    case 2:
    case 20:
    case 21:
    case 30:
    case 31:
    case 32:
    case 34:
    case 33:
    case 35:
    case 36:
    case 40:
    case 41:
    case 42:
        return true;
    default:
        return false;
    }
}
template <class T, int D> SPH_HD RadialJet<T> radial_shape_raw(T q, int kernel) {
    if (q > T(1))
        return {};
    RadialJet<T> x(q, T(1));
    switch (kernel) {
    case 0:
        return wendland2_shape<T, D>(x);
    case 1:
        return wendland4_shape<T, D>(x);
    case 2:
        return wendland6_shape<T, D>(x);
    case 20:
        return poly6_shape<T, D>(x);
    case 21:
        return spiky_shape<T, D>(x);
    case 30:
        return cubicSpline_shape<T, D>(x);
    case 31:
        return quarticSpline_shape<T, D>(x);
    case 32:
        return quinticSpline_shape<T, D>(x);
    case 34:
        return B7_shape<T, D>(x);
    case 33:
        return B8_shape<T, D>(x);
    case 35:
        return hoct4_shape<T, D>(x);
    case 36:
        return gaussian_shape<T, D>(x);
    case 40:
        return viscosityKernel_shape<T, D>(x);
    case 41:
        return cohesionKernel_shape<T, D>(x);
    case 42:
        return adhesionKernel_shape<T, D>(x);
    default:
        return {};
    }
}
template <class T, int D> SPH_HD T normalization(int kernel) {
    switch (kernel) {
    case 0:
        return D == 1 ? T(1.25) : (D == 2 ? T(2.228169203286535) : T(3.3422538049298023));
    case 1:
        return D == 1 ? T(1.5) : (D == 2 ? T(2.8647889756541161) : T(4.9238560519055126));
    case 2:
        return D == 1 ? T(1.71875) : (D == 2 ? T(3.5468815889050962) : T(6.7889530412636603));
    case 20:
        return D == 1 ? T(1.09375) : (D == 2 ? T(1.2732395447351628) : T(1.5666814710608448));
    case 21:
        return D == 1 ? T(2) : (D == 2 ? T(3.183098861837907) : T(4.7746482927568605));
    case 30:
        return D == 1 ? T(2.6666666666666665)
                      : (D == 2 ? T(3.6378272706718935) : T(5.0929581789406511));
    case 31:
        return D == 1 ? T(4.069010416666667)
                      : (D == 2 ? T(6.2221751104525387) : T(9.7140468195736904));
    case 32:
        return D == 1 ? T(6.0750000000000002)
                      : (D == 2 ? T(10.194573321313079) : T(17.403593027098754));
    case 34:
        return D == 1 ? T(8.9360134548611114)
                      : (D == 2 ? T(16.21749330689801) : T(29.866424970413458));
    case 33:
        return D == 1 ? T(13.003174603174603)
                      : (D == 2 ? T(25.251756598314479) : T(49.668468335573586));
    case 35:
        return D == 1 ? T(2.0684349363000001) : (D == 2 ? T(3.7158334191) : T(6.5150499306));
    case 36:
        return D == 1 ? T(6.3830764864229232)
                      : (D == 2 ? T(40.743665431525208) : T(260.06993278665107));
    case 40:
        return D == 1 ? T(2.3873241463784303)
                      : (D == 2 ? T(0.35367765131532297) : T(2.3873241463784303));
    case 41:
        return D == 1 ? T(10.185916357881302)
                      : (D == 2 ? T(10.185916357881302) : T(10.185916357881302));
    case 42:
        return D == 1 ? T(0.0070000000000000001)
                      : (D == 2 ? T(0.0070000000000000001) : T(0.0070000000000000001));
    default:
        return T(0);
    }
}
template <class T, int D> SPH_HD T kernel_scale(int kernel) {
    switch (kernel) {
    case 0:
        return D == 1 ? T(1.620185) : (D == 2 ? T(1.897367) : T(1.9364920000000001));
    case 1:
        return D == 1 ? T(1.9364920000000001)
                      : (D == 2 ? T(2.1712389999999999) : T(2.2079399999999998));
    case 2:
        return D == 1 ? T(2.2079399999999998)
                      : (D == 2 ? T(2.4152300000000002) : T(2.4494899999999999));
    case 20:
        return D == 1 ? T(1) : (D == 2 ? T(1) : T(1));
    case 21:
        return D == 1 ? T(1) : (D == 2 ? T(1) : T(1));
    case 30:
        return D == 1 ? T(1.732051) : (D == 2 ? T(1.7780020000000001) : T(1.825742));
    case 31:
        return D == 1 ? T(1.9364920000000001)
                      : (D == 2 ? T(1.9771730000000001) : T(2.0189319999999999));
    case 32:
        return D == 1 ? T(2.121321) : (D == 2 ? T(2.158131) : T(2.1957749999999998));
    case 34:
        return D == 1 ? T(2.2912880000000002) : (D == 2 ? T(2.32517) : T(2.3597000000000001));
    case 33:
        return D == 1 ? T(2.4494899999999999)
                      : (D == 2 ? T(2.4810439999999998) : T(2.5131230000000002));
    case 35:
        return D == 1 ? T(2.2243230000000001)
                      : (D == 2 ? T(2.1924630000000001) : T(2.1896844099999999));
    case 36:
        return D == 1 ? T(8) : (D == 2 ? T(8) : T(8));
    case 40:
        return D == 1 ? T(1) : (D == 2 ? T(1) : T(1));
    case 41:
        return D == 1 ? T(1) : (D == 2 ? T(1) : T(1));
    case 42:
        return D == 1 ? T(1) : (D == 2 ? T(1) : T(1));
    default:
        return T(0);
    }
}
template <class T> SPH_HD T packing_ratio(int kernel) {
    switch (kernel) {
    case 0:
        return T(1.487);
    case 1:
        return T(1.643);
    case 2:
        return T(1.8660000000000001);
    case 20:
        return T(1);
    case 21:
        return T(1);
    case 30:
        return T(1.3146100000000001);
    case 31:
        return T(1.2030000000000001);
    case 32:
        return T(1.8222875000000001);
    case 34:
        return T(1.8222875000000001);
    case 33:
        return T(1.8222875000000001);
    case 35:
        return T(2.1579999999999999);
    case 36:
        return T(1.337);
    case 40:
        return T(1);
    case 41:
        return T(1);
    case 42:
        return T(1);
    default:
        return T(0);
    }
}
// The analytic origin derivative of every smooth radial kernel is zero.
// Factored polynomial product rules can cancel its O(q) term near zero; use
// its local derivative expansion where omitted terms are below scalar precision.
template <class T, int D> SPH_HD RadialJet<T> radial_shape(T q, int kernel) {
    auto f = radial_shape_raw<T, D>(q, kernel);
    if (q >= T(0) && q < (sizeof(T) == 4 ? T(1e-5) : T(1e-9)) && kernel != 21 && kernel != 35 &&
        kernel != 40 && kernel != 42) {
        auto origin = radial_shape_raw<T, D>(T(0), kernel);
        f.d1 = origin.d2 * q + origin.d3 * q * q / T(2);
    }
    return f;
}
} // namespace sph_native
