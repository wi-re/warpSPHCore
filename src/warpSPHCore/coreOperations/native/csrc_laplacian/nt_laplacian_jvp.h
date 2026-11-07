#pragma once
#include "execution/explicitForceProduct.h"
#include "nt_laplacian_product.h"
namespace sph_native::laplacian_jvp {
using Buffers=force_jvp::Buffers;
using Rows=force_jvp::Rows<5>;
}
