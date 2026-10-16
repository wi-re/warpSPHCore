import torch

# from warpSPHCore.renorm.wp_pinv2x2 import pinv2x2
# from warpSPHCore.pinv import pinv2x2 as pinv2x2_warp
from ..type_config import *

def computeCRKTermsWarp(
        m_0: torch.Tensor, m_1: torch.Tensor, m_2: torch.Tensor,
        dm_0dgamma: torch.Tensor, dm_1dgamma: torch.Tensor, dm_2dgamma: torch.Tensor,
        num_nbrs: torch.Tensor, supports: torch.Tensor):


    m_2_det = torch.det(m_2).abs()
    mask = (num_nbrs < 2) | (m_2_det < 1e-14)
    identity = torch.eye(m_2.shape[-1], dtype=m_2.dtype, device=m_2.device)

    def safe_moments(mask, m0, m1, m2, dm0, dm1, dm2):
        # The limiter is a constant identity correction. Keep inactive algebra
        # finite BEFORE reciprocals and inverses, including its adjoints.
        return (torch.where(mask, torch.ones_like(m0), m0),
                torch.where(mask[:, None], torch.zeros_like(m1), m1),
                torch.where(mask[:, None, None], identity, m2),
                torch.where(mask[:, None], torch.zeros_like(dm0), dm0),
                torch.where(mask[:, None, None], torch.zeros_like(dm1), dm1),
                torch.where(mask[:, None, None, None], torch.zeros_like(dm2), dm2))

    m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma = safe_moments(
        mask, m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma)
    m_2_inv = torch.linalg.pinv(m_2)
    projection = torch.einsum('nij, ni, nj -> n', m_2_inv, m_1, m_1)
    schur = m_0 - projection
    # Invertible spatial moments do not imply independent constant/linear
    # constraints: references on an affine plane away from the query can
    # have nonsingular m2 but singular augmented moments. Reject nonpositive
    # or cancellation-sized Schur complements relative to their own terms.
    tolerance = 32 * torch.finfo(m_0.dtype).eps * (m_0.abs() + projection.abs())
    mask = mask | (~torch.isfinite(schur)) | (schur <= tolerance)
    m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma = safe_moments(
        mask, m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma)
    m_2_inv = torch.where(mask[:, None, None], identity, m_2_inv)
    # m_2_inv = 1/m_2 
    
    # print(f'm_2_det: {m_2_det.min():scalar_t(8.3)g}, {m_2_det.max():scalar_t(8.3)g}, {m_2_det.mean():scalar_t(8.3)g} has nan: {torch.isnan(m_2_det).any()} has inf: {torch.isinf(m_2_det).any()}')
    # 
    # print(f'Number of singular matrices: {is_singular.sum()}')
    #     # Eq. 12.
    # ai = scalar_t(1.0)/(m0 - dot(temp_vec, m1, d))
    A = 1 / torch.where(mask, torch.ones_like(schur), schur)
    # # Eq. 13.
    # mat_vec_mult(m2inv, m1, d, bi)
    # for gam in range(d):
    #     bi[gam] = -bi[gam]
    B = - torch.einsum('nij, nj -> ni', m_2_inv, m_1)

    gradA = torch.zeros_like(dm_0dgamma)
    gradB = torch.zeros_like(dm_1dgamma)

    # print(gradA.shape, gradB.shape)
    nu = gradA.shape[1]
    gradATerm1 = dm_0dgamma
    gradATerm2 = torch.einsum('nij, nj, nki -> nk', m_2_inv, m_1, dm_1dgamma)
    # gradATerm3 = torch.einsum('nij, ncj, ni -> nc', m_2_inv, dm_1dgamma, m_1)
    gradATerm4 = torch.einsum('nil, nklm, nmj, nj, ni -> nk', m_2_inv, dm_2dgamma, m_2_inv, m_1, m_1)
    gradA = - (A **2).view(-1,1) * ( gradATerm1 - 2 * gradATerm2 + gradATerm4)

    gradBTerm1 = torch.einsum('nij, nkj -> nki', m_2_inv, dm_1dgamma)
    gradBTerm2 = torch.einsum('nil, nklm, nmj, nj -> nki', m_2_inv, dm_2dgamma, m_2_inv, m_1)
    gradB = -gradBTerm1 + gradBTerm2

    # num_nbrs = coo_to_csr(actualNeighbors).rowEntries

    if mask.any():
        print(f"Warning: {mask.sum()} particles have insufficient neighbors or degenerate moment systems. Applying limiter to CRK terms for these particles.")

    # if N_NBRS < 2 or is_singular > scalar_t(0.0):
    # d_ai[d_idx] = scalar_t(1.0)
    # for i in range(d):
    #     d_gradai[d * d_idx + i] = scalar_t(0.0)
    #     d_bi[d * d_idx + i] = scalar_t(0.0)
    #     for j in range(d):
    #         d_gradbi[d2 * d_idx + d * i + j] = scalar_t(0.0)
    # A = torch.where(mask.view(-1,1), (1.0), A)
    # gradA = torch.where(mask.view(-1,1).repeat(1, nu), (0.0), gradA)
    # B = torch.where(mask.view(-1,1).repeat(1, nu), (0.0), B)
    # gradB = torch.where(mask.view(-1,1,1).repeat(1, nu, nu), (0.0), gradB)

    mask_n = mask
    mask_nu = mask.view(-1, 1)
    mask_nuu = mask.view(-1, 1, 1)

    A = torch.where(mask_n, torch.ones_like(A), A)
    gradA = torch.where(mask_nu, torch.zeros_like(gradA), gradA)
    B = torch.where(mask_nu, torch.zeros_like(B), B)
    gradB = torch.where(mask_nuu, torch.zeros_like(gradB), gradB)
            
    return A, B, gradA, gradB
