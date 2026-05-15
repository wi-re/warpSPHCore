import torch

from sphWarpCore.renorm.wp_covariance import pinv2x2
from sphWarpCore.pinv import pinv2x2 as pinv2x2_warp

def computeCRKTermsWarp(
        m_0: torch.Tensor, m_1: torch.Tensor, m_2: torch.Tensor,
        dm_0dgamma: torch.Tensor, dm_1dgamma: torch.Tensor, dm_2dgamma: torch.Tensor,
        num_nbrs: torch.Tensor, supports: torch.Tensor):


    m_2_det = torch.det(m_2).abs()
    m_2_inv = torch.linalg.pinv(m_2)# if m_2.shape[1] != 2 else pinv2x2(m_2)[0]
    # m_2_inv = 1/m_2 
    
    # print(f'm_2_det: {m_2_det.min():8.3g}, {m_2_det.max():8.3g}, {m_2_det.mean():8.3g} has nan: {torch.isnan(m_2_det).any()} has inf: {torch.isinf(m_2_det).any()}')
    # 
    is_singular = torch.where(m_2_det < 1e-7, 1.0, 0.0)
    # print(f'Number of singular matrices: {is_singular.sum()}')
    #     # Eq. 12.
    # ai = 1.0/(m0 - dot(temp_vec, m1, d))
    A = 1 / (m_0 - torch.einsum('nij, ni, nj -> n', m_2_inv, m_1, m_1))
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

    mask = (num_nbrs < 2) | (is_singular > 0.0)

    # if N_NBRS < 2 or is_singular > 0.0:
    # d_ai[d_idx] = 1.0
    # for i in range(d):
    #     d_gradai[d * d_idx + i] = 0.0
    #     d_bi[d * d_idx + i] = 0.0
    #     for j in range(d):
    #         d_gradbi[d2 * d_idx + d * i + j] = 0.0
    A[mask] = 1.0
    for i in range(nu):
        gradA[mask, i] = 0.0
        B[mask, i] = 0.0
        for j in range(nu):
            gradB[mask, i, j] = 0.0
            
    return A, B, gradA, gradB