"""Torch CRK factor algebra shared by Warp and native preparation passes."""
import torch


def computeCRKTerms(m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma,
                    num_nbrs, supports):
    """Return A, B, gradA and gradB; derivative axes precede component axes.

    The existing limiter returns the identity correction for fewer than two
    samples, near-singular spatial moments or a degenerate augmented system.
    Its masks stay on the tensor device. ``supports`` retains the shared call
    signature; the current limiter uses the moment determinant directly.
    """
    identity = torch.eye(m_2.shape[-1], dtype=m_2.dtype, device=m_2.device)
    mask = (num_nbrs < 2) | (torch.det(m_2).abs() < 1e-14)

    def safe_moments(mask, m0, m1, m2, dm0, dm1, dm2):
        # Make inactive algebra finite before reciprocals/inverses, including
        # their Torch adjoints. The limiter itself is a constant correction.
        return (torch.where(mask, torch.ones_like(m0), m0),
                torch.where(mask[:, None], torch.zeros_like(m1), m1),
                torch.where(mask[:, None, None], identity, m2),
                torch.where(mask[:, None], torch.zeros_like(dm0), dm0),
                torch.where(mask[:, None, None], torch.zeros_like(dm1), dm1),
                torch.where(mask[:, None, None, None], torch.zeros_like(dm2), dm2))

    m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma = safe_moments(
        mask, m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma)
    inverse = torch.linalg.pinv(m_2)
    projection = torch.einsum('nij,ni,nj->n', inverse, m_1, m_1)
    schur = m_0 - projection
    # Invertible spatial moments can still have dependent constant/linear
    # constraints. Preserve the existing scale-relative Schur limiter.
    tolerance = 32 * torch.finfo(m_0.dtype).eps * (m_0.abs() + projection.abs())
    mask = mask | (~torch.isfinite(schur)) | (schur <= tolerance)
    m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma = safe_moments(
        mask, m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma)
    inverse = torch.where(mask[:, None, None], identity, inverse)

    A = 1 / torch.where(mask, torch.ones_like(schur), schur)
    B = -torch.einsum('nij,nj->ni', inverse, m_1)
    gradient_projection = torch.einsum('nij,nj,nki->nk', inverse, m_1, dm_1dgamma)
    gradient_inverse = torch.einsum('nil,nklm,nmj,nj,ni->nk',
                                  inverse, dm_2dgamma, inverse, m_1, m_1)
    gradA = -A[:, None]**2 * (dm_0dgamma - 2 * gradient_projection + gradient_inverse)
    gradB = (-torch.einsum('nij,nkj->nki', inverse, dm_1dgamma)
             + torch.einsum('nil,nklm,nmj,nj->nki', inverse, dm_2dgamma, inverse, m_1))
    return (torch.where(mask, torch.ones_like(A), A),
            torch.where(mask[:, None], torch.zeros_like(B), B),
            torch.where(mask[:, None], torch.zeros_like(gradA), gradA),
            torch.where(mask[:, None, None], torch.zeros_like(gradB), gradB))


# Source compatibility for the existing Warp wrapper and downstream callers.
computeCRKTermsWarp = computeCRKTerms
