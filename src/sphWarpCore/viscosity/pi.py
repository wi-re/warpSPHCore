import torch
from diffSPH.sphOperations.shared import getTerms, compute_xij
from diffSPH.kernels import Kernel_xi
from diffSPH.enums import ViscositySwitch

def getAlphas(particles_a, particles_b, neighborhood, solverConfig):
    viscositySwitch     = solverConfig.get('diffusionSwitch',{}).get('scheme', None)
    if viscositySwitch is not None and viscositySwitch != ViscositySwitch.NoneSwitch:
        alpha_i = particles_a.alphas[neighborhood.row]
        alpha_j = particles_b.alphas[neighborhood.col]
        return alpha_i, alpha_j
    else:
        return torch.ones(neighborhood.row.shape[0], dtype = particles_a.densities.dtype, device = particles_a.densities.device), torch.ones(neighborhood.col.shape[0], dtype = particles_b.densities.dtype, device = particles_b.densities.device) #* solverConfig.get('alpha', 1.0)
    
def compute_Pi(particles_a, particles_b, domain, neighborhood, config, useJ = False, thermalConductivity = False):
    if 'diffusion' not in config:
        config['diffusion'] = {}
    correctXi           = config.get('diffusion',{}).get('correctXi', False)
    viscosityTerm       = config.get('diffusion',{}).get('viscosityTerm', None)
    switch              = config.get('diffusion',{}).get('monaghanSwitch', True)
    use_rho_bar         = config.get('diffusion',{}).get('use_rho_bar', None)
    use_c_bar           = config.get('diffusion',{}).get('use_cbar', None)
    use_h_bar           = config.get('diffusion',{}).get('use_hbar', None)
    C_l                 = config.get('diffusion',{}).get('C_l', 1)
    C_q                 = config.get('diffusion',{}).get('C_q', 2)
    K                   = config.get('diffusion',{}).get('K', None)   
    switchScaling       = config.get('diffusion',{}).get('switchScaling', 'ij')
    viscosityFormulation= config.get('diffusion',{}).get('viscosityFormulation', 'MonaghanGingold1983')
    if thermalConductivity:
        viscosityFormulation = config['diffusion']['thermalConductivityFormulation'] if 'thermalConductivityFormulation' in config['diffusion'] else viscosityFormulation
        C_l = config['diffusion']['Cu_l'] if 'Cu_l' in config['diffusion'] else C_l
        C_q = config['diffusion']['Cu_q'] if 'Cu_q' in config['diffusion'] else C_q
    scaleBeta           = config['diffusion'].get('scaleBeta', False)

    if viscosityFormulation == 'MonaghanGingold1983':
        K               = 1 if K is None else K
        use_rho_bar     = True if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1992'
    elif viscosityFormulation == 'Cleary1998':
        K               = 1 if K is None else K
        use_rho_bar     = False if use_rho_bar is None else use_rho_bar
        use_c_bar       = False if use_c_bar is None else use_c_bar
        use_h_bar       = False if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1997'
    elif viscosityFormulation == 'Monaghan1992':
        K               = 1 if K is None else K
        use_rho_bar     = True if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1992'
    elif viscosityFormulation in ['Monaghan1997', 'Dukowicz', 'Price2012', 'Price2012_98', 'Price2008', 'Wadsley2008']:
        K               = 1 if K is None else K
        use_rho_bar     = True if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1997'
    elif viscosityFormulation == 'delta':
        K = 1 if K is None else K
        use_rho_bar     = False if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1992'
    

    rho_i, rho_j, rho_bar   = getTerms(neighborhood.row, neighborhood.col, (particles_a.densities, particles_b.densities))
    c_i, c_j, c_bar         = getTerms(neighborhood.row, neighborhood.col, (particles_a.soundspeeds, particles_b.soundspeeds))
    h_i, h_j, h_bar         = getTerms(neighborhood.row, neighborhood.col, (particles_a.supports, particles_b.supports))
    alpha_i, alpha_j        = getAlphas(particles_a, particles_b, neighborhood, config)

    d   = particles_a.positions.shape[1]
    rho = rho_bar   if use_rho_bar  else (rho_j if useJ else rho_i)
    c   = c_bar     if use_c_bar    else (c_j   if useJ else c_i)
    h   = h_bar     if use_h_bar    else (h_j   if useJ else h_i)
    
    # if viscosityFormulation == 'delta':
        # rho = rho_i / solverConfig['fluid']['rho0']
        # c = solverConfig['fluid']['c_s']
    
    kernel = config['kernel']

    xi  = Kernel_xi(kernel, particles_a.positions.shape[1]) if correctXi else 1.0

    # print(f'alpha_i: {alpha_i.shape}, min: {alpha_i.min()}, max: {alpha_i.max()}')
    # print(f'alpha_j: {alpha_j.shape}, min: {alpha_j.min()}, max: {alpha_j.max()}')    
    
    C_l = 1.0/2.0 * (alpha_i + alpha_j) * C_l
    C_q = 1.0/2.0 * (alpha_i + alpha_j) * C_q
    if switchScaling == 'i':
        # print('Scaling with i')
        C_l = C_l * alpha_i
        C_q = C_q * alpha_i
    elif switchScaling == 'j':
        # print('Scaling with j')
        C_l = C_l * alpha_j
        C_q = C_q * alpha_j
    
        
    if scaleBeta:
        # print('Scaling beta')
        C_q = C_q * C_l
        
    # print(solverConfig)
    # print(f'C_l: {C_l.shape}, min: {C_l.min()}, max: {C_l.max()}')
    # print(f'C_q: {C_q.shape}, min: {C_q.min()}, max: {C_q.max()}')
        

    x_ij, r_ij  = compute_xij(particles_a, particles_b, neighborhood, domain)
    u_ij        = particles_a.velocities[neighborhood.row] - particles_b.velocities[neighborhood.col]
    ux_ij       = torch.einsum('ij,ij->i', u_ij, x_ij)

    if viscosityTerm == 'Monaghan' or viscosityTerm == 'Monaghan1992':
        mu_ij = ux_ij / (r_ij*r_ij + 1e-14 * h*h) * h / xi
    else: # 'Monaghan1997' also referred to as 'j' in the paper
        mu_ij = ux_ij / (r_ij + 1e-14 * h)
    
    # print(f'mu_ij: {mu_ij.shape}, min: {mu_ij.min()}, max: {mu_ij.max()}')
    
    if switch:
        mu_ij[ux_ij > 0] = 0

    # There is some confusion regarding the terms here.
    # Monaghan 2005 prescribes several potential terms for the viscosity
    # Notationally we use Pi_ab = - K / rho v_sig mu_ab
    
    if viscosityFormulation == 'MonaghanGingold1983':
    # Monaghan and Gingold 1983: The terms are given in (8.3) and (8.4)  of Monaghan 2005 and are
    # Pi_ab = -nu ( v_ab \cdot r_ab) / (r_ab^2 + epsilon^2 h_ab^2)
    # nu = alpha h_bar c_bar / rho_bar
    # Rewording this slightly we get the 'Monaghan1992' viscosity Term (with xi correction)
    # combined with using c_bar, rho_bar and h_bar. 
    # Consequently this uses 
    # v_sig = c_bar
    # K = 1
        v_sig = c_bar
    elif viscosityFormulation == 'Cleary1998':
    # Cleary 1998: The terms are given in (8.8) and (8.9) of Monaghan 2005 and are
    # mu_a = 1/8 alpha_a h_a c_a rho_a
    # Pi_ab = - 16 mu_a mu_b / (rho_a rho_b (mu_a + mu_b)) mu_ij
        f = 1/(2*(d+2)) # Based on estimations based on Monaghan 2005, not given for 1D
        mu_i = f * alpha_i * C_l * h_i * c_i * rho_i / xi
        mu_j = f * alpha_j * C_l * h_j * c_j * rho_j / xi
        # 19.8 based on Cleary and Ha 2002
        v_sig = 19.8 * mu_i * mu_j / (rho_i * rho_j * (mu_i + mu_j)) / (r_ij + 1e-14 * h)
    elif viscosityFormulation == 'Monaghan1992':
    # Monaghan 1992: The term is given in (8.10) of Monaghan 2005 and is
    # mu = h / rho ( alpha c - beta mu_ij)
    # This uses the Monaghan 1992 viscosity term with alpha = 1 and beta = 2
        v_sig = C_l * c - C_q * mu_ij
    elif viscosityFormulation == 'Monaghan1997a':
    # Monaghan 1997: The term is given in (8.11) of Monaghan 2005 and is very similar
    # to the Monaghan1992 term but uses the Monaghan1997 viscosity term. denoted as j
    # in the 1997 paper and has a strange wording in 2005 of using 1.0/2.0 instead of 1 for K
    # c_i + c_j instead of c_bar and beta = 4. Cancelling these terms out gives the normal
    # c_bar term with alpha = 1 and beta = 4! This is also eq 3.7 in Monaghan1997
        v_sig = C_l * c - C_q * mu_ij
    elif viscosityFormulation == 'Monaghan1997b':
    # Based on Monaghan 1997 eq 4.7:
        v_sig = (c_i*c_i + C_q * mu_ij*mu_ij)**0.5 + (c_j*c_j + C_q * mu_ij*mu_ij)**0.5 - C_q * mu_ij
    elif viscosityFormulation == 'Dukowicz':
    # The term is given in (4.8) of Monaghan 1997 and is simply the 1997a term with a 3/4 factor
        v_sig = C_l * c - 3.0/4.0 * C_q * mu_ij
    # Next are the formulations based on Price's SPMHD paper from 2012
    elif viscosityFormulation == 'Price2012_98':
    # This term is identical to Monaghan 1992, equation 98 in Price 2012
        v_sig = C_l * c - C_q * mu_ij
    elif viscosityFormulation == 'Price2012':
    # Based on equation 103
        v_sig = C_l * c - C_q / 2.0 * mu_ij
    elif viscosityFormulation == 'Price2008':
    # This formulation and the next are only mentioned in the Price 2012 after equation 103, no explicit equation numbers
        P_i, P_j    = particles_a.pressures[neighborhood.row], particles_b.pressures[neighborhood.col]
        rho_bar     = (rho_i + rho_j) / 2.0
        v_sig       = C_l * torch.sqrt(torch.abs(P_i - P_j) / (rho_bar + 1e-14 * h))
    elif viscosityFormulation == 'Wadsley2008':
        v_sig = C_l * torch.abs(mu_ij)
    else:
        v_sig = C_l * c - C_q * mu_ij
    if thermalConductivity:
        return -K / rho * v_sig * (particles_a.internalEnergies[neighborhood.row] - particles_b.internalEnergies[neighborhood.col])
    return -K / rho * v_sig * mu_ij
    

    return -K/ rho * C_l * (c * mu_ij - C_q * mu_ij**2)


def compute_Pi_v2(particles_a, particles_b, domain, neighborhood, config, useJ = False, thermalConductivity = False):
    if 'diffusion' not in config:
        config['diffusion'] = {}
    correctXi           = config.get('diffusion',{}).get('correctXi', False)
    viscosityTerm       = config.get('diffusion',{}).get('viscosityTerm', None)
    switch              = config.get('diffusion',{}).get('monaghanSwitch', True)
    use_rho_bar         = config.get('diffusion',{}).get('use_rho_bar', None)
    use_c_bar           = config.get('diffusion',{}).get('use_cbar', None)
    use_h_bar           = config.get('diffusion',{}).get('use_hbar', None)
    C_l                 = config.get('diffusion',{}).get('C_l', 1)
    C_q                 = config.get('diffusion',{}).get('C_q', 2)
    K                   = config.get('diffusion',{}).get('K', None)   
    switchScaling       = config.get('diffusion',{}).get('switchScaling', 'ij')
    viscosityFormulation= config.get('diffusion',{}).get('viscosityFormulation', 'MonaghanGingold1983')
    if thermalConductivity:
        viscosityFormulation = config['diffusion']['thermalConductivityFormulation'] if 'thermalConductivityFormulation' in config['diffusion'] else viscosityFormulation
        C_l = config['diffusion']['Cu_l'] if 'Cu_l' in config['diffusion'] else C_l
        C_q = config['diffusion']['Cu_q'] if 'Cu_q' in config['diffusion'] else C_q
    scaleBeta           = config['diffusion'].get('scaleBeta', False)

    if viscosityFormulation == 'MonaghanGingold1983':
        K               = 1 if K is None else K
        use_rho_bar     = True if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1992'
    elif viscosityFormulation == 'Cleary1998':
        K               = 1 if K is None else K
        use_rho_bar     = False if use_rho_bar is None else use_rho_bar
        use_c_bar       = False if use_c_bar is None else use_c_bar
        use_h_bar       = False if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1997'
    elif viscosityFormulation == 'Monaghan1992':
        K               = 1 if K is None else K
        use_rho_bar     = True if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1992'
    elif viscosityFormulation in ['Monaghan1997', 'Dukowicz', 'Price2012', 'Price2012_98', 'Price2008', 'Wadsley2008']:
        K               = 1 if K is None else K
        use_rho_bar     = True if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1997'
    elif viscosityFormulation == 'delta':
        K = 1 if K is None else K
        use_rho_bar     = False if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1992'
    

    rho_i, rho_j, rho_bar   = getTerms(neighborhood.row, neighborhood.col, (particles_a.densities, particles_b.densities))
    c_i, c_j, c_bar         = getTerms(neighborhood.row, neighborhood.col, (particles_a.soundspeeds, particles_b.soundspeeds))
    h_i, h_j, h_bar         = getTerms(neighborhood.row, neighborhood.col, (particles_a.supports, particles_b.supports))
    alpha_i, alpha_j        = getAlphas(particles_a, particles_b, neighborhood, config)

    d   = particles_a.positions.shape[1]
    rho = rho_bar   if use_rho_bar  else (rho_j if useJ else rho_i)
    c   = c_bar     if use_c_bar    else (c_j   if useJ else c_i)
    h   = h_bar     if use_h_bar    else (h_j   if useJ else h_i)
    
    # if viscosityFormulation == 'delta':
        # rho = rho_i / solverConfig['fluid']['rho0']
        # c = solverConfig['fluid']['c_s']
    
    kernel = config['kernel']

    xi  = Kernel_xi(kernel, particles_a.positions.shape[1]) if correctXi else 1.0

    # print(f'alpha_i: {alpha_i.shape}, min: {alpha_i.min()}, max: {alpha_i.max()}')
    # print(f'alpha_j: {alpha_j.shape}, min: {alpha_j.min()}, max: {alpha_j.max()}')    
    
    C_l = 1.0/2.0 * (alpha_i + alpha_j) * C_l
    C_q = 1.0/2.0 * (alpha_i + alpha_j) * C_q
    if switchScaling == 'i':
        # print('Scaling with i')
        C_l = C_l * alpha_i
        C_q = C_q * alpha_i
    elif switchScaling == 'j':
        # print('Scaling with j')
        C_l = C_l * alpha_j
        C_q = C_q * alpha_j
    
        
    if scaleBeta:
        # print('Scaling beta')
        C_q = C_q * C_l
        
    # print(solverConfig)
    # print(f'C_l: {C_l.shape}, min: {C_l.min()}, max: {C_l.max()}')
    # print(f'C_q: {C_q.shape}, min: {C_q.min()}, max: {C_q.max()}')
        

    x_ij, r_ij  = compute_xij(particles_a, particles_b, neighborhood, domain)
    u_ij        = particles_a.velocities[neighborhood.row] - particles_b.velocities[neighborhood.col]
    ux_ij       = torch.einsum('ij,ij->i', u_ij, x_ij)

    if viscosityTerm == 'Monaghan' or viscosityTerm == 'Monaghan1992':
        mu_ij = ux_ij / (r_ij*r_ij + 1e-14 * h*h) * h / xi
    else: # 'Monaghan1997' also referred to as 'j' in the paper
        mu_ij = ux_ij / (r_ij + 1e-14 * h)
    
    # print(f'mu_ij: {mu_ij.shape}, min: {mu_ij.min()}, max: {mu_ij.max()}')
    
    if switch:
        mu_ij[ux_ij > 0] = 0

    # There is some confusion regarding the terms here.
    # Monaghan 2005 prescribes several potential terms for the viscosity
    # Notationally we use Pi_ab = - K / rho v_sig mu_ab
    
    if viscosityFormulation == 'MonaghanGingold1983':
    # Monaghan and Gingold 1983: The terms are given in (8.3) and (8.4)  of Monaghan 2005 and are
    # Pi_ab = -nu ( v_ab \cdot r_ab) / (r_ab^2 + epsilon^2 h_ab^2)
    # nu = alpha h_bar c_bar / rho_bar
    # Rewording this slightly we get the 'Monaghan1992' viscosity Term (with xi correction)
    # combined with using c_bar, rho_bar and h_bar. 
    # Consequently this uses 
    # v_sig = c_bar
    # K = 1
        v_sig = c_bar
    elif viscosityFormulation == 'Cleary1998':
    # Cleary 1998: The terms are given in (8.8) and (8.9) of Monaghan 2005 and are
    # mu_a = 1/8 alpha_a h_a c_a rho_a
    # Pi_ab = - 16 mu_a mu_b / (rho_a rho_b (mu_a + mu_b)) mu_ij
        f = 1/(2*(d+2)) # Based on estimations based on Monaghan 2005, not given for 1D
        mu_i = f * alpha_i * C_l * h_i * c_i * rho_i / xi
        mu_j = f * alpha_j * C_l * h_j * c_j * rho_j / xi
        # 19.8 based on Cleary and Ha 2002
        v_sig = 19.8 * mu_i * mu_j / (rho_i * rho_j * (mu_i + mu_j)) / (r_ij + 1e-14 * h)
    elif viscosityFormulation == 'Monaghan1992':
    # Monaghan 1992: The term is given in (8.10) of Monaghan 2005 and is
    # mu = h / rho ( alpha c - beta mu_ij)
    # This uses the Monaghan 1992 viscosity term with alpha = 1 and beta = 2
        v_sig = C_l * c - C_q * mu_ij
    elif viscosityFormulation == 'Monaghan1997a':
    # Monaghan 1997: The term is given in (8.11) of Monaghan 2005 and is very similar
    # to the Monaghan1992 term but uses the Monaghan1997 viscosity term. denoted as j
    # in the 1997 paper and has a strange wording in 2005 of using 1.0/2.0 instead of 1 for K
    # c_i + c_j instead of c_bar and beta = 4. Cancelling these terms out gives the normal
    # c_bar term with alpha = 1 and beta = 4! This is also eq 3.7 in Monaghan1997
        v_sig = C_l * c - C_q * mu_ij
    elif viscosityFormulation == 'Monaghan1997b':
    # Based on Monaghan 1997 eq 4.7:
        v_sig = (c_i**2 + C_q * mu_ij**2)**0.5 + (c_j**2 + C_q * mu_ij**2)**0.5 - C_q * mu_ij
    elif viscosityFormulation == 'Dukowicz':
    # The term is given in (4.8) of Monaghan 1997 and is simply the 1997a term with a 3/4 factor
        v_sig = C_l * c - 3/4 * C_q * mu_ij
    # Next are the formulations based on Price's SPMHD paper from 2012
    elif viscosityFormulation == 'Price2012_98':
    # This term is identical to Monaghan 1992, equation 98 in Price 2012
        v_sig = C_l * c - C_q * mu_ij
    elif viscosityFormulation == 'Price2012':
    # Based on equation 103
        v_sig = C_l * c - C_q / 2 * mu_ij
    elif viscosityFormulation == 'Price2008':
    # This formulation and the next are only mentioned in the Price 2012 after equation 103, no explicit equation numbers
        P_i, P_j    = particles_a.pressures[neighborhood.row], particles_b.pressures[neighborhood.col]
        rho_bar     = (rho_i + rho_j) / 2
        v_sig       = C_l * torch.sqrt(torch.abs(P_i - P_j) / (rho_bar + 1e-14 * h))
    elif viscosityFormulation == 'Wadsley2008':
        v_sig = C_l * torch.abs(mu_ij)
    else:
        v_sig = C_l * c - C_q * mu_ij
    val = K / rho * v_sig #* mu_ij
    if thermalConductivity:
        val = K / rho * v_sig #* (particles_a.internalEnergies[neighborhood.row] - particles_b.internalEnergies[neighborhood.col])
    # else:
    if viscosityTerm == 'Monaghan' or viscosityTerm == 'Monaghan1992':
        val = val * h / xi
    else:
        val = val * r_ij
    
    if switch and not thermalConductivity:
        val[ux_ij > 0] = 0
    return 1.0/2.0 * val
    

    return -K/ rho * C_l * (c * mu_ij - C_q * mu_ij**2)

import warp as wp
from enum import Enum

class ViscosityTerms(Enum):
    Default = 0
    MonaghanGingold1983 = 1
    Cleary1998 = 2
    Monaghan1992 = 3
    Monaghan1997a = 4
    Monaghan1997b = 5
    Dukowicz = 6
    Price2012_98 = 7
    Price2012 = 8
    Price2008 = 9
    Wadsley2008 = 10
    DeltaSPH = 11


from ..kernels.wp_kernel import sphKernel_xi
from ..mathutil import computeDistanceVec, safe_sqrt

@wp.func
def compute_mu_ij(
    ux_ij: wp.float32, r_ij: wp.float32, h: wp.float32, viscosityTerm: wp.int32, xi: wp.float32
):
    mu_ij = ux_ij / (r_ij + 1e-14 * h) # Always start with this as the base

    scalingFactor = h / (xi * (r_ij + 1e-14 * h))
    scaled_mu_ij = mu_ij * scalingFactor
    
    if viscosityTerm == 0: # Default to Monaghan1992
        return scaled_mu_ij, h / xi
    elif viscosityTerm == 1: # MonaghanGingold1983
        return scaled_mu_ij, h / xi
    elif viscosityTerm == 2: # Cleary1998
        return mu_ij, r_ij
    elif viscosityTerm == 3: # Monaghan1992
        return scaled_mu_ij, h / xi
    elif viscosityTerm == 4: # Monaghan1997a
        return mu_ij, r_ij
    elif viscosityTerm == 5: # Monaghan1997b
        return mu_ij, r_ij
    elif viscosityTerm == 6: # Dukowicz
        return mu_ij, r_ij
    elif viscosityTerm == 7: # Price2012_98
        return mu_ij, r_ij
    elif viscosityTerm == 8: # Price2012
        return mu_ij, r_ij
    elif viscosityTerm == 9: # Price2008
        return mu_ij, r_ij
    elif viscosityTerm == 10: # Wadsley2008
        return mu_ij, r_ij
    elif viscosityTerm == 11: # DeltaSPH
        return scaled_mu_ij, h / xi
    else:
        return scaled_mu_ij, h / xi

@wp.func
def compute_bars(
    rho_i : wp.float32, rho_j : wp.float32, rho_bar : wp.float32, 
    c_i : wp.float32, c_j : wp.float32, c_bar : wp.float32,
    h_i : wp.float32, h_j : wp.float32, h_bar : wp.float32,
    viscosityTerm: wp.int32, useJ: bool
):
    use_rho_bar = wp.bool(False)
    use_c_bar = wp.bool(False)
    use_h_bar = wp.bool(False)

    if viscosityTerm == 0:
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 1: # MonaghanGingold1983
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 2: # Cleary1998
        use_rho_bar = False
        use_c_bar = False
        use_h_bar = False
    elif viscosityTerm == 3: # Monaghan1992
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 4: # Monaghan1997a
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 5: # Monaghan1997b
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 6: # Dukowicz
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 7: # Price2012_98
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 8: # Price2012    
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 9: # Price2008
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 10: # Wadsley2008
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True
    elif viscosityTerm == 11: # DeltaSPH
        use_rho_bar = False
        use_c_bar = True
        use_h_bar = True
    else:        
        use_rho_bar = True
        use_c_bar = True
        use_h_bar = True

    rho = wp.float32(0.0)
    c = wp.float32(0.0)
    h = wp.float32(0.0)

    if use_rho_bar:
        rho = rho_bar
    else:
        if useJ:
            rho = rho_j
        else:            
            rho = rho_i    

    if use_c_bar:
        c = c_bar
    else:
        if useJ:
            c = c_j
        else:            
            c = c_i

    if use_h_bar:
        h = h_bar
    else:
        if useJ:
            h = h_j
        else:            
            h = h_i

    return rho, c, h
import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Optional
import torch
from ..utils.wp_autograd import *
from ..radiusSearch.radius_util import convertModeToUint

from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *

@wp.func
def computePi_actual(
    x_i: vector(dtype = wp.float32, length=Any), x_j:  vector(dtype = wp.float32, length=Any), # type: ignore
    h_i: wp.float32, h_j: wp.float32, # type: ignore
    m_i: wp.float32, m_j: wp.float32, # type: ignore
    rho_i: wp.float32, rho_j: wp.float32, # type: ignore
    v_i: vector(dtype = wp.float32, length=Any), v_j: vector(dtype = wp.float32, length=Any), # type: ignore
    domainMin: wp.array(dtype = wp.float32), domainMax: wp.array(dtype = wp.float32), periodicity: wp.array(dtype = wp.bool), # type: ignore
    dim: wp.int32,
    kernel_int : wp.int32,

    c_s: wp.float32, # Speed of sound, used in some formulations to compute the signal velocity
    C_l: wp.float32, C_q: wp.float32, # Viscosity coefficients also referred to as alpha and beta in some formulations
    K_: wp.float32, # Overall viscosity scaling factor
    viscosityTerm: wp.int32, # Viscosity formulation to use, e.g. Monaghan1992, Monaghan1997, Cleary1998 etc.
    scaleBeta : wp.bool = False, # If true then the quadratic viscosity term is scaled by the linear viscosity term, as suggested in some papers to reduce excessive viscosity in certain scenarios. This is only relevant for formulations that use a quadratic term, such as Monaghan1992 and Monaghan1997.
    switch : wp.bool = True, # Whether to apply the Monaghan switch that turns off viscosity for diverging particles, i.e. particles that are moving away from each other. This is a common technique to reduce excessive viscosity in expanding flows and is used in many formulations such as Monaghan1992 and Monaghan1997.
    correctXi : wp.bool = False, # Whether to apply the xi correction factor to the viscosity term. This is a correction factor that can be applied to account for errors in the estimation of the velocity divergence and is discussed in some papers such as "Correcting SPH for accurate viscous forces" by Adami et al. 2013.
    useJ : wp.bool = False, # Whether to use the properties of the j particle instead of the i particle in the viscosity computation. This can be relevant for certain formulations and scenarios, such as when computing the viscosity force on particle i due to particle j, it might make sense to use the properties of particle j in the computation. This is also related to the use of rho_bar, c_bar and h_bar which are typically computed as averages of the i and j particle properties.
):
    rho_bar = 1.0/2.0 * (rho_i + rho_j)

    c_i = c_s
    c_j = c_s
    c_bar = c_s
    alpha_i = 1.0
    alpha_j = 1.0 # No viscosity switch here

    h_bar = 1.0/2.0 * (h_i + h_j)

    xi = sphKernel_xi(kernel_int, dim) if correctXi else 1.0

    C_l = 1.0/2.0 * (alpha_i + alpha_j) * C_l
    C_q = 1.0/2.0 * (alpha_i + alpha_j) * C_q
    if scaleBeta:
        C_q = C_q * C_l

    x_ij = computeDistanceVec(x_i, x_j, periodicity, domainMin, domainMax)
    r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

    u_ij = v_i - v_j
    ux_ij = wp.dot(u_ij, x_ij)

    mu_ij, scalingFactor = compute_mu_ij(ux_ij, r_ij, h_bar, viscosityTerm, xi)

    if switch and ux_ij > 0:
        mu_ij = 0.0

    v_sig = wp.float32(0.0)
    K = wp.float32(1.0)

    rho, c, h = compute_bars(
        rho_i, rho_j, rho_bar, 
        c_i, c_j, c_bar, 
        h_i, h_j, h_bar, 
        viscosityTerm, useJ)

    if viscosityTerm == 1: # MonaghanGingold1983
        # Monaghan and Gingold 1983: The terms are given in (8.3) and (8.4)  of Monaghan 2005 and are
        # Pi_ab = -nu ( v_ab \cdot r_ab) / (r_ab^2 + epsilon^2 h_ab^2)
        # nu = alpha h_bar c_bar / rho_bar
        # Rewording this slightly we get the 'Monaghan1992' viscosity Term (with xi correction)
        # combined with using c_bar, rho_bar and h_bar. 
        # Consequently this uses 
        # v_sig = c_bar
        # K = 1
        v_sig = c_bar
        K = 1.0
    elif viscosityTerm == 2: # Cleary1998
        # Cleary 1998: The terms are given in (8.8) and (8.9) of Monaghan 2005 and are
        # mu_a = 1/8 alpha_a h_a c_a rho_a
        # Pi_ab = - 16 mu_a mu_b / (rho_a rho_b (mu_a + mu_b)) mu_ij
        f = 1.0/(2.0*(wp.float32(dim)+2.0)) # Based on estimations based on Monaghan 2005, not given for 1D
        mu_i = f * alpha_i * C_l * h_i * c_i * rho_i / xi
        mu_j = f * alpha_j * C_l * h_j * c_j * rho_j / xi
        # 19.8 based on Cleary and Ha 2002
        v_sig = 19.8 * mu_i * mu_j / (rho_i * rho_j * (mu_i + mu_j)) / (r_ij + 1e-14 * h)
        K = 1.0
    elif viscosityTerm == 3: # Monaghan1992         
        # Monaghan 1992: The term is given in (8.10) of Monaghan 2005 and is
        # mu = h / rho ( alpha c - beta mu_ij)
        # This uses the Monaghan 1992 viscosity term with alpha = 1 and beta = 2
        v_sig = C_l * c - C_q * mu_ij
        K = 1.0
    elif viscosityTerm == 4: # Monaghan1997a
        # Monaghan 1997: The term is given in (8.11) of Monaghan 2005 and is very similar
        # to the Monaghan1992 term but uses the Monaghan1997 viscosity term. denoted as j
        # in the 1997 paper and has a strange wording in 2005 of using 1.0/2.0 instead of 1 for K
        # c_i + c_j instead of c_bar and beta = 4. Cancelling these terms out gives the normal
        # c_bar term with alpha = 1 and beta = 4! This is also eq 3.7 in Monaghan1997
        v_sig = C_l * c - C_q * mu_ij
        K = 1.0
    elif viscosityTerm == 5: # Monaghan1997b
        # Based on Monaghan 1997 eq 4.7:
        v_sig = safe_sqrt(c_i*c_i + C_q * mu_ij*mu_ij) + safe_sqrt(c_j*c_j + C_q * mu_ij*mu_ij) - C_q * mu_ij
        K = 1.0
    elif viscosityTerm == 6: # Dukowicz
        # The term is given in (4.8) of Monaghan 1997 and is simply the 1997a term with a 3/4 factor
        v_sig = C_l * c - 3.0/4.0 * C_q * mu_ij
        K = 1.0
    elif viscosityTerm == 7: # Price2012_98
        # This term is identical to Monaghan 1992, equation 98 in Price 2012
        v_sig = C_l * c - C_q * mu_ij
        K = 1.0
    elif viscosityTerm == 8: # Price2012
        # Based on equation 103
        v_sig = C_l * c - C_q / 2.0 * mu_ij
        K = 1.0
    elif viscosityTerm == 9: # Price2008
        # This formulation and the next are only mentioned in the Price 2012 after equation 103, no explicit equation numbers
        # P_i = queryPressures[i]
        # P_j = referencePressures[j]
        # rho_bar = (rho_i + rho_j) / 2
        # v_sig = C_l * safe_sqrt(wp.abs(P_i - P_j) / (rho_bar + 1e-14 * h))
        # K = 1.0
        # Since we don't have access to the pressures here, we can use an approximation based on the ideal gas law, P = rho * c^2
        P_i = rho_i * c_i * c_i
        P_j = rho_j * c_j * c_j
        rho_bar = (rho_i + rho_j) / 2.0
        v_sig = C_l * safe_sqrt(wp.abs(P_i - P_j) / (rho_bar + 1e-14 * h))
        K = 1.0
    elif viscosityTerm == 10: # Wadsley2008
        v_sig = C_l * wp.abs(mu_ij)
        K = 1.0
    elif viscosityTerm == 11: # DeltaSPH
        v_sig = C_l * c - C_q * mu_ij
        K = 1.0

    val = K / rho * v_sig * scalingFactor

    return 1.0/2.0 * val

from ..operations.wp_gradient import computeSPHGradientTensor_Func


@wp.func
def computePiViscosityKernel_Func(   
    xi: vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32, rhoi: wp.float32, fi : vector(dtype = wp.float32, length=Any), # type: ignore
    
    positions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), supports : wp.array(dtype = wp.float32), masses: wp.array(dtype = wp.float32), densities: wp.array(dtype = wp.float32), values: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32, 
    
    neighborList: wp.array(dtype = wp.int64), # type: ignore
    neighborOffset : wp.int64, numNeighs: wp.int32, 
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    
    outputValue: vector(length=Any, dtype=wp.float32), # type: ignore

    c_s: wp.float32,
    C_l: wp.float32, C_q: wp.float32,
    K_: wp.float32,
    viscosityTerm: wp.int32,
    scaleBeta: wp.bool,
    switch: wp.bool,
    correctXi: wp.bool,
    useJ: wp.bool,
):
    
    f_interpolated = type(fi)(0.0)
    dim = wp.int32(xi.length)
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + wp.int64(neighborIndex)
        j = wp.int32(neighborList[jj])
        
        mj = masses[j]
        rhoj = densities[j]
        apparentVolume = mj / rhoj

        Pi_ij = computePi_actual(
            xi, positions[j],
            hi, supports[j],
            mi, mj,
            rhoi, rhoj,
            fi, values[j],
            domainMin, domainMax, periodicity,
            dim,
            kernel_int,
            c_s, C_l, C_q, K_, viscosityTerm, scaleBeta, switch, correctXi, useJ
        )
        kernelGradient = sphKernelGradient(xi, positions[j], hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)
        f_interpolated += apparentVolume * Pi_ij * kernelGradient

    
    return f_interpolated

@wp.kernel
def computeViscosityKernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),  # type: ignore
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32), # type: ignore
    queryValues: wp.array(dtype =vector(dtype = wp.float32, length=Any)), referenceValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int64), numNeighbors: wp.array(dtype = wp.int64), 
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    
    c_s: wp.float32, # Speed of sound, used in some formulations to compute the signal velocity
    C_l: wp.float32, C_q: wp.float32, # Viscosity coefficients also referred to as alpha and beta in some formulations
    K_: wp.float32, # Overall viscosity scaling factor
    viscosityTerm: wp.int32, # Viscosity formulation to use, e.g. Monaghan1992, Monaghan1997, Cleary1998 etc.
    scaleBeta : wp.bool, # If true then the quadratic viscosity term is scaled by the linear viscosity term
    switch : wp.bool, # Whether to apply the Monaghan switch that turns off viscosity for diverging particles
    correctXi : wp.bool, # Whether to apply the xi correction factor
    useJ : wp.bool, # Whether to use the properties of the j particle instead of the i particle
    outputValues : wp.array(dtype = vector(length=Any, dtype = wp.float32)), # type: ignore
):
                                                                      
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    xi = queryPositions[i]
    hi = querySupports[i]
    mi = queryMasses[i]
    rhoi = queryDensities[i]
    fi = queryValues[i]

    outputValues[i] = computePiViscosityKernel_Func(
        xi, hi, mi, rhoi, fi,
        referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,
        periodicity, domainMin, domainMax,
        mode_uint, kernel_int,
        neighborList, neighborListRowOffsets[i], wp.int32(numNeighbors[i]),
        numDims, flatInputShape, flatOutputShape,
        type(outputValues[i])(0.0),
        c_s, C_l, C_q, K_, viscosityTerm, scaleBeta, switch, correctXi, useJ
    )

def computePiViscosity(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues : Optional[torch.Tensor], referenceValues : Optional[torch.Tensor],
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,    
    adjacency: AdjacencyListWarp,
    
    c_s: float,
    C_l: float, C_q: float,
    K: float,
    viscosityTerm: ViscosityTerms,
    scaleBeta: bool,
    switch: bool,
    correctXi: bool,
    useJ: bool,        
):
    domainMin = domain.min
    domainMax = domain.max
    periodicity = domain.periodic

    modeUint = wp.uint32(mode.value)
    kernelInt = wp.int32(kernel.value)
    outputShape = queryValues.shape[0]
    

    inputShape = queryValues.shape[1:]
    flatInputShape = 1
    for dim in inputShape:
        flatInputShape *= dim
        
    # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
    outputSize = (queryPositions.shape[0])
    # For the output shape we keep the same shape as the input as the laplacian of a scalar field is still a scalar field, and the laplacian of a vector field is still a vector field. We just need to make sure to flatten the inner dimensions for the warp kernel.
    outputShape = inputShape

    flatOutputShape = 1
    for dim in outputShape:
        flatOutputShape *= dim
    numDims = len(inputShape)

    wpValues = castTorchToWarpAsBuiltins(queryValues)

    warp_interpolation = warpWrapper(
        launch_kernel, computeViscosityKernel, outputSize, vector(length=flatOutputShape, dtype = wp.float32),
        queryPositions, referencePositions,
        querySupports, referenceSupports,
        queryMasses, referenceMasses,
        queryDensities, referenceDensities,
        queryValues, referenceValues,
        
        domainMin, domainMax, periodicity,
        modeUint,
        kernelInt,
        
        adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors,
        wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape),
        
        c_s, C_l, C_q, K, viscosityTerm.value, scaleBeta, switch, correctXi, useJ

    )

    return warp_interpolation

