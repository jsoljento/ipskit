import numpy as np

def error_analysis(mag_up,mag_up_std,mag_down,mag_down_std,pla_up,
					    pla_up_std,pla_down,pla_down_std,char_up,char_up_std,
                        Te_std,doo_Cs_doo_Te,doo_Vms_doo_Te,doo_beta_doo_Te,
                        V_shock,MA,Mms,normal,normal_type,shock_type):
    
    #magnetic field 
    B_u = mag_up[0]
    B_u_std = mag_up_std[0]
    Bx_u = mag_up[1]
    Bx_u_std = mag_up_std[1]
    By_u = mag_up[2]
    By_u_std = mag_up_std[2]
    Bz_u = mag_up[3]
    Bz_u_std = mag_up_std[3]
    B_d = mag_down[0]
    B_d_std = mag_down_std[0]
    Bx_d = mag_down[1]
    Bx_d_std = mag_down_std[1]
    By_d = mag_down[2]
    By_d_std = mag_down_std[2]
    Bz_d = mag_down[3]
    Bz_d_std = mag_down_std[3]

    #plasma data
    Np_u = pla_up[0]
    Np_u_std = pla_up_std[0]
    Np_d = pla_down[0]
    Np_d_std = pla_down_std[0]
    Tp_u = pla_up[1]
    Tp_u_std = pla_up_std[1]
    Tp_d = pla_down[1]
    Tp_d_std = pla_down_std[1]
    V_u = pla_up[2]
    V_u_std = pla_up_std[2]
    V_d = pla_down[2]
    V_d_std = pla_down_std[2]
    Vx_u = pla_up[3]
    Vx_u_std = pla_up_std[3]
    Vx_d = pla_down[3]
    Vx_d_std = pla_down_std[3]
    Vy_u = pla_up[4]
    Vy_u_std = pla_up_std[4]
    Vy_d = pla_down[4]
    Vy_d_std = pla_down_std[4]
    Vy_u = pla_up[4]
    Vy_u_std = pla_up_std[4]
    Vz_u = pla_up[5]
    Vz_u_std = pla_up_std[5]
    Vz_d = pla_down[5]
    Vz_d_std = pla_down_std[5]

    #plasma characteristics
    Cs = char_up[0]
    Cs_std = char_up_std[0] #std resulting from variations of Tp
    VA = char_up[1]
    VA_std = char_up_std[1] #std resulting from variations of B and Np
    Vms = char_up[2]
    Vms_std = char_up_std[2] #std resulting from variations of B, Np and Tp
    beta = char_up[3]
    beta_std = char_up_std[3] #std resulting from variations of Tp and Np
    Dp = char_up[4]
    Dp_std = char_up_std[4]

    #shock speed
    Vsh = V_shock
    #shock normal
    nx = normal[0]
    ny = normal[1]
    nz = normal[2]



#-------------------------------------------------------------------------------
# Errors of the variables dependent on electron temperature
#-------------------------------------------------------------------------------

    #since original standard deviation of the variables (e.g. Cs_std) resulting 
    #from fluctuations in other plasma parameters (but not from constant Te), 
    #and theoretically estimated error resulting for the Te-estimate are 
    #independent they can be combined using simple square addition
    Cs_std = np.sqrt(Cs_std**2 + doo_Cs_doo_Te**2*Te_std**2)
    Vms_std = np.sqrt(Vms_std**2 + doo_Vms_doo_Te**2*Te_std**2)
    beta_std = np.sqrt(beta_std**2 + doo_beta_doo_Te**2*Te_std**2)

#-------------------------------------------------------------------------------
# Errors for the ratios and velocity jump
#-------------------------------------------------------------------------------

    s_B_rat = np.sqrt((B_d/B_u**2)**2*B_u_std**2+(B_d_std/B_u)**2)
    s_Np_rat = np.sqrt((Np_d/Np_u**2)**2*Np_u_std**2+(Np_d_std/Np_u)**2)
    s_Tp_rat = np.sqrt((Tp_d/Tp_u**2)**2*Tp_u_std**2+(Tp_d_std/Tp_u)**2)
    s_V_jump = np.sqrt(V_u_std**2+V_d_std**2)


#-------------------------------------------------------------------------------
# Errors analysis of the shock normal
#-------------------------------------------------------------------------------

    delta_Bx = Bx_d-Bx_u
    delta_By = By_d-By_u
    delta_Bz = Bz_d-Bz_u
    delta_Vx = Vx_d-Vx_u
    delta_Vy = Vy_d-Vy_u
    delta_Vz = Vz_d-Vz_u

    #order of partial derivatives [doo/(doo Bx_d),doo/(doo Bx_u),doo/(doo Vx_d),
    #                              doo/(doo Vx_d),doo/(doo By_d),doo/(doo By_u),
    #                              ...]
    #altogether 12 partial derivatives

    # the standard deviation vector in same order
    std_vec_normal = [Bx_d_std,Bx_u_std,Vx_d_std,Vx_u_std,
                      By_d_std,By_u_std,Vy_d_std,Vy_u_std,
                      Bz_d_std,Bz_u_std,Vz_d_std,Vz_u_std]
    
    #normal definition: n = [nx,ny,nz]
    #nx = Nx/norm(N) = Nx/sqrt(Nx^2+Ny^2+Nz^2)
    delta_B = np.array([delta_Bx, delta_By, delta_Bz])
    delta_V = np.array([delta_Vx, delta_Vy, delta_Vz])
    B_vec_u = np.array([Bx_u,By_u,Bz_u])
    B_vec_d = np.array([Bx_d,By_d,Bz_d])
    V_vec_u = np.array([Vx_u,Vy_u,Vz_u])
    V_vec_d = np.array([Vx_d,Vy_d,Vz_d])

    if normal_type == 0:
        # MX3-method: N = cross(delta_B, cross(delta_B, delta_V))
        N = np.cross(delta_B, np.cross(delta_B, delta_V))

        # Extract components for readability
        delta_Bx, delta_By, delta_Bz = delta_B
        delta_Vx, delta_Vy, delta_Vz = delta_V

        # Partial derivatives of each component of N
        der_Nx = np.zeros(12)
        der_Nx[0] = delta_By * delta_Vy + delta_Bz * delta_Vz
        der_Nx[1] = -der_Nx[0]
        der_Nx[2] = -delta_By**2 - delta_Bz**2
        der_Nx[3] = -der_Nx[2]
        der_Nx[4] = delta_Bx * delta_Vy - 2 * delta_By * delta_Vx
        der_Nx[5] = -der_Nx[4]
        der_Nx[6] = delta_Bx * delta_By
        der_Nx[7] = -der_Nx[6]
        der_Nx[8] = delta_Bx * delta_Vz - 2 * delta_Bz * delta_Vx
        der_Nx[9] = -der_Nx[8]
        der_Nx[10] = delta_Bx * delta_Bz
        der_Nx[11] = -der_Nx[10]

        der_Ny = np.zeros(12)
        der_Ny[0] = delta_By * delta_Vx - 2 * delta_Bx * delta_Vy
        der_Ny[1] = -der_Ny[0]
        der_Ny[2] = delta_Bx * delta_By
        der_Ny[3] = -der_Ny[2]
        der_Ny[4] = delta_Bz * delta_Vz + delta_Bx * delta_Vx
        der_Ny[5] = -der_Ny[4]
        der_Ny[6] = -delta_Bz**2 - delta_Bx**2
        der_Ny[7] = -der_Ny[6]
        der_Ny[8] = delta_By * delta_Vz - 2 * delta_Bz * delta_Vy
        der_Ny[9] = -der_Ny[8]
        der_Ny[10] = delta_Bz * delta_By
        der_Ny[11] = -der_Ny[10]

        der_Nz = np.zeros(12)
        der_Nz[0] = delta_Bz * delta_Vx - 2 * delta_Bx * delta_Vz
        der_Nz[1] = -der_Nz[0]
        der_Nz[2] = delta_Bx * delta_Bz
        der_Nz[3] = -der_Nz[2]
        der_Nz[4] = delta_Bz * delta_Vy - 2 * delta_By * delta_Vz
        der_Nz[5] = -der_Nz[4]
        der_Nz[6] = delta_By * delta_Bz
        der_Nz[7] = -der_Nz[6]
        der_Nz[8] = delta_Bx * delta_Vx + delta_By * delta_Vy
        der_Nz[9] = -der_Nz[8]
        der_Nz[10] = -delta_Bx**2 - delta_By**2
        der_Nz[11] = -der_Nz[10]


    if normal_type == 1:
        # MC-method: N = cross(delta_B, cross(B_vec_d, B_vec_u))
        N = np.cross(delta_B, np.cross(B_vec_d, B_vec_u))

        # Extract components for readability
        delta_Bx, delta_By, delta_Bz = delta_B
        Bx_d, By_d, Bz_d = B_vec_d
        Bx_u, By_u, Bz_u = B_vec_u

        # Partial derivatives of each component of N
        der_Nx = np.zeros(12)
        der_Nx[0] = By_u * delta_By + Bz_u * delta_Bz
        der_Nx[1] = -By_d * delta_By - Bz_d * delta_Bz
        der_Nx[4] = Bx_d * By_u - Bx_u * (2 * By_d - By_u)
        der_Nx[5] = Bx_u * By_d - Bx_d * (2 * By_u - By_d)
        der_Nx[8] = Bx_d * Bz_u - Bx_u * (2 * Bz_d - Bz_u)
        der_Nx[9] = Bx_u * Bz_d - Bx_d * (2 * Bz_u - Bz_d)

        der_Ny = np.zeros(12)
        der_Ny[0] = By_d * Bx_u + By_u * (2 * Bx_d - Bx_u)
        der_Ny[1] = By_u * Bx_d + By_d * (2 * Bx_u - Bx_d)
        der_Ny[4] = Bz_u * delta_Bz + Bx_u * delta_Bx
        der_Ny[5] = -Bz_d * delta_Bz - Bx_d * delta_Bx
        der_Ny[8] = By_d * Bz_u - By_u * (2 * Bz_d - Bz_u)
        der_Ny[9] = By_u * Bz_d - By_d * (2 * Bz_u - Bz_d)

        der_Nz = np.zeros(12)
        der_Nz[0] = Bz_d * Bx_u + Bz_u * (2 * Bx_d - Bx_u)
        der_Nz[1] = Bz_u * Bx_d + Bz_d * (2 * Bx_u - Bx_d)
        der_Nz[4] = Bz_d * By_u - Bz_u * (2 * By_d - By_u)
        der_Nz[5] = Bz_u * By_d - Bz_d * (2 * By_u - By_d)
        der_Nz[8] = Bx_u * delta_Bx + By_u * delta_By
        der_Nz[9] = -Bx_d * delta_Bx - By_d * delta_By

    if N[0]*normal[0] < 0:
        N *= -1
        der_Nx *= -1
        der_Ny *= -1
        der_Nz *= -1

    if np.linalg.norm(normal - N / np.linalg.norm(N)) > 1e-10:
        print('Normal vector of the error analysis inconsistent with the actual one.')


    # Partial derivatives of each component of normalized normal vector n
    der_nnx = np.zeros(12)
    der_nny = np.zeros(12)
    der_nnz = np.zeros(12)
    norm_N = np.linalg.norm(N)

    for i in range(len(der_Nx)):
        dot_product = np.dot(N, [der_Nx[i], der_Ny[i], der_Nz[i]])
        norm_N_cubed = norm_N ** 3
        der_nnx[i] = der_Nx[i] / norm_N - N[0] * dot_product / norm_N_cubed
        der_nny[i] = der_Ny[i] / norm_N - N[1] * dot_product / norm_N_cubed
        der_nnz[i] = der_Nz[i] / norm_N - N[2] * dot_product / norm_N_cubed

    # Calculate error of function n(Nx, Ny, Nz) = N / norm(N) using error propagation
   
    s_nx = np.sqrt(np.dot(np.array(der_nnx) ** 2, np.array(std_vec_normal) ** 2))
    s_ny = np.sqrt(np.dot(np.array(der_nny) ** 2, np.array(std_vec_normal) ** 2))
    s_nz = np.sqrt(np.dot(np.array(der_nnz) ** 2, np.array(std_vec_normal) ** 2))
    s_normal = np.array([s_nx, s_ny, s_nz])


#-------------------------------------------------------------------------------
# Error of the V_shock
#-------------------------------------------------------------------------------   
  
    #order of partial derivatives [doo/(doo Bx_d),doo/(doo Bx_u),doo/(doo Vx_d),
    #                              doo/(doo Vx_u),doo/(doo By_d),doo/(doo By_u),
    #                              ...]
    #additionally 2 partial derivatives of doo/(doo Np_d), doo/(doo Np_u)
    #altogether 14 partial derivatives in this order

    #standard deviations (in the same order as above)

    std_vec_Vsh = np.append(std_vec_normal, [Np_d_std, Np_u_std])
   

    #partial derivatives vector der_Vsh (same order as above)
    der_Vsh = np.zeros(14)
    delta_Np = Np_d-Np_u
    der_Vsh[12] = -Np_u*np.dot(delta_V,normal)/(delta_Np**2)
    der_Vsh[13] = Np_d*np.dot(delta_V,normal)/(delta_Np**2)

    #mass flux vector rho, V_sh = abs(dot(rho,normal))
    rho = (Np_d*V_vec_d-Np_u*V_vec_u)/delta_Np

    #doo rho / doo x_i dot normal (where x_i has the same order as above
    #(no density terms Np since they are already included in der_Vsh)
    der_rho_dot_n = np.zeros(12)
    der_rho_dot_n[2] = Np_d/delta_Np*nx
    der_rho_dot_n[3] = -Np_u/delta_Np*nx
    der_rho_dot_n[6] = Np_d/delta_Np*ny
    der_rho_dot_n[7] = -Np_u/delta_Np*ny
    der_rho_dot_n[10] = Np_d/delta_Np*nz
    der_rho_dot_n[11] = -Np_u/delta_Np*nz

    #first 12 terms of partial derivatives of Vsh (= abs(dot(rho,normal)) =>
    #doo Vsh/doo x_i = abs(doo rho/doo x_i dot normal+doo normal/doo xi dot rho)
    for i in range(12):
        der_Vsh[i] = (der_rho_dot_n[i] + np.dot([der_nnx[i], der_nny[i], der_nnz[i]], rho))

    #sign of the mass flux dot normal determines the final sign of all the terms
    #in der_Vsh
    if np.dot(rho, normal) < 0:
        der_Vsh *= -1

    s_Vsh = np.sqrt(np.dot(np.array(der_Vsh)**2, std_vec_Vsh**2))


#-------------------------------------------------------------------------------
# Error of the M_A and Mms 
#-------------------------------------------------------------------------------
 
    #same order of partial derivatives as above. Additionally the partial 
    #derivative of Alfvén/magnetosonic speed (altogether 15 derivatives)
   
    std_vec_MA = np.append(std_vec_Vsh, [VA_std])
    std_vec_Mms = np.append(std_vec_Vsh, [Vms_std])

    der_MA = np.zeros(15)
    der_Mms = np.zeros(15)
    #Alfven speed term to the end
    der_MA[14] = MA/VA
    der_Mms[14] = Mms/Vms

    #doo V_vec_u / doo x_i dot normal (order of x_i as above)
    der_V_vec_u_dot_n = np.zeros(14)
    der_V_vec_u_dot_n[3] = nx
    der_V_vec_u_dot_n[7] = ny
    der_V_vec_u_dot_n[11] = nz

    #sign of the V_shock-term in the Mach number calculations
    Vsh_sign = -1 #forward shock and N-shock
    if shock_type == 'reverse':
        Vsh_sign = 1 #reverse shock

    # Add the two density derivatives (=0) to the end of the normal derivatives
    der_nnx = np.append(der_nnx, [0, 0])
    der_nny = np.append(der_nny, [0, 0])
    der_nnz = np.append(der_nnz, [0, 0])

    for i in range(14):
        #print(der_nnx)
        #print([der_nnx[i], der_nny[i], der_nnz[i]])
        #print(V_vec_u)
        der_MA[i] = (der_V_vec_u_dot_n[i] + 
                    np.dot([der_nnx[i], der_nny[i], der_nnz[i]], V_vec_u) + 
                    Vsh_sign * der_Vsh[i]) / VA
        der_Mms[i] = (der_V_vec_u_dot_n[i] + 
                    np.dot([der_nnx[i], der_nny[i], der_nnz[i]], V_vec_u) + 
                    Vsh_sign * der_Vsh[i]) / Vms
    
    s_MA = np.sqrt(np.dot(np.array(der_MA)**2, np.array(std_vec_MA)**2))
    s_Mms = np.sqrt(np.dot(np.array(der_Mms)**2,np.array(std_vec_Mms)**2))


#-------------------------------------------------------------------------------
# Error of shock theta
#-------------------------------------------------------------------------------

    dot_n_B = np.dot(B_vec_u,normal)
    norm_Bu = np.linalg.norm(B_vec_u)
    if (abs(dot_n_B)/norm_Bu != 1.0):
        #d(arccos(x))/dx = -(1-x^2)^(-0.5)
        arccos_der_term = -(1-dot_n_B**2/norm_Bu**2)**(-0.5)
    else:
        arccos_der_term = 0
    
    std_vec_theta =  std_vec_normal
    
    der_B_vec_u_dot_n = np.zeros(12)
    der_B_vec_u_dot_n[1] = nx
    der_B_vec_u_dot_n[5] = ny
    der_B_vec_u_dot_n[9] = nz
    der_B_vec_u_dot_B_vec_u = np.zeros(12)
    der_B_vec_u_dot_B_vec_u[1] = Bx_u
    der_B_vec_u_dot_B_vec_u[5] = By_u
    der_B_vec_u_dot_B_vec_u[9] = Bz_u

    der_theta = np.zeros(12)
   
    #loop variable (calculated outside the loop)
    dot_Bu_n_norm_Bu = dot_n_B/norm_Bu**3
    for i in range(12):
        der_theta[i] = ((der_B_vec_u_dot_n[i] + 
                        np.dot([der_nnx[i], der_nny[i], der_nnz[i]], B_vec_u)) / norm_Bu - 
                        der_B_vec_u_dot_B_vec_u[i] * dot_Bu_n_norm_Bu) * arccos_der_term

    s_theta = np.sqrt(np.dot(np.array(der_theta)**2, np.array(std_vec_theta)**2))
    #transformation from radians to degrees
    s_theta *= 180.0/np.pi

#-------------------------------------------------------------------------------
# Return all standard deviations. Order specified in the docstring.
#-------------------------------------------------------------------------------

    return [B_u_std,Bx_u_std,By_u_std,Bz_u_std,B_d_std,Bx_d_std,By_d_std,Bz_d_std,
         s_B_rat,V_u_std,Vx_u_std,Vy_u_std,Vz_u_std,V_d_std,Vx_d_std,Vy_d_std,
         Vz_d_std,s_V_jump,Np_u_std,Np_d_std,s_Np_rat,Tp_u_std,Tp_d_std,
         s_Tp_rat,Cs_std,VA_std,Vms_std,beta_std,s_nx,s_ny,s_nz,s_theta,s_Vsh,
         s_MA,s_Mms,Dp_std]
