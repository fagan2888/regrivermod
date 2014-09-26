from econlearn.tilecode cimport Tilecode

cdef class Storage:

    cdef public Tilecode policy
    cdef public double[:] X

    ##################      Parameters      ##################

    cdef public double K, delta0, alpha, rho, theta_I, k_I, delta1a, delta1b
    
    cdef public double omega_mu, omega_sig, delta_a, delta_b, delta_Ea, delta_Eb, delta_R 
    cdef public double[:] omega_ab, F_bar
    
    ##################      Variables      ##################
   
    cdef public double I, I_tilde, I_bar, S, Spill, Loss, loss_12, C
    cdef public double F1, F2, F3
    cdef public double F1_tilde, F2_tilde, F3_tilde, min_F2, max_E
   
    cdef double[:] EPS      # eps_I shock series 
    cdef double[:] OMEGA    # omega shock series 

    #########################################################

    cdef double pi
    cdef public double Imax

    cdef public double[:, :] I_grid
    cdef public double[:, :] I_pr
    
    cdef double update(self, double W, int t)

    cdef double release(self, double W)
    
    cdef void river_flow(self, double W, double E, int M)

    cdef double update_ch7(self, double W, double E, int M, int t)

    cdef double storage_transition(self, double W)

    cdef void natural_flows(self, double W, double max_E, int M)
