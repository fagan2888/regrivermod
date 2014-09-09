from econlearn.tilecode cimport Tilecode

cdef class Storage:

    cdef public Tilecode policy
    cdef public double[:] X

    ##################      Parameters      ##################

    cdef public double K
    cdef public double delta0
    cdef public double alpha
    cdef public double rho 
    cdef public double theta_I
    cdef public double k_I
    cdef public double delta1a
    cdef public double delta1b
    
    cdef public double omega_mu
    cdef public double omega_sig
    cdef public double[:] omega_ab
    cdef public double delta_a          
    cdef public double delta_b
    cdef public double delta_Ea
    cdef public double delta_Eb
    cdef public double delta_R  
    cdef public double[:] F_bar
    
    ##################      Variables      ##################
   
    cdef public double I 
    cdef public double I_bar
    cdef public double S                                                         
    cdef public double Spill                                                          
    cdef public double Loss
    cdef public double loss_12
    cdef public double C
    cdef public double F1
    cdef public double F2
    cdef public double F3
    cdef public double F1_tilde
    cdef public double F2_tilde
    cdef public double F3_tilde
   
    cdef double[:] EPS      # eps_I shock series 
    cdef double[:] OMEGA    # omega shock series 

    #########################################################

    cdef double pi
    cdef public double Imax

    cdef public double[:, :] I_grid
    cdef public double[:, :] I_pr
    
    cdef double update(self, double W, int t)

    cdef double release(self, double W)
    
    cdef double release_ch7(self, double W, int M)

    cdef double extract_ch7(self, double E)

    cdef void river_flow(self, double W, double E, int M)

    cdef double update_ch7(self, double W, double E, int M, int t)

    cdef double storage_transition(self, double W)
