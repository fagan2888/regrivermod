cdef class Storage:

    ##################      Parameters      ##################
        
    cdef public double K
    cdef public double delta0
    cdef public double alpha
    cdef public double rho 
    cdef public double theta_I
    cdef public double k_I
    cdef public double delta1a
    cdef public double delta1b
    
    ##################      Variables      ##################
   
    cdef public double I 
    cdef public double I_bar
    cdef public double S                                                         
    cdef public double Spill                                                          
    cdef public double Loss 

    cdef double[:] EPS      # eps_I shock series 

    #########################################################

    cdef double pi
    cdef public double Imax

    cdef public double[:, :] I_grid
    cdef public double[:, :] I_pr
    
    cdef double update(self, double W, int t)

    cdef double release(self, double W)

