from econlearn.tilecode cimport Tilecode

cdef class Environment:

    cdef public Tilecode policy
    cdef public double w, a, q, u, p, d, pi
    cdef public double d_c, d_b    
    cdef public int explore
    cdef public double min_F2, min_q    
    cdef public double b2, b3
    cdef public double Lambda_I, Lambda_K
        
    cdef public double delta_R, delta_Eb, delta_a 
    
    cdef public double DELTA 

    cdef double[:] state_zero
    
    cdef double consume(self, double P, double t_cost, double p, double a)
    
    cdef void allocate(self, double a, double min_F2, double F3_tilde)

    cdef double payoff(self, double F1, double F3, double F1_tilde, double F3_tilde)

    cdef double withdraw(self, double S, double s, double I, int M)