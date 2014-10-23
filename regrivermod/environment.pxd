from econlearn.tilecode cimport Tilecode

cdef class Environment:

    cdef public Tilecode policy0
    cdef public Tilecode policy1
    cdef public Tilecode value0
    cdef public Tilecode value1

    cdef public double w, a, q, u, p, d, pi
    cdef public double d_c, d_b    
    cdef public int explore
    cdef public double min_F2, min_q    
    cdef public double b1, b3
    cdef public double Lambda_I, Lambda_K
        
    cdef public double delta_R, delta_Eb, delta_a 
    
    cdef public double DELTA 
    cdef public double t_cost

    cdef public int turn_off

    cdef double[:] state_zero
    cdef public double Pmax

    cdef double consume(self, double P, int planner)

    cdef allocate(self, double a, double min_F2, double F3_tilde)

    cdef double payoff(self, double F1, double F3, double F1_tilde, double F3_tilde, double P)

    cdef double withdraw(self, double S, double s, double I, int M)
