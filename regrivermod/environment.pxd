from econlearn.tilecode cimport Tilecode
from regrivermod.utility cimport Utility

cdef class Environment:

    cdef public Tilecode policy0
    cdef public Tilecode value0
    cdef public Tilecode policy1
    cdef public Tilecode value1

    cdef public double w, a, q, u, p, d, pi
    cdef public double d_c, d_b    
    cdef public int explore
    cdef public double min_q    
    cdef public double b1, b3, b_value, Bhat, Bhat_alpha
    cdef public double Lambda_I, Lambda_K
    cdef public double e_sig
    cdef public double budget 
    cdef public double budget_in
    cdef public double budget_out 
    cdef public double budget_tc
    cdef public double delta_R, delta_Eb
    cdef public double[:] delta_a 
    cdef public double[:] e

    cdef public double DELTA0, DELTA1
    cdef public double t_cost
    cdef public int t

    cdef public double k_I, theta_I

    cdef public int turn_off

    cdef double[:] state_zero
    cdef public double Pmax

    cdef double consume(self, double P, int M, int planner)

    cdef void allocate(self, double a, double Z, double max_R, double F1_tilde, double F3_tilde, double Pr, int M)

    cdef double payoff(self, double F1, double F3, double F1_tilde, double F3_tilde, double P)

    cdef double withdraw(self, double S, double s, double I, int M)

    cdef double update(self)
