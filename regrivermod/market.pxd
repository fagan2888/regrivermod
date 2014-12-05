from econlearn.tilecode cimport Tilecode
from regrivermod.storage cimport Storage
from regrivermod.users cimport Users
from regrivermod.environment cimport Environment
from regrivermod.utility cimport Utility

cdef class Market:

    cdef public Tilecode market_d
    cdef public Tilecode perf_market
    cdef public Tilecode d_low
    cdef public Tilecode d_high
    cdef public Tilecode d_env

    cdef int nat

    cdef public int N
    cdef public int M

    cdef public double[:] d_beta, d_cons, p, a

    cdef public double Pmax, t_cost, users_Pmax, EX, ePmax
    
    cdef public double min_q, d_b_e, d_c_e, p_e, a_e

    cdef public double[:] threezeros

    cdef void open_market(self, Users users, Environment env, int M)

    cdef double solve_price(self, double Q, double I, double Bhat, int init, int plan)
