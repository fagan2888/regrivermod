from econlearn.tilecode cimport Tilecode
from regrivermod.storage cimport Storage
from regrivermod.users cimport Users
from regrivermod.environment cimport Environment

cdef class Market:

    cdef public Tilecode market_d
    cdef public Tilecode perf_market

    cdef public int N

    cdef public double[:] d_beta, d_cons, p, a

    cdef public double Pmax, tcost
    
    cdef public double min_q, d_b_e, d_c_e, p_e, a_e

    cdef void open_market(self, Users users, Environment env)

    cdef double solve_price(self, double Q, double P_guess)

    cdef void estimate_market_demand(self, Storage storage, Users users, Environment env, para)
