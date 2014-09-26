from econlearn.tilecode cimport Tilecode
from regrivermod.storage cimport Storage
from regrivermod.users cimport Users
from regrivermod.environment cimport Environment
from regrivermod.utility cimport Utility

cdef class Market:

    cdef public Tilecode market_d
    cdef public Tilecode perf_market

    cdef public int N

    cdef public double[:] d_beta, d_cons, p, a

    cdef public double Pmax, t_cost
    
    cdef public double min_q, d_b_e, d_c_e, p_e, a_e

    cdef public double[:] twozeros

    cdef void open_market(self, Users users, Environment env)

    cdef double solve_price(self, double Q, double I, int init)

    cpdef estimate_market_demand(self, Storage storage, Users users, Environment env, Utility utility, para)
