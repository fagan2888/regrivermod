from regrivermod.storage cimport Storage
from regrivermod.environment cimport Environment
from regrivermod.users cimport Users
from econlearn.tilecode cimport Tilecode

cdef class Utility:

    cdef int N, N_low, N_high
    cdef int[:] I_low, I_high
    cdef int I_env
    cdef int fail

    cdef double K
    cdef public int sr
    cdef public int ls
    cdef public int HL
    cdef public int envoff

    cdef public double A_bar, A, Lambda_high, max_E
    cdef public double[:] s, l, x, a, w, maxx, minx, c_F, c_K, acc_max, J
    
    cdef public double loss_account     # Utilities loss account for fixed delivery losses 
    cdef public double fixed_loss        
    cdef public double fixed_loss_co
    cdef public double loss
    cdef public int delivered
    cdef public double target
    cdef public double[:] ss
    cdef public int it

    cdef double delta1a
    cdef double[:] delta_a
    cdef double delta_Ea
    cdef double delta1b, delta_R, max_R

    cdef int M, ch7

    cdef double c_pi

    cdef double[:] temp
    cdef double[:] temp2
    cdef double[:] temp3
    cdef double[:] temp4
    cdef double[:] temp5
    cdef public double oldS

    cdef public Tilecode policy0
    cdef public Tilecode policy1
    cdef public double[:] two_zeros, three_zeros
    cdef public double[:] N_zeros
    cdef public int explore
    cdef public double d

    cdef double release(self, double[:] w, double S)
        
    cdef void update(self, double S, double I, double L, double Spill, double[:] w, double A)
    
    cdef void update_storage_accounts(self, double S, double I, double L, double Spill, double[:] w)

    cdef double deliver_ch7(self, Storage storage,  int M, double We)
    
    cdef double[:] allocate(self, double A, double[:] a) nogil

    cdef double extract(self, double qe)

    cdef double withdraw_ch7(self, double S, double I, double Bhat, int M, int envoff)

    cdef double record_trades(self, Users users, Environment env, Storage storage)

    cdef double make_allocations(self, double[:] users_w, double env_w)
