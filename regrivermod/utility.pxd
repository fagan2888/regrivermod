from regrivermod.storage cimport Storage

cdef class Utility:

    cdef int N, N_low, N_high
    cdef int[:] I_low, I_high
    cdef int I_env

    cdef double K
    cdef public int sr
    cdef public int ls
    cdef public int HL
    
    cdef public double A_bar, A, Lambda_high, max_E
    cdef public double[:] s, l, x, a, w, maxx, minx, c_F, c_K, acc_max, J
    
    cdef public double loss_account     # Utilities loss account for fixed delivery losses 
    cdef public double fixed_loss        
    cdef public double fixed_loss_co
    cdef public double loss
    cdef public int delivered

    cdef double delta1a
    cdef double delta_Ea
    cdef double delta1b

    cdef int M, ch7

    cdef double[:] temp
    cdef double[:] temp2
    cdef double[:] temp3
    cdef double[:] temp4

    cdef double release(self, double[:] w, double S)
        
    cdef void update(self, double S, double I, double L, double Spill, double[:] w, double A)
    
    cdef void update_storage_accounts(self, double S, double I, double L, double Spill, double[:] w)

    cdef double deliver_ch7(self, double[:] users_w, double env_w, Storage storage,  int M)
    
    cdef double[:] allocate(self, double A, double[:] a) nogil

    cdef double extract(self, double qe)

