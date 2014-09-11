cdef class Utility:

    cdef int N
    cdef int N_low
    cdef int N_high
    cdef int[:] I_low
    cdef int[:] I_high

    cdef double K
    cdef public int sr
    cdef public int ls
    cdef public int HL
    cdef public double A_bar
    cdef public double Lambda_high

    cdef public double[:] s
    cdef public double[:] l
    cdef public double[:] x
    cdef public double[:] a
    cdef public double[:] maxx
    cdef public double[:] minx
    cdef public double[:] c_F
    cdef public double[:] c_K
    cdef public double[:] acc_max
    cdef public double[:] J
    
    cdef public double loss_account     # Utilities loss account for fixed delivery losses 
    cdef public double fixed_loss        
    cdef public double fixed_loss_co
    cdef public double loss
    cdef public int delivered

    cdef double delta1a
    cdef double delta1b

    cdef int M

    cdef double[:] temp
    cdef double[:] temp2
    cdef double[:] temp3
    cdef double[:] temp4

    cdef double release(self, double[:] w, double S)
        
    cdef void update(self, double S, double I, double L, double Spill, double[:] w, double A)
    
    cdef void update_storage_accounts(self, double S, double I, double L, double Spill, double[:] w)

    cdef double deliver_ch7(self, double[:] w, double S, int M)

