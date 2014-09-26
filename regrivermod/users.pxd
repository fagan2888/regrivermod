cimport cython
from econlearn.tilecode cimport Tilecode, Function_Group

cdef class Users:

    cdef public int N
    cdef public int N_low 
    cdef public int N_high
    cdef public int[:] I_low 
    cdef public int[:] I_high

    cdef public double low_c  
    cdef public double c_F_low
    cdef public double c_K_low
    cdef public double c_F_high
    cdef public double c_K_high 
    cdef public double[:] c_F
    cdef public double[:] c_K
    
    cdef public double delta                        
    cdef public double K
    cdef public double[:,:] theta 
    cdef public double rho_eps
    cdef public double sig_eta
    cdef public double[:] risk
    cdef public int utility

    cdef public double[:] L 
    
    cdef public double[:] d_beta                                            
    cdef public double[:] d_cons 
    cdef public double Pmax 
    cdef public double[:] MV 
    cdef public double t_cost 

    cdef public bint exploring
    cdef public int N_e
    cdef public double d
    cdef public int[:] I_e_l
    cdef public int[:] I_e_h
    cdef public int[:] I_e 
    
    cdef public int share_explore
    cdef public int[:] share_e_l
    cdef public int[:] share_e_h
    cdef public double low_gain
    cdef public double high_gain
    cdef public double share_adj

    cdef public double[:] w
    cdef public double[:] w_scaled
    cdef public double W_low
    cdef public double W_high
    cdef public double S_low
    cdef public double S_high
    cdef public double X_low
    cdef public double X_high
    cdef public double trade_low
    cdef public double trade_high
    cdef public double[:] trade
    cdef public double tradeVOL
    cdef public double[:] a
    cdef public double[:] q
    cdef public double[:] profit
    cdef public double U_low
    cdef public double U_high
    cdef public double[:] e
   
    cdef public Function_Group policy
    cdef public Tilecode w_f_low
    cdef public Tilecode W_f
    cdef public Tilecode w_f_high
    cdef public Tilecode v_f_high
    cdef public Tilecode v_f_low
    cdef public Tilecode SW_f
    cdef public Tilecode market_d
    cdef public Tilecode perf_market
    cdef public Tilecode w_f
    
    cdef public int test_idx
    cdef public int testing
    cdef public int test_explore

    cdef double c_pi
    cdef double[:, :] state_zero
    cdef double[:] two_zeros
    cdef double[:] state_single_zero 
    cdef double[:,:] state_planner_zero
    cdef double[:] N_zeros
    cdef int[:] N_ints

    cdef public int init
    cdef double delta1a

    cdef public int lowexplore
    cdef public int highexplore

    cdef double[:] withdraw(self, double S, double[:] s, double I)
    
    cdef mv(self, double I)

    cdef void explore(self, double[:] s)

    cdef double consume(self, double P, double I, int planner)

    cdef void update(self)

    cdef demand_para(self, double I = ?)

    cdef double clear_market(self, double I, Tilecode market_d, int planner)

    cdef double solve_price(self, double Q, double P_guess, double Pmax, double t_cost)

    cdef void user_stats(self, double[:] s, double[:] x)
        
    cdef void allocate(self, double[:] a, double I)
