=========================================
Water property rights in regulated rivers
=========================================

All the code for my thesis: Water property rights in regulated rivers.  For more info see my `ANU site  <https://crawford.anu.edu.au/people/phd/neal-hughes/>`_.

Overview
========

Figure x summarises the structure of the code: 

.. image:: /_images/tree.jpg
    :scale: 60 %

Most of the work is performed by two cython modules ``econlearn`` (a machine learning toolkit) and ``regrivermod`` (a simulation model of a regulated river). All of the parameter assumptions are contained in ``para.py``. For a given set of parameters ``model.py`` combines ``econlearn`` and ``regrivermod`` to solve the various versions of the model. The scripts ``chapter3.py, ..., chapter8.py`` implement sensitivity analysis. Finally the ``results`` module is used to generate all of the figures and tables for my thesis.

regrivermod
-----------

``regrivermod`` provides a simulation model of a simplified river system. The main purpose of ``regrivermod`` is to perform monte carlo simulation and record data (e.g., state transitions and payoffs). This data is then used to solve water storage problems by batch reinforcement learning (using the separate ``econlearn`` module).

``regrivermod`` contains the following classes:

    - Storage:
      contains all of the hydrological detail of the model: inflows, storage, river flows and losses etc.

    - Utility:
      plays the role of a water utility, implements water accounting / property rights systems

    - Users:
      contains all detail related to consumptive water users: storage policy functions, demand functions etc. and solves the spot market 

    - Simulation:
      combines all of the above classes to perform simulations and record data

    - Sdp:
      a class for Stochastic Dynamic Programming, used to solve the planner's storage problem 

Here is a usage example, simulating the planner's (SDP) solution (of the model from chapter 3)::

    from regrivermod import *
    import Para                 # Parameter assumptions
    
    # Build parameters
    para = Para()
    para.central_case() 
    para.solve_para()
    para.set_property_rights(scenario = 'CS')   # Capacity sharing water rights

    # Build objects
    storage = Storage(para)
    users = Users(para)
    sim = Simulation(para)
    utility = Utility(users, storage, para)

    # Solve planner's problem by SDP
    sdp = SDP(para.SDP_GRID,  users, storage, para)    
    sdp.policy_iteration(self.para.SDP_TOL, self.para.SDP_ITER) 

    # Simulate the planner's solution
    sim.simulate(users, storage, utility, 100, 4, planner=True, policy=True, polf=sdp.W_f)


Econlearn
---------

``Econlearn`` is a machine learning toolkit for economist's. Its main purposes is to implement the batch reinforcement learning algorithm: fitted $Q$-$V$ iteration, using tile coding for function approximation. ``econlearn`` contains a range of fast machine learning algorithms suitable for low (i.e., <10) dimensions.

``econlearn`` contains the following files:

    - tilecode.pyx:
      implements tile coding function approximation (supervised learning)
    - Qlearn.py:
      implements fitted Q-V iteration, with either tilecoding or random forest approximation (via scikit-learn).
    - samplegrid.pyx:
      selects approximately equidistantly spaced grids (i.e., sample grids) 

I plan to add some other machine learning algorithms to ``econlearn`` including: fast approximate nearest neighbors (using tilecoding), fast local quadratic regression (using tilecoding), RBF network regression and density estimation (using RBFs and tilecoding).

Here is a usage example, solving the same planner's problem by fitted Q-V iteration (see chapter 8)::

    from econlearn import Qlearn

    T = [6, 6, 3]               # Number of tiles per dimension
    L = 25                      # Number of tile layers
    
    # Feasibility constraints
    Alow = lambda X: 0                  # W > 0
    Ahigh = lambda X: X[0]              # W < S
    
    sim.simulate(users, storage, utility, 50000, 4, planner=True, policy=False,  
    planner_explore=True)
        
    qv = Qlearn.QVtile(2, T, L, 1, 1, para.s_points1, para.s_radius1, para, asgd=True)
        
    qv.iterate(sim.XA_t, sim.X_t1, sim.series['SW'], Alow, Ahigh, ITER=50) 

Installation
============

This code requires installation of python with packages: cython, numpy, scipy, scikit-learn and pandas. If you don't have python yet a good option is to install `Anaconda <http://docs.continuum.io/anaconda/>`_.  

Next download or clone this repository. Then you need to compile the cython modules. On linux you can run ``build.sh`` script from the terminal. Just navigate to the install directory then type:

    bash build.sh

Otherwise run ``setup.py`` for both ``econlearn`` and ``regrivermod``:

    cd econlearn
    python setup.py build_ext --inplace
    cd ..
    cd regrivermod
    python setup.py build_ext --inplace
    cd..

