
import os,time
from math import radians,pi
import numpy as np
from matplotlib import pyplot as plt
import bayesik as bik





#(0) Single simulation iteration:
np.random.seed(1)
sigma        = 0.005  # marker noise
sigmas       = dict(r=0.1, theta=pi/4)  # SDs for prior normal distributions
bit,bb,bth   = 1000,500,1
# bit,bb,bth   = 10000,1000,1
q,seglengths = np.array([0, 0, radians(20)]), [0.45]
q,seglengths = np.array([0, 0, radians(20), radians(45)]), [0.45,0.35]
q,seglengths = np.array([0, 0, radians(20), radians(45), radians(90)]), [0.45,0.35,0.25]
chain        = bik.nlink.KinematicChain( seglengths )
chain.set_posture( q )
rmobs        = chain.get_rm_noisy( sigma )
qls          = chain.ik_ls(rmobs, x0=q)
qb           = chain.ik_bayes(rmobs, sigma, qls, niter=bit, nburn=bb, thin=bth, sigma_r=sigmas['r'], sigma_theta=sigmas['theta'])
bik.util.report_sim_iteration(q, [qls, qb], absolute_q=False, labels=('LS-IK','B-IK'))





# #(1) Multiple iterations:
# # user parameters:
# fast        = True     # fast simulation (few Bayesian iterations)
# nlinks      = 1
# niter       = 1000
# sigmas      = dict(r=0.1, theta=pi/4)  # SDs for prior normal distributions
# seglengths  = [0.45,0.35,0.25][:nlinks]
# bit,bb,bth  = (1e4,1e3,2) if fast else (1e5,1e4,5)  # MCMC parameters
# # derived parameters:
# np.random.seed(nlinks)
# nq          = 2 + nlinks
# dirname     = 'sim-nlink-fast' if fast else 'sim-nlink'
# fnameNPZ    = os.path.join( bik.dirREPO, 'Data', dirname, f'links{nlinks}.npz')
# SIGMA       = np.random.uniform(0.0002, 0.002, niter)
# THETA       = np.random.uniform(-pi/2, pi/2, size=(niter,nlinks))
# # run simulation:
# Q,QLS,QB    = [np.empty( (niter, nq) )   for i in range(3)]
# TLS,TB      = [np.empty(niter)   for i in range(2)]
# for i in range(niter):
# 	print( f'Iteration {i+1} of {niter}...' )
# 	q_true      = np.hstack( [[0, 0], THETA[i]] )
# 	chain       = bik.nlink.KinematicChain( seglengths )
# 	chain.set_posture( q_true )
# 	rmobs       = chain.get_rm_noisy( SIGMA[i] )
# 	# least squares IK:
# 	t0          = time.time()
# 	qls         = chain.ik_ls(rmobs, x0=q_true)
# 	tls         = time.time() - t0
# 	# Bayesian IK:
# 	t0          = time.time()
# 	qb          = chain.ik_bayes(rmobs, SIGMA[i], qls, niter=bit, nburn=bb, thin=bth, sigma_r=sigmas['r'], sigma_theta=sigmas['theta'])
# 	tb          = time.time() - t0
# 	### save results:
# 	[Q[i], QLS[i], QB[i], TLS[i], TB[i]] = [q_true, qls, qb, tls, tb]
# 	np.savez(fnameNPZ, Q=Q, QLS=QLS, QB=QB, TLS=TLS, TB=TB, SIGMA=SIGMA, THETA=THETA)
# 	### report:
# 	bik.util.report_sim_iteration(q_true, [qls, qb], absolute_q=False, labels=('LS-IK','B-IK'))
# 	bik.util.report_sim_summary(Q, [QLS, QB], i, [TLS, TB], labels=('LS-IK','B-IK'))
# 	print('\n\n\n')















