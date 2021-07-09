
import os,time
from math import cos,sin,radians,pi
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
import pymc
import bayesik as bik




def add_noise(r, sigma=0.002):
	return r + sigma * np.random.randn( *r.shape )


def get_true_marker_positions(q=(0,0,0), length=0.45, mplength=0.1, mpwidth=0.04):
	# set original (noiseless) marker positions
	ll,ww = 0.5*mplength, 0.5*mpwidth
	r     = [0.7*length,0] + np.array([  [-ll,-ww], [ll,-ww], [ll,ww], [-ll,ww]  ])
	# transform marker positions:
	rt    = fk(r, q)
	return rt


def fk(r, q):
	dx,dy,dphi  = q
	s,c         = sin(dphi), cos(dphi)
	# T0          = np.array( [[1,0,-dx],[0,1,-dy],[0,0,1]] )  # translate to rotation center
	T1          = np.array( [[c,-s,0], [s,c,0],  [0,0,1]] )  # rotate
	T2          = np.array( [[1,0,dx], [0,1,dy], [0,0,1]] )  # translate back to original location
	# T           = T2 @ T1 @ T0
	# T           = T0 @ T1 @ T2
	T           = T2 @ T1
	rh          = np.vstack([r.T, np.ones(r.shape[0])]).T
	rnew        = (T @ rh.T).T
	return rnew[:,:2]


def ik_ls(r0, r1, q0=[0,0,0]):
	def objfn(x, r0, r1):
		r0t = fk(r0, x)
		return np.linalg.norm(r0t-r1, axis=1).sum()
	q     = optimize.minimize(objfn, q0, args=(r0,r1)).x
	return q


def ik_bayes_model(robs0, robs1, priors):
	# assemble prior parameters:
	marker_tau0 = 1. / ( priors['marker']['upper']**2 )  # lower limit for marker precision prior (i.e., upper limit for SD)
	marker_tau1 = 1. / ( priors['marker']['lower']**2 )  # upper limit for marker precision prior (i.e., lower limit for SD)
	trans_mu    = priors['translation']['mu']
	trans_tau   = 1. / ( priors['translation']['sigma']**2 )
	rot_mu      = priors['rotation']['mu']
	rot_tau     = 1. / ( priors['rotation']['sigma']**2 )
	# construct priors:
	tau         = pymc.Uniform("tau", marker_tau0, marker_tau1)
	r           = pymc.Normal("r",    trans_mu, trans_tau, size=2, value=trans_mu)
	phi         = pymc.Normal("phi",  rot_mu, rot_tau, value=rot_mu)
	# create deterministic model
	@pymc.deterministic
	def observations_model(r=r, phi=phi):
		q      = (r[0], r[1], phi)  # generalized transformation parameters
		rt     = fk(robs0, q)       # forward kinematics for first frame's (noisy) marker positions
		return rt
	# assemble Bayesian model:
	bmodel     = pymc.Normal("qobs", observations_model, tau, value=robs1, observed=True)
	return bmodel, tau, r, phi


def ik_bayes(robs0, robs1, priors, sample_options=None, progress_bar=True):
	if sample_options is None:
		sample_options = dict(niter=1000, burn=500, thin=1)
	model   = ik_bayes_model(robs0, robs1, priors)
	mcmc    = pymc.MCMC( model )
	niter,burn,thin = sample_options['niter'], sample_options['burn'], sample_options['thin']
	mcmc.sample(niter*thin+burn, burn, thin, progress_bar=progress_bar)
	if progress_bar:
		print() # the pymc progress bar does not print a newline character when complete
	# assemble results:
	R       = mcmc.trace('r')[:]
	PHI     = mcmc.trace('phi')[:]
	Q       = np.vstack( [R.T, PHI] ).T
	q       = Q.mean(axis=0)   # mle for transformation parameters
	return q





#(0) Single simulation iteration:
# user parameters:
seed               = 1       # random number generator seed (for noisy marker positions only)
length             = 0.450   # segment length
sigma              = 0.002   # marker noise (standard deviation)
q0                 = np.array([0, 0, radians(30)])   # true initial pose
q1                 = np.array([0.05, 0, radians(60)])   # true final pose
# derived parameters:
np.random.seed(seed)
qtrue              = q1 - q0                         # true transformation parameters
r0                 = get_true_marker_positions(q0, length=length)   # true initial marker positions
r1                 = get_true_marker_positions(q1, length=length)   # true final marker positions
r0n                = add_noise(r0)   # noisy initial marker positions
r1n                = add_noise(r1)   # noisy final marker positions
# least-squares solutions:
qh                 = bik.ik.halvorsen(r0n, r1n)
qs                 = bik.ik.soderkvist(r0n, r1n)
qls                = ik_ls(r0n, r1n, q0=qs)
# Bayesian solution:
sample_options     = dict(niter=1000, burn=500, thin=1)     # MCMC sampling options
prior_marker_error = dict(lower=0.1*sigma, upper=10*sigma)  # Uniform marker error prior
prior_translation  = dict(mu=qs[:2], sigma=0.001)           # Normal translation prior
prior_rotation     = dict(mu=qs[2],  sigma=pi/40)           # Normal rotation prior
priors             = dict(marker=prior_marker_error, translation=prior_translation, rotation=prior_rotation)
qb                 = ik_bayes(r0n, r1n, priors, sample_options=sample_options)
# report results:
bik.util.report_sim_iteration(qtrue, [qh,qs,qls,qb], absolute_q=True, labels=['Halvorsen', 'Soderkvsit', 'LS', 'Bayes'])
# plot:
plt.close('all')
plt.figure()
ax = plt.axes()
ax.plot(r0[:,0], r0[:,1], 'ko', ms=10)
ax.plot(r1[:,0], r1[:,1], 'ro', ms=10)
ax.plot(r0n[:,0], r0n[:,1], 'ko', ms=5, mfc='w')
ax.plot(r1n[:,0], r1n[:,1], 'ro', ms=5, mfc='w')
### plot mean
m0,m1 = r0.mean(axis=0), r1.mean(axis=0)
ax.plot([q0[0], m0[0]], [q0[1], m0[1]], 'k-', lw=3)
ax.plot([q1[0], m1[0]], [q1[1], m1[1]], 'r-', lw=3)
ax.axis('equal')
plt.show()





# #(1) Multiple iterations:
# # user parameters:
# niter              = 1000    # number of iterations (marker position datasets) to test
# seed               = 1       # random number generator seed (for noisy marker positions only)
# length             = 0.450   # segment length
# sigma              = 0.002   # marker noise (standard deviation)
# q0                 = np.array([0, 0, radians(30)])   # true initial pose
# q1                 = np.array([0.1, 0, radians(60)])   # true final pose
# fnameNPZ           = os.path.join( bik.dirREPO, 'Data', f'twoframe.npz')
# # derived parameters:
# qtrue              = q1 - q0                         # true transformation parameters
# r0                 = get_true_marker_positions(q0, length=length)   # true initial marker positions
# r1                 = get_true_marker_positions(q1, length=length)   # true final marker positions
#
# # run simulation:
# np.random.seed(seed)
# Q,QH,QS,QLS,QB     = [np.zeros( (niter, 3) )   for i in range(5)]
# TH,TS,TLS,TB       = [np.zeros(niter)   for i in range(4)]
# for i in range(niter):
# 	print( f'Iteration {i+1} of {niter}...' )
# 	r0n            = add_noise(r0)   # noisy initial marker positions
# 	r1n            = add_noise(r1)   # noisy final marker positions
# 	# least squares IK (Halvorsen):
# 	t0          = time.time()
# 	qh          = bik.ik.halvorsen(r0n, r1n)
# 	th          = time.time() - t0
# 	# least squares IK (Soderkvist):
# 	t0          = time.time()
# 	qs          = bik.ik.soderkvist(r0n, r1n)
# 	ts          = time.time() - t0
# 	# least squares (scipy):
# 	t0          = time.time()
# 	qls         = ik_ls(r0n, r1n, qtrue)
# 	tls         = time.time() - t0
# 	# Bayesian IK:
# 	t0          = time.time()
# 	# Bayesian priors:
# 	# bit,bb,bth  = (10000,1000,1) if fast else (1e5,1e4,5)
# 	# sample_options     = dict(niter=1000, burn=500, thin=1)     # MCMC sampling options (fast, to check for errors)
# 	# sample_options     = dict(niter=10000, burn=1000, thin=1)     # MCMC sampling options (fast, to check preliminary results)
# 	sample_options     = dict(niter=1e5, burn=1e4, thin=5)      # MCMC sampling options
# 	prior_marker_error = dict(lower=0.1*sigma, upper=10*sigma)  # Uniform marker error prior
# 	prior_translation  = dict(mu=qs[:2], sigma=0.001)           # Normal translation prior
# 	prior_rotation     = dict(mu=qs[2],  sigma=pi/40)           # Normal rotation prior
# 	priors             = dict(marker=prior_marker_error, translation=prior_translation, rotation=prior_rotation)
# 	qb          = ik_bayes(r0n, r1n, priors, sample_options=sample_options)
# 	tb          = time.time() - t0
# 	### save results:
# 	[Q[i], QH[i], QS[i], QLS[i], QB[i], TH[i], TS[i], TLS[i], TB[i]] = [qtrue, qh, qs, qls, qb, th, ts, tls, tb]
# 	np.savez(fnameNPZ, Q=Q, QH=QH, QS=QS, QLS=QLS, QB=QB, TH=TH, TS=TS, TLS=TLS, TB=TB)
# 	### report:
# 	bik.util.report_sim_iteration(qtrue, [qh,qs,qls,qb], labels=('Halvorsen','Soderkvist','LS','Bayes'), absolute_q=True)
# 	bik.util.report_sim_summary(Q, [QH, QS, QLS, QB], i, [TH, TS, TLS, TB], labels=('Halvorsen','Soderkvist','LS','Bayes'))
# 	print('\n\n\n')















