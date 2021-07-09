
from copy import deepcopy
from math import pi
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from . seg import Segment
import pymc



class KinematicChain(object):
	def __init__(self, seglengths=[0.45,0.35,0.25]):
		self.segs  = [Segment(length=x)  for x in seglengths]
		self.q     = np.array( [0, 0] + [0]*self.nlinks )

	@property
	def nlinks(self):
		return len( self.segs )
	@property
	def nq(self):
		return self.links + 2
	
	@property
	def rm(self):
		return np.vstack( [seg.rm for seg in self.segs] )

	def copy(self):
		return deepcopy(self)
	
	def get_rm_noisy(self, sigma=0.01):
		return self.rm + sigma * np.random.randn( *self.rm.shape )

	def ik_bayes_model(self, robs, sigma, q0, sigma_r=0.1, sigma_theta=pi/4):
		# assemble kin parameters:
		r_ls      = q0[:2]
		theta_ls  = q0[2:]
		# assemble precisions:
		tau_true  = 1. / (sigma**2)
		tau_r     = 1. / (sigma_r**2)
		tau_theta = 1. / (sigma_theta**2)
		tau       = pymc.Uniform('tau', 0.1*tau_true, 10*tau_true)
		r         = pymc.Normal("r", r_ls, tau_r, size=2, value=r_ls)
		theta     = pymc.Normal("theta", theta_ls, tau_theta, size=self.nlinks, value=theta_ls)

		@pymc.deterministic
		def observations_model(r=r, theta=theta):
			q    = np.hstack([r, theta])
			kc   = self.copy()
			kc.set_posture(q)
			return kc.rm
		fwmodel  = pymc.Normal("qobs", observations_model, tau, value=robs, observed=True)
		return fwmodel, tau, r, theta


	def ik_bayes(self, robs, sigma, q0, niter=10000, nburn=5000, thin=1, progress_bar=True, sigma_r=0.1, sigma_theta=pi/4):
		model   = self.ik_bayes_model(robs, sigma, q0, sigma_r=sigma_r, sigma_theta=sigma_theta)
		mcmc    = pymc.MCMC( model )
		mcmc.sample(niter*thin+nburn, nburn, thin, progress_bar=progress_bar)
		R       = mcmc.trace('r')[:]
		THETA   = mcmc.trace('theta')[:]
		Q       = np.hstack( [R, THETA] )
		qstar   = Q.mean(axis=0)
		return qstar


	def ik_ls_objfun(self, q, robs):
		kc = self.copy()
		kc.set_posture(q)
		d  = np.linalg.norm( robs-kc.rm , axis=1)
		return (d**2).sum()

	def ik_ls(self, robs, x0=None, tol=1e-6, eps=1.5e-8, gtol=1e-6, disp=False):
		x0      = np.zeros(self.nq) if (x0 is None) else x0
		options = {'disp': disp, 'gtol': gtol, 'eps': eps, 'return_all': False, 'maxiter': None, 'norm': np.inf}
		results = optimize.minimize(self.ik_ls_objfun, x0, tol=tol, options=options, args=(robs,))
		return results.x

	def plot(self, ax=None, rmobs=None):
		for i,seg in enumerate(self.segs):
			if rmobs is None:
				seg.plot(ax=ax)
			else:
				seg.plot(ax=ax, rmobs=rmobs[4*i:4*(i+1)])

	def set_posture(self, q):
		self.q     = q
		r0         = q[:2]
		for i,seg in enumerate(self.segs):
			qq     = np.array( [r0[0], r0[1], q[2:3+i].sum()] )
			seg.set_posture(qq)
			r0     = seg.r1
