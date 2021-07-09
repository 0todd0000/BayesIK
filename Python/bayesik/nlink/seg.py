
from math import cos,sin,radians,pi
import numpy as np
from matplotlib import pyplot as plt



class Segment(object):
	def __init__(self, length=1, mplength=0.1, mpwidth=0.04):
		self.length  = length
		self.q       = np.array([0, 0, 0])
		ll,ww,n      = 0.5*mplength, 0.5*mpwidth, 4
		self._rm0    = [0.7*length,0] + np.array([  [-ll,-ww], [ll,-ww], [ll,ww], [-ll,ww]  ])
		
	@property
	def r0(self):
		return self.q[:2]
	@property
	def r1(self):
		l = self.length
		a = self.theta
		return self._transform( np.array([[self.length, 0]]) )[0]
	@property
	def rm(self):
		return self._transform( self._rm0 )
	@property
	def theta(self):
		return self.q[2]
	
	@staticmethod
	def _get_transform_matrix(q):
		x,y,a    = q
		c,s      = cos(a), sin(a)
		T        = np.array( [ [c, -s, x], [s, c, y], [0, 0, 1] ] )
		return T
	
	def _transform(self, r, T=None):
		T        = self._get_transform_matrix(self.q) if (T is None) else T
		r        = np.vstack( [r.T, np.ones(r.shape[0])] )
		rt       = (T @ r).T
		return rt[:,:2]
	
	def get_rm_noisy(self, sigma=0.01):
		return self.rm + sigma * np.random.randn( *self.rm.shape )
	
	def plot(self, ax=None, rmobs=None, colors=None):
		ax         = plt.gca() if (ax is None) else ax
		colors     = dict(link='0.5',joint='k',markers='0.8',observations='r') if (colors is None) else colors
		x0,y0      = self.r0
		x1,y1      = self.r1
		rmx,rmy    = self.rm.T
		ax.plot([x0,x1], [y0,y1], 'o-', color=colors['link'], lw=20, ms=12, mfc=colors['joint'])
		ax.plot(rmx, rmy, 'o', color=colors['markers'], ms=8)
		if rmobs is not None:
			ax.plot(rmobs[:,0], rmobs[:,1], 'o', color=colors['observations'], ms=5)
		ax.axis('equal')
		
	def set_posture(self, q):
		self.q  = q
		self.T  = self._get_transform_matrix(q)