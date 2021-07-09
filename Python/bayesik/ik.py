
from math import acos
import numpy as np


def _asmatrixlist(r):
	return [ np.matrix( rr.tolist() + [0] ).T for rr in r]

def halvorsen(r0, r1):
	'''
	Halvorsen KM, Lesser M, Lundberg A (1999) 
	A new method for estimating the axis of rotation and the center of rotation.
	Journal of Biomechanics 32(11): 1221-1227.
	'''
	P0,P1        = _asmatrixlist(r0), _asmatrixlist(r1)
	### compute instantaneous rotation using Halvorsen (1999) method:
	DP           = [p1-p0   for p0,p1 in zip(P0,P1)]  #marker displacements
	### compute rotation axis (Equation 2):
	C            = np.matrix(   np.zeros((3,3))   )
	for dp in DP:
		C       += dp*dp.T
	val,vec      = np.linalg.eig( C )
	ind          = val.argmin()
	omega        = vec[:,ind]   #instantaneous rotation axis
	### compute center of rotation:
	dP           = np.vstack( [dp.T  for dp in DP ]  )
	b            = np.vstack( [dp.T * (0.5 * (p0+p1))  for dp,p0,p1 in zip(DP,P0,P1) ]  )
	q            = np.linalg.pinv(dP) * b
	q            = np.asarray(q).flatten()
	### compute angular displacement:  (not from original article)
	m0,m1        = [np.squeeze(np.array(P)).mean(axis=0)   for P in (P0,P1)]
	sgn          = np.sign( np.cross(m0, m1)[2] )
	a,b,c        = m0-q, m1-q, m1-m0
	a,b,c        = [np.linalg.norm(x)  for x in (a,b,c)]
	theta        = sgn * acos(  (a**2+b**2-c**2) / (2*a*b) )
	return q, theta


def soderkvist(x, y):
	'''
	Soderkvist I, Wedin PA (1993)
	Determining the movements of the skeleton using well-configured markers.
	Journal of Biomechanics 26(12): 1473-1477.
	'''
	xm,ym = x.mean(axis=1), y.mean(axis=1)
	A,B   = np.matrix((x.T-xm).T), np.matrix((y.T-ym).T)
	C     = B * A.T
	u,s,v = np.linalg.svd(C)
	detPQ = np.linalg.det(u*v.T)
	R     = u * np.diag([1,detPQ]) * v.T
	d     = np.asarray(   np.matrix(ym).T - R*np.matrix(xm).T   ).flatten()
	sgn   = np.sign( R[1,0] )
	theta = sgn * acos(R[0,0])
	return d, theta