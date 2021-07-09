
import os
from math import pi
import numpy as np
from matplotlib import pyplot as plt
import bayesik as bik


fast      = False


nlinks    = 3
dirname   = 'sim-nlink-fast' if fast else 'sim-nlink'
fnameNPZ  = os.path.join( bik.dirREPO, 'Data', dirname, f'links{nlinks}.npz')
scale     = [1000, 1000] + [180/pi]*nlinks
with np.load(fnameNPZ) as Z:
	Q     = scale * Z['Q']
	QB    = scale * Z['QB']
	QLS   = scale * Z['QLS']
EB0,ELS0  = QB - Q, QLS - Q




plt.close('all')
fig,AX = plt.subplots( 2, 3, figsize=(8,5) )
fontname = u'Times New Roman'
AX[0,2].set_visible(False)
AX       = AX.flatten()[[0,1,3,4,5]]

c0,a0,ec0  = 'k', 1, '0.7'
cB,aB,ecB  = 'b', 0.7, '0'

### posture model errors:
rangesLS = [(-5,5), (-5,5), (-2,2), (-2,2), (-2,2)]
rangesB  = rangesLS
for i,ax in enumerate(AX):
	ax.hist(ELS0[:,i], bins=21, range=rangesLS[i], color=c0, ec=ec0, alpha=a0, zorder=-1, label='Least-squares')
	ax.hist(EB0[:,i],  bins=21, range=rangesB[i], color=cB, ec=ecB, alpha=aB, label='Bayesian')
	if i==0:
		leg = ax.legend(loc='upper right', bbox_to_anchor=(1.1,1))
		plt.setp(leg.get_texts(), size=8, name=fontname)


### label axes:
[ax.set_ylabel('Frequency', name=fontname, size=14) for ax in AX[[0,2]]]
labels = [r'$r_x$  $\textrm{error (mm)}$', r'$r_y$  $\textrm{error (mm)}$', r'$\phi_1$  $\textrm{error (mm)}$', r'$\phi_2$  $\textrm{error (mm)}$', r'$\phi_3$  $\textrm{error (mm)}$']
[ax.set_xlabel(s, size=14, usetex=True, name=fontname)   for ax,s in zip(AX,labels)]
[plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), name=fontname, size=9) for ax in AX.ravel()]


plt.tight_layout()
plt.show()


# plt.savefig(   os.path.join( bik.dirREPO, 'Appendix', 'ipynb', 'figs', f'error-{nlinks}-link.png')   )



