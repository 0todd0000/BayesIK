
import os
from math import pi
import numpy as np
from matplotlib import pyplot as plt
import bayesik as bik




#(0) Load 1-link results:
nlinks    = 1
fnameNPZ  = os.path.join( bik.dirREPO, 'Data', f'links{nlinks}.npz')
scale     = [1000, 1000, 180/pi]
with np.load(fnameNPZ) as Z:
	Q     = scale * Z['Q']
	QB    = scale * Z['QB']
	QLS   = scale * Z['QLS']
EB0,ELS0  = QB - Q, QLS - Q


#(1) Load two-frame results:
fnameNPZ  = os.path.join( bik.dirREPO, 'Data', f'twoframe-sigma.npz')
scale     = [1000, 1000, 180/pi]
with np.load(fnameNPZ) as Z:
	Q     = scale * Z['Q']
	QH    = scale * Z['QH']
	QS    = scale * Z['QS']
	QB    = scale * Z['QB']
EH1,ES1,EB1 = QH - Q, QS - Q, QB - Q




plt.close('all')
fig,AX = plt.subplots( 2, 3, figsize=(10,5) )
fontname = u'Times New Roman'

c0,a0,ec0  = 'k', 1, '0.7'
c1,a1,ec1  = '0.8', 0.8, 'k'
cB,aB,ecB  = 'b', 0.8, '0'

### 1-link errors:
rangesLS = [(-7,7), (-7,7), (-2,2)]
rangesB  = rangesLS
# rangesB  = [(-2,2), (-2,2), (-0.5,0.5)]
for i,ax in enumerate(AX[0]):
	ax.hist(ELS0[:,i], bins=21, range=rangesLS[i], color=c0, ec=ec0, alpha=a0, zorder=-1, label='Least-squares')
	ax.hist(EB0[:,i],  bins=21, range=rangesB[i], color=cB, ec=ecB, alpha=aB, label='Bayesian')
	if i==0:
		leg = ax.legend(loc='upper right', bbox_to_anchor=(1.1,1))
		plt.setp(leg.get_texts(), size=8, name=fontname)

### two-frame errors:
rangesLS = [(-20,20), (-20,20), (-10,10)]
rangesB  = rangesLS
for i,ax in enumerate(AX[1]):
	ax.hist(EH1[:,i], bins=21, range=rangesLS[i], color=c0, ec=ec0, alpha=a0, zorder=-1, label='Halvorsen')
	ax.hist(ES1[:,i], bins=21, range=rangesLS[i], color=c1, ec=ec1, alpha=a1, zorder=-1, label='Soderkvist')
	ax.hist(EB1[:,i], bins=21, range=rangesB[i],  color=cB, ec=ecB, alpha=aB, zorder=-1, label='Bayesian')
	if i==0:
		leg = ax.legend(loc='upper right', bbox_to_anchor=(1.1,1))
		plt.setp(leg.get_texts(), size=8, name=fontname)


### label axes:
[ax.set_ylabel('Frequency', name=fontname, size=14) for ax in AX[:,0]]
labels = [r'$r_x$  $\textrm{error (mm)}$', r'$r_y$  $\textrm{error (mm)}$', r'$\phi$  $\textrm{error (mm)}$']
[ax.set_xlabel(s, size=14, usetex=True, name=fontname)   for ax,s in zip(AX[1],labels)]
[plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), name=fontname, size=9) for ax in AX.ravel()]

labels = '1-link posture', '1-link rotation'
[ax.text(-0.35, 0.5, label, rotation=90, va='center', transform=ax.transAxes, name=fontname, size=18) for ax,label in zip(AX[:,0],labels)]
[ax.text(0.05, 0.9, '(%s)'%chr(97+i), transform=ax.transAxes, name=fontname, size=14) for i,ax in enumerate(AX.flatten())]


plt.tight_layout()
plt.show()



# plt.savefig(   os.path.join( bik.dirREPO, 'Figures', f'fig5.pdf')   )



