
import os
from math import pi
import numpy as np
from matplotlib import pyplot as plt
import bayesik as bik




#(0) Load two-frame results:
fnameNPZ  = os.path.join( bik.dirREPO, 'Data', f'twoframe-sigma.npz')


scale     = [1000, 1000, 180/pi]
with np.load(fnameNPZ) as Z:
	Q     = scale * Z['Q']
	QH    = scale * Z['QH']
	QS    = scale * Z['QS']
	QB    = scale * Z['QB']
	TH    = Z['TH']
	TS    = Z['TS']
	TB    = Z['TB']
	SIGMA = 1000 * Z['SIGMA']
	THETA = Z['THETA']
### calculate errors:
EH,ES,EB  = np.abs(QH-Q), np.abs(QS-Q), np.abs(QB-Q)




#(1) Separate into noise
def windowed_averages(x, y, centers, width):
	m,s = [],[]
	for c in centers:
		x0,x1  = c-width, c+width
		i   = (x >= x0) & (x<x1)
		if i.sum()==0:
			m.append(0)
			s.append(0)
		else:
			m.append(  np.median( np.abs(y[i]), axis=0 )  )
			s.append( np.percentile(np.abs(y[i]), [50 ,75], axis=0) )
	m,s = np.array(m), np.array(s)
	return m, s

centers,width = 1e-3*np.arange(200, 2000, 200), 1e-3*100
mh,sh         = windowed_averages(SIGMA, EH, centers, width)
ms,ss         = windowed_averages(SIGMA, ES, centers, width)
mb,sb         = windowed_averages(SIGMA, EB, centers, width)



#(2) Plot
plt.close('all')
fig,AX = plt.subplots( 2, 3, figsize=(10,5) )
fontname = u'Times New Roman'

colors   = '0.4', '0.8', 'b'
colorss  = '0.4', '0.8', 'b'

labels   = 'Error (mm)', 'Error (mm)', 'Error (deg)'
c        = centers
plabels  = r'$r_x$', r'$r_y$', r'$\Delta\phi$'

for i,ax in enumerate(AX[0]):
	ax.bar(c, mh[:,i], 0.18, label='Halvorsen (1999)', color=colors[0])
	ax.bar(c, ms[:,i], 0.14, label='Soderkvist (1993)', color=colors[1])
	ax.bar(c, mb[:,i], 0.10, label='Bayesian', color=colors[2])
	
	for cc,ssh,sss,ssb in zip(c,sh[:,:,i],ss[:,:,i],sb[:,:,i]):
		ax.plot([cc,cc], ssh, color=colorss[0], lw=2)
		ax.plot([cc,cc], sss, color=colorss[1], lw=2)
		ax.plot([cc,cc], ssb, color=colorss[2], lw=2)

	ax.set_xticks(c)
	ax.set_ylabel(labels[i], name=fontname, size=14)
	if i==2:
		leg = ax.legend(loc='upper left', bbox_to_anchor=(0.15,0.98))
		plt.setp(leg.get_texts(), name=fontname)
	ax.text(0.03, 0.89, '(%s)'%chr(97+i), name=fontname, size=16, transform=ax.transAxes)
	ax.text(0.50, 1.05, plabels[i], name=fontname, size=16, transform=ax.transAxes, ha='center')
	


AX[0,2].set_ylim(-0.1, 6.5)
plt.setp(AX[0], xticklabels=[])



for i,ax in enumerate(AX[1]):
	ax.bar(c, mb[:,i], 0.12, label='Bayesian', color=colors[2])
	
	for cc,ssh,sss,ssb in zip(c,sh[:,:,i],ss[:,:,i],sb[:,:,i]):
		ax.plot([cc,cc], ssb, color=colorss[2], lw=2)

	ax.set_xticks(c)
	ax.set_xlabel('Noise SD (mm)', name=fontname, size=14)
	ax.set_ylabel(labels[i], name=fontname, size=14)
	if i==2:
		leg = ax.legend(loc='upper left', bbox_to_anchor=(0.01,0.85))
		plt.setp(leg.get_texts(), name=fontname)
	ax.text(0.03, 0.89, '(%s)'%chr(97+3+i), name=fontname, size=16, transform=ax.transAxes)



plt.tight_layout()
plt.show()



# plt.savefig(   os.path.join( bik.dirREPO, 'Figures', f'fig6.pdf')   )



