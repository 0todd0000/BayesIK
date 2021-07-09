
import os
import numpy as np
import bayesik as bik


#(0) Load and summarize n-link data:
for nlinks in [1,2,3]:
	fnameNPZ  = os.path.join( bik.dirREPO, 'Data', f'links{nlinks}.npz')
	with np.load(fnameNPZ) as Z:
		Q,QB,QLS = Z['Q'], Z['QB'], Z['QLS']
		EB,ELS   = np.abs(QB-Q), np.abs(QLS-Q)
		print( f'nlinks = {nlinks}  ', (EB<ELS).mean(axis=0) )
print()



#(1) Load and summarize two-frame data:
fnameNPZ  = os.path.join( bik.dirREPO, 'Data', 'twoframe.npz')
with np.load(fnameNPZ) as Z:
	Q,QH,QS,QLS,QB = Z['Q'], Z['QH'], Z['QS'], Z['QLS'], Z['QB']
	i              = np.logical_not(np.all(Q==0, axis=1))
	Q,QH,QS,QLS,QB = [x[i]  for x in (Q,QH,QS,QLS,QB)]
	EH,ES,ELS,EB   = np.abs(QH-Q), np.abs(QS-Q), np.abs(QLS-Q), np.abs(QB-Q)
	print( f'Halvorsen     ', (EB<EH).mean(axis=0) )
	print( f'Soderkvist    ', (EB<ES).mean(axis=0) )



