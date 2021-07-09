


from math import radians
import numpy as np
from matplotlib import pyplot as plt
import bayesik as bik






#(0) Plot segment:
np.random.seed(0)
sigma       = 0.002
q           = np.array([0, 0, radians(30)])
seg         = bik.nlink.Segment(length=0.45)
seg.set_posture( q )
rmobs       = seg.get_rm_noisy( sigma )
### plot:
plt.close('all')
plt.figure()
ax = plt.axes()
seg.plot(ax=ax, rmobs=rmobs)
ax.set_title('Single segment', size=14)
plt.show()



#(1) Plot chain:
np.random.seed(1)
sigma        = 0.005
q,seglengths = np.array([0, 0, radians(20)]), [0.45]
q,seglengths = np.array([0, 0, radians(20), radians(45)]), [0.45,0.35]
q,seglengths = np.array([0, 0, radians(20), radians(45), radians(90)]), [0.45,0.35,0.25]
chain        = bik.nlink.KinematicChain( seglengths )
chain.set_posture( q )
rmobs        = chain.get_rm_noisy( sigma )
### plot:
plt.figure()
ax = plt.axes()
# chain.plot(ax=ax, rmobs=None)
chain.plot(ax=ax, rmobs=rmobs)
ax.set_title('n-link Kinematic Chain', size=14)
plt.show()












