
import pathlib
from . import ik
from . import nlink
from . import util


dirREPO         = pathlib.Path( __file__ ).parent.parent.parent
KinematicChain  = nlink.KinematicChain
Segment         = nlink.Segment
SegmentTwoFrame = twoframe.SegmentTwoFrame

