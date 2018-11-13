from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

class arglist:
    platform = 'knl'

def setenvs(inargv):
    args = arglist()
    for i in range(0, len(inargv) - 1):
        if inargv[i] == '--platform' :
            args.platform = inargv[i + 1]
    assert (args.platform == 'knl' or args.platform == 'bdw')
    # print 'Using platform ', args.platform
    # print 'Groups set to ', args.groups
    if (args.platform == 'bdw'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["MKL_NUM_THREADS"] = "8"
        os.environ["OMP_DYNAMIC"] = "false"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    else:
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["MKL_NUM_THREADS"] = "8"
        os.environ["OMP_DYNAMIC"] = "false"
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,explicit,proclist=[4-67]"
        # os.environ["KMP_AFFINITY"] = "granularity=core,verbose,scatter"
    return args
