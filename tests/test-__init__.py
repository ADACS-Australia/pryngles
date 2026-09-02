##################################################################
#                                                                #
#.#####...#####...##..##..##..##...####...##......######...####..#
#.##..##..##..##...####...###.##..##......##......##......##.....#
#.#####...#####.....##....##.###..##.###..##......####.....####..#
#.##......##..##....##....##..##..##..##..##......##..........##.#
#.##......##..##....##....##..##...####...######..######...####..#
#................................................................#
#                                                                #
# PlanetaRY spanGLES                                             #
#                                                                #
##################################################################
# License http://github.com/seap-udea/pryngles-public            #
##################################################################
import unittest
from pryngles import *
class Test(unittest.TestCase):
	def test_common(self):
	    import numpy as np
	    import pandas as pd
	    import os
	
	    Verbose.VERBOSITY=VERB_ALL
	
	    p=PrynglesCommon()
	    p.casa=dict(perro=0,gato=3)
	    p.data=pd.DataFrame(np.random.rand(4000,100))
	    p.save_to("/tmp/save.pck")
	    print("File size:",os.path.getsize("/tmp/save.pck")/1e6," Mb")
	    g=PrynglesCommon()
	    g.load_from("/tmp/save.pck")
	    print(g.casa,np.array(g.data).shape)
	
	    Verbose.VERBOSITY=VERB_NONE
	
if __name__=="__main__":
   unittest.main(argv=['first-arg-is-ignored'],exit=False)
