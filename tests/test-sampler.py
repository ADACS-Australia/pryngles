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
	def test_circle(self):
	
	    Verbose.VERBOSITY=VERB_ALL
	
	    #Generate circle
	    S = Sampler(N=1000, seed=10)
	    S.gen_circle()
	    S.plot()
	    S.plot(c='b', spangled=dict(color='r'))
	    S.ax.set_title(f"N = {S.N}, dmed = {S.dmed:.4f}, deff = {S.deff:.4f}", fontsize=10)
	    S.fig.tight_layout()
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_cut(self):
	
	    Verbose.VERBOSITY=VERB_ALL
	
	    #Generate circle
	    S = Sampler(N=1000, seed=10)
	    S.gen_circle()
	    S._cut_hole(0.5)
	    S.plot()
	    S.plot(c='b',spangled=dict(color='r'))
	    S.ax.set_title(f"N = {S.N}, dmed = {S.dmed:.4f}, deff = {S.deff:.4f}", fontsize=10)
	    S.fig.tight_layout()
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_ring(self):
	
	    Verbose.VERBOSITY=VERB_ALL
	
	    #Generate rings
	    S = Sampler(N=500,seed=10)
	    fi = 0.6
	    S.gen_ring(fi)
	    print(S.aes)
	
	    #Test area
	    print(S.A)
	    print(np.pi * (1-fi**2))
	    print(np.pi * (S.deff/2)**2*S.N)
	
	    S.plot()
	    S.plot(spangled = dict(color='r'))
	    S.ax.set_title(f"N = {S.N}, dmed = {S.dmed:.4f}, deff = {S.deff:.4f}",fontsize = 10)
	    S.fig.tight_layout()
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_sphere(self):
	
	    Verbose.VERBOSITY=VERB_ALL
	
	    #Generate sphere
	    S = Sampler(N=100, seed=10)
	    S.gen_sphere()
	    S.plot()
	    S.plot(spangled=dict(color='r'))
	    S.ax.set_title(f"N = {S.N}, dmed = {S.dmed:.4f}, deff = {S.deff:.4f}", fontsize=10)
	    S.fig.tight_layout()
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_purge(self):
	
	    Verbose.VERBOSITY=VERB_ALL
	
	    #Generate sphere
	    S=Sampler(N=1000, seed=10)
	    S.gen_sphere()
	    S.purge_sample()
	    S.plot()
	    S.ax.set_title(f"N = {S.N}, dmed = {S.dmed:.4f}, deff = {S.deff:.4f}", fontsize=10)
	    S.fig.tight_layout()
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_update(self):
	
	    Verbose.VERBOSITY=VERB_ALL
	
	    S=Sampler(N=100, seed=10)
	
	    #Generate ring
	    fi = 0.6
	    S.gen_ring(fi)
	    ss=S.ss*2
	    S.ns=S.update_normals(S.ss)
	    S.plot(spangled=dict(color='r'))
	
	    #Generate sphere
	    S.gen_sphere()
	    ss=S.ss*2
	    S.ns=S.update_normals(S.ss)
	    S.plot(spangled=dict(color='r'))
	    S.ax.set_title(f"N = {S.N}, dmed = {S.dmed:.4f}, deff = {S.deff:.4f}", fontsize=10)
	    S.fig.tight_layout()
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_pre(self):
	
	    Verbose.VERBOSITY=VERB_ALL
	
	    sp=Sampler(preset=("sphere",dict()), N=2750)
	    print(sp.Npreset,sp.N)
	    sp.plot(spangled=dict(color='b', alpha=0.1))
	
	    sp=Sampler(preset=("circle",dict()), N=850)
	    print(sp.Npreset, sp.N)
	    sp.plot(spangled=dict(color='b', alpha=0.1))
	
	    sp=Sampler(preset=("ring",dict(ri=0.7)), N=1150)
	    print(sp.Npreset, sp.N)
	    sp.plot(spangled=dict(color='b', alpha=0.1))
	
	    Verbose.VERBOSITY=VERB_NONE
	
if __name__=="__main__":
   unittest.main(argv=['first-arg-is-ignored'],exit=False)
