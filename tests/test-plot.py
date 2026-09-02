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
	def test_fun(self):
	    fig=plt.figure()
	    ax=fig.add_subplot(111,projection='3d')
	    ax.plot([0],[0],[0],'ko',markersize=5)
	    Plot.circle3d(ax,(0,0,0),0.5,zDir=[1,1,0],fill='None',alpha=0.3)
	    #Decoration
	    ax.set_xlim(-1,1)
	    ax.set_ylim(-1,1)
	    ax.set_zlim(-1,1)
	    ax.set_box_aspect([1,1,1])
	    ax.set_xlabel("$x_{ecl}$")
	    ax.set_ylabel("$y_{ecl}$")
	    ax.set_zlabel("$z_{ecl}$")
	    ax.view_init(elev=15,azim=1)
	    Plot.pryngles_mark(ax)
	
	    #rgb
	    print(Plot.rgb([27,0.5,0.5]))
	
	    #Color sample
	    Plot.rgb_sample(27)
	    
	    plt.show()
	
	def test_flyby(self):
	
	    Verbose.VERBOSITY=VERB_SIMPLE
	
	    nvecs=Plot.calc_flyby(normal=[0,0,1],lat=60)
	
	    fig=plt.figure()
	    ax=fig.add_subplot(111,projection='3d')
	
	    for i in range(len(nvecs)):
	        ax.scatter(nvecs[i,0],nvecs[i,1],nvecs[i,2],c='r',s=5)
	        ax.text(nvecs[i,0],nvecs[i,1],nvecs[i,2],i)
	
	    ax.set_xlabel("x")
	    ax.set_ylabel("y")
	    ax.set_zlabel("z")
	
	    plt.show()
	    
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_animrb(self):
	
	    Verbose.VERBOSITY=VERB_NONE
	
	    sim=rb.Simulation()
	    ms=1
	    sim.add(m=1)
	    mp=1e-3
	    xp=0.3
	    vp=np.sqrt(sim.G*ms/xp)
	    sim.add(m=mp,x=xp,vy=vp)
	    mm=1e-8
	    xm=0.01
	    vm=np.sqrt(sim.G*mp/xm)
	    sim.add(m=mm,x=xp+xm,vy=vp+vm)
	    P=sim.particles[1].P
	
	    anim=Plot.animate_rebound(sim,traces=True,nsnap=200,axis=True,ms=1)
	
	    anim=Plot.animate_rebound(sim,filename="/tmp/animate-rebound.gif",interval=20)
	
	    plt.show()
	    
	    Verbose.VERBOSITY=VERB_NONE
	
if __name__=="__main__":
   unittest.main(argv=['first-arg-is-ignored'],exit=False)
