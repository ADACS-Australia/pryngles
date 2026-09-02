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
	def test_orbody(self):
	
	    S=Orbody(m=1)
	    P=Orbody(parent=S,m=0.1,a=1,e=0.5)
	
	    print(S.m,S.elements)
	    print(P.m,P.elements)
	
	    S.show_tree()
	
	def test_Orbit(self):
	
	    global S4
	
	    #Quintuple system
	    S1=Orbit(m1=1,m2=1,a=1,e=0.7,M=0)
	    S2=Orbit(m1=1,m2=1,a=1,e=0,M=0)
	    S3=Orbit(m1=S1,m2=S2,a=5,e=0)
	    S4=Orbit(m1=S3,m2=1,a=20,e=0,E=45*Consts.deg)
	    S4.calculate_orbit()
	    print(S4.get_states())
	    print(S4.Ps)
	
	    #Using custom units
	    #Quintuple system
	
	    #Initialize positions
	    units=['au','msun','yr']
	    hn=Orbit(
	        m1=1,
	        m2=Orbit(m1=1e-3,m2=1e-7,a=0.5,e=0.0,units=units),
	        units=units,
	        a=20,e=0.0)
	    hn.calculate_orbit()
	    sim,states=hn.get_states()
	    print(states)
	
	    #SImple 
	
	    #Use this code to animate:
	    #Plot.animate_rebound(S4.sim)
	
	def test_build(self):
	
	    global orbit
	    
	    units=['au','msun','yr']
	    S1=Orbody(name="Star1",m=1)
	    P1S1=Orbody(name="Planet1Star1",parent=S1,m=0.1,a=1,e=0.2)
	    P2S1=Orbody(name="Planet1Star1",parent=S1,m=0.1,a=2,e=0.2)
	    M1P1=Orbody(name="Moon1Planet1",parent=P1S1,m=0.01,a=0.1,e=0.5)
	    SM1M1=Orbody(name="Moon1Planet1",parent=P1S1,m=0.001,a=0.01,e=0.2)
	
	    S2=Orbody(name="Star1",m=1,a=5,e=0.3)
	    P1S2=Orbody(name="Planet1Star2",parent=S2,m=0.1,a=1.5,e=0.5)
	
	    orbital_tree=[[S1,[P1S1,[M1P1,SM1M1]]],[S2,P1S2]]
	    orbit,pelements=OrbitUtil.build_system(orbital_tree,units)
	    orbit.calculate_orbit()
	    #Plot.animate_rebound(orbit.sim,filename="tmp/hierarchical-system.mp4",color='b',ms=3)
	
if __name__=="__main__":
   unittest.main(argv=['first-arg-is-ignored'],exit=False)
