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
	def test_coords(self):
	
	    #Test axis
	    axis=[
	        [+1,+0,+0],[-1,+0,+0],
	        [+0,+1,+0],[+0,-1,+0],
	        [+0,+0,+1],[+0,+0,-1],
	    ]
	    for i,xyz in enumerate(axis):
	        rqf=Science.spherical(xyz)
	        cqsqsf=Science.cospherical(xyz)
	        rhofct=Science.pcylindrical(xyz)
	        print(f"Axis {i+1}:")
	        print("\tSpherical:",rqf[0],rqf[1]*Consts.rad,rqf[2]*Consts.rad)
	        print("\tCospherical:",cqsqsf)
	        print("\tVerify:",mh.cos(rqf[1]),mh.sin(rqf[1]),mh.sin(rqf[2]))
	        print("\tPseudo cilyndrical:",rhofct)
	        print("\tVerify:",rqf[0]*np.cos(rqf[2]),rqf[1],mh.sin(rqf[2]))
	
	    #Test spherical
	    octants=[
	        [+1,+1,+1],[-1,+1,+1],[-1,-1,+1],[+1,-1,+1],
	        [+1,+1,-1],[-1,+1,-1],[-1,-1,-1],[+1,-1,-1]
	    ]
	    for i,xyz in enumerate(octants):
	        rqf=Science.spherical(xyz)
	        cqsqsf=Science.cospherical(xyz)
	        rhofct=Science.pcylindrical(xyz)
	        print(f"Octant {i+1}:")
	        print("\tSpherical:",rqf[0],rqf[1]*Consts.rad,rqf[2]*Consts.rad)
	        print("\tCospherical:",cqsqsf)
	        print("\tVerify:",mh.cos(rqf[1]),mh.sin(rqf[1]),mh.sin(rqf[2]))
	        print("\tPseudo cilyndrical:",rhofct)
	        print("\tVerify:",rqf[0]*np.cos(rqf[2]),rqf[1],mh.sin(rqf[2]))
	
	    #Test cartesian
	    octants=[
	        [1,45*Consts.deg,45*Consts.deg],[1,135*Consts.deg,45*Consts.deg],
	        [1,225*Consts.deg,45*Consts.deg],[1,315*Consts.deg,45*Consts.deg],
	        [1,45*Consts.deg,-45*Consts.deg],[1,135*Consts.deg,-45*Consts.deg],
	        [1,225*Consts.deg,-45*Consts.deg],[1,315*Consts.deg,-45*Consts.deg]
	    ]
	    for i,rqf in enumerate(octants):
	        xyz=Science.cartesian(rqf)
	        print(f"Octant {i+1}:",xyz) 
	
	    #Test direction
	    nvec=Science.direction(120,45)
	    print(nvec,Science.direction(*nvec))
	
	def test_rot(self):
	
	    Verbose.VERBOSITY=VERB_ALL
	
	    #Test rotation
	    Msys2uni,Muni2sys=Science.rotation_matrix([0,0,1],0)
	    print(Msys2uni)
	
	    Msys2uni,Muni2sys=Science.rotation_matrix([1,0,-1],0)
	    print(Msys2uni)
	    print(Muni2sys)
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_limb(self):
	
	    Verbose.VERBOSITY=VERB_ALL
	
	    cs=[np.random.rand()]
	    I=Science.limb_darkening(0.8,cs)
	    print(I)
	
	    fig=plt.figure()
	    ax=fig.gca()
	
	    rhos=np.linspace(0,1,100)
	    coefs=[0.6550]
	    ax.plot(rhos,Science.limb_darkening(rhos,coefs))
	    coefs=[0.6022,0.0654]
	    ax.plot(rhos,Science.limb_darkening(rhos,coefs))
	    coefs=[0.9724,-0.4962,0.2029]
	    ax.plot(rhos,Science.limb_darkening(rhos,coefs))    
	    coefs=[-0.2018,2.1000,-2.0247,0.7567]
	    ax.plot(rhos,Science.limb_darkening(rhos,coefs))
	    Plot.pryngles_mark(ax)
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_hull(self):
	
	    Verbose.VERBOSITY=VERB_ALL
	
	    rng = np.random.default_rng()
	    points = rng.random((30, 2))
	    hull = Science.get_convexhull(points)
	
	    ps = rng.random((30, 2))-0.5
	    cond=Science.points_in_hull(ps,hull)
	    print(len(cond),len(ps))
	
	    import matplotlib.pyplot as plt
	    plt.figure()
	    for simplex in hull.simplices:
	        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
	
	    for p in ps[cond]:
	        plt.plot(p[0],p[1],'r*')
	
	    for p in ps[~cond]:
	        plt.plot(p[0],p[1],'co')
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_plane(self):
	
	    Verbose.VERBOSITY=VERB_SIMPLE
	
	    plt.close("all")
	
	    #Calculate plane
	
	    #Test plane
	    p1=[-1,2,1]
	    p2=[0,-3,2]
	    p3=[1,1,-4]
	    plane=Plane(p1,p2,p3)
	    print(plane)
	    #Debe dar: {'a': 26, 'b': 7, 'c': 9, 'd': 3}
	
	            #Check if point is above with respect to a direction
	    p=[2,2,5]
	    print("Is above: ",plane.is_above(p,[0,0,-1]))
	
	    v=plane.get_projection(p)
	    print("Projection: ",v)
	    plane.plot_plane(p=p,alpha=0.1,color='r')
	
	    Verbose.VERBOSITY=VERB_NONE
	
if __name__=="__main__":
   unittest.main(argv=['first-arg-is-ignored'],exit=False)
