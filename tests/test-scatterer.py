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
	def test_interface(self):
	    
	    global MySurface
	    Verbose.VERBOSITY=VERB_ALL
	    
	    class MySurface(Scatterer):
	        def __init__(self,**params):
	            if self.register(self,params):
	                verbose(VERB_SIMPLE,f"Initializing {self.params['name']} with hash {self.hash}")
	                #Read parameters of the scatterer
	                self.A=params["A"]
	                #Initialize scatterer
	                self._initialize_scatterer()
	
	        #Mandatory methods
	        def get_albedo(self,eta,zeta,delta,lamb,**params):
	            albedo=self.AA*eta
	            return albedo
	
	        # Private methods to prepare scatterer
	        def _initialize_scatterer(self):
	            self.AA=self.A**2
	        
	    Scatterer.reset_catalogue()
	    S=MySurface(A=1)
	    print(S.hash)
	    print(S.get_albedo(0.5,0,0,0))
	    S=MySurface(A=1)
	    print(S.hash)
	    print(S.get_albedo(0.5,0,0,0))
	    
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_scatters(self):
	    
	    global LA
	    
	    Verbose.VERBOSITY=VERB_ALL
	    
	    print(NeutralSurface().get_albedo(0,0,0,0))
	    print(BlackBodySurface().get_albedo(0,0,0,0))
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_lambsurface(self):
	    
	    global LA
	    
	    Verbose.VERBOSITY=VERB_ALL
	    
	    LA=LambertianGraySurface(AL=0.5)
	    etas=np.linspace(0,1,1000)
	    fig,axs=plt.subplots(1,1)
	
	    ax=axs
	    ax.plot(etas,LA.get_albedo(etas,0,0,0))
	    ax.set_xlabel(r"$\zeta = \cos Z$")
	    ax.set_ylabel(r"$\alpha$")
	    ax.set_title(rf"Planetary Lambertian Albedo, $A_L=${LA.AL}");
	
	    fig.tight_layout()
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_lambatmos(self):
	    
	    global LA
	    
	    Verbose.VERBOSITY=VERB_ALL
	    
	    LA=LambertianGrayAtmosphere(AS=0.5)
	    etas=np.linspace(0,1,1000)
	    fig,axs=plt.subplots(1,1)
	
	    ax=axs
	    ax.plot(etas,LA.get_albedo(etas,0,0,0))
	    ax.set_xlabel(r"$\zeta = \cos Z$")
	    ax.set_ylabel(r"$\alpha$")
	    ax.set_title(rf"Atmospheric Lambertian Albedo, $A_S=${LA.AS}");
	
	    fig.tight_layout()
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_catalogue(self):
	    
	    global LA
	    
	    Verbose.VERBOSITY=VERB_NONE
	    
	    for key,item in SCATTERERS_CATALOGUE.items():
	        print(f"{key}: {item}")
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_upscat(self):
	    
	    global sys
	    
	    Verbose.VERBOSITY=VERB_NONE
	    
	    nspangles=50
	    sys=System()
	    S=sys.add(kind="Star",nspangles=nspangles,
	              radius=Consts.rsun/sys.ul,limb_coeffs=[0.65])
	    P=sys.add(kind="Planet",parent=S,nspangles=nspangles,
	              a=0.2,e=0.0,radius=Consts.rsaturn/sys.ul,spangle_type=SPANGLE_ATMOSPHERIC)
	    R=sys.add(kind="Ring",parent=P,nspangles=nspangles,
	              fi=1.5,fe=2.5,i=30*Consts.deg,roll=-90*Consts.deg)
	    
	    
	    #Prepare system
	    sys.initialize_simulation()
	    sys.spangle_system()    
	    sys.update_scatterers()
	    
	    #Show scatterers
	    for key,item in SCATTERERS_CATALOGUE.items():
	        print(f"{key}: {item.params['name']}")
	        
	    #Show scatterers
	    print(sys.data.scatterer.unique())
	    
	    Verbose.VERBOSITY=VERB_NONE
	
if __name__=="__main__":
   unittest.main(argv=['first-arg-is-ignored'],exit=False)
