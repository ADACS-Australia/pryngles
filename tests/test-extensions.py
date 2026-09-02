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
	def test_readf(self):
	    Verbose.VERBOSITY=VERB_SIMPLE
	    
	    filename=Misc.get_data("fou_gasplanet.dat")
	    S=StokesScatterer(filename)
	    Verbose.VERBOSITY=VERB_NONE
	    pass
	
	def test_fun(self):
	    global cpixx_ext
	    
	    Verbose.VERBOSITY=VERB_SIMPLE
	    
	    #Test routines
	    n=3
	    m=3
	    p=3
	    xmu=np.random.rand(m)
	    rfou=np.random.rand(n,m,p)
	    rtra=np.random.rand(n,m,p)
	    print(rfou)
	    print(rfou.sum())
	    F=FourierCoefficients(n,m,p,xmu,rfou,rtra)
	    suma=cpixx_ext.sum_structure(F,n,m,p)
	    print(suma)
	    C=ExtensionUtil.ptr2cub(F.rfou,*rfou.shape)
	    print(C)
	    
	    Verbose.VERBOSITY=VERB_NONE
	    pass
	
	def test_stokes(self):
	    global S,phi,beta,theta0,theta,apix
	    Verbose.VERBOSITY=VERB_SIMPLE
	
	    #Test values
	    npix=1
	    phi=np.zeros(npix)
	    beta=np.zeros(npix)
	    theta0=np.zeros(npix)
	    theta=np.zeros(npix)
	    apix=np.zeros(npix)
	
	    #######################
	    #Planet
	    #######################
	    #Scatterer
	    filename=Misc.get_data("fou_gasplanet.dat")
	    S=StokesScatterer(filename)
	    
	    phi[0]=1.0448451569439827;
	    beta[0]=3.069394277348945;
	    theta0[0]=0.04990329026929557;
	    theta[0]=0.02509670973070432;
	    apix[0]=9.432328787795567e-05;
	    print("Expected values:\n4.597857424902560283e-07 2.251229972872198058e-07 1.834400800563127439e-09 4.896421313424954569e-01");
	    
	    #Calculate albedo
	    stokes=S.calculate_stokes(phi,beta,theta0,theta,apix)
	    print(stokes)
	    
	    #######################
	    #Ring
	    #######################
	    #Scatterer
	    filename=Misc.get_data("fou_ring_0_4_0_8.dat")
	    S=StokesScatterer(filename)
	    
	    #Test values for backscattering
	    phi[0]=1.490116119384765625e-08;
	    beta[0]=0.000000000000000000e+00;
	    theta0[0]=4.999999999999998890e-01;
	    theta[0]=5.000000000000000000e-01;
	    apix[0]=1.163314390931110409e-04;
	    print("Expected:\n6.193646058775441171e-06 -3.046132218287162406e-07 -9.759550180589223642e-15 4.918156751904256135e-02");  
	
	    #Calculate albedo
	    stokes=S.calculate_stokes(phi,beta,theta0,theta,apix)
	    print(stokes)
	    
	    #Test values for forwardscattering
	    phi[0]=1.601029385538801364e+00;
	    beta[0]=1.601029385538801364e+00;
	    theta0[0]=1.744974835125044643e-02;
	    theta[0]=5.000000000000000000e-01;
	    apix[0]=1.163314390931110409e-04;
	    print("Expected:\n1.688771016436214060e-07 -1.485273114913191718e-08 2.674513729889046925e-09 8.936444506313186154e-02");  
	
	    #Calculate albedo
	    stokes=S.calculate_stokes(phi,beta,theta0,theta,apix,qreflection=0)
	    print(stokes)
	
	    Verbose.VERBOSITY=VERB_NONE
	    pass
	
	def test_stokes_mass(self):
	    from time import time
	    global S,phi,beta,theta0,theta,apix
	    Verbose.VERBOSITY=VERB_SIMPLE
	
	    #Interpolation file
	    filename=Misc.get_data("fou_ring_0_4_0_8.dat")
	    fileinterp=Misc.get_data("ring-interpolation.mat")
	    
	    data=np.loadtxt(fileinterp)
	    npix=len(data)
	    phi=data[:npix,0].copy()
	    beta=data[:npix,1].copy()
	    theta0=data[:npix,2].copy()
	    theta=data[:npix,3].copy()
	    apix=data[:npix,7].copy()
	    spixx=data[:npix,8:]
	
	    #Scatterer
	    S=StokesScatterer(filename)
	    
	    #Calculate albedo
	    st=time()
	    stokes=S.calculate_stokes(phi,beta,theta0,theta,apix)
	    et=time()
	    print("Maximum difference:",abs(spixx-stokes).max())
	    print(f"Calculation time (per vector): {(et-st)/npix*1e3} ms")
	    
	    Verbose.VERBOSITY=VERB_NONE
	    pass
	
if __name__=="__main__":
   unittest.main(argv=['first-arg-is-ignored'],exit=False)
