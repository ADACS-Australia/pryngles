import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pryngles import *
from pryngles import pixx
import time
extension="pixx"
#extension="cpixx"
import multiprocessing as mp
import os,sys,glob
import matplotlib.ticker as ticker
from scipy.optimize import minimize, differential_evolution,least_squares
import cv2 as cv

def pool_handler(loc_num, params, func, multiplier,max_process=None):
    
    if max_process is None:
        max_process = math.floor(mp.cpu_count()*multiplier)
    
    # Max allowed thread count on the server is 14
    if max_process>14: max_process=14
    
    if loc_num > max_process:
        run_num = 0
        while loc_num > 0:
            lbound = run_num*max_process
            if loc_num > max_process: ubound = (run_num+1)*max_process
            else: ubound = lbound + loc_num
            processes = [mp.Process(target=func, args=(params[i]))
                        for i in range(lbound, ubound)]
            for p in processes:
                time.sleep(0.1)
                p.start()
            for p in processes:
                time.sleep(0.1)
                p.join()
            loc_num = loc_num - max_process
            run_num+=1
    else:
        processes = [mp.Process(target=func, args=(params[i]))
                    for i in range(0, loc_num)]
        for p in processes:
            time.sleep(0.1)
            p.start()
        for p in processes:
            time.sleep(0.1)
            p.join()
    
def crossCalc(ring_l,ring_i,orbit_i):
    def ring_plane_cross(x,lambdar,gamma,i):
        return np.sin(x[0])*np.sin(lambdar)*np.cos(gamma) - \
                 np.sin(gamma)*np.cos(i)*np.cos(x[0]) + \
                 np.cos(lambdar)*np.cos(gamma)*np.sin(i)*np.cos(x[0])

    lambdar = ring_l*np.pi/180
    gamma   = ring_i*np.pi/180
    i       = orbit_i*np.pi/180
    x0_array = np.linspace(0,2*np.pi,100)
    results = np.zeros(2)

    for ii,x0 in enumerate(x0_array):
        res = least_squares(ring_plane_cross, np.array([x0]), args=(lambdar,gamma,i))
        nu = res.x
        if ii == 0:
            results[0] = nu%(2*np.pi)
        elif abs(nu%(2*np.pi)-results[0]) > 1e-1:
            results[1] = nu%(2*np.pi)
            break
    results = np.sort(results)
    return results*180/np.pi

def normalConverter(i,phi):
    def func(x,i,phi):
        return [np.sin(x[1])*np.cos(x[0]) - np.sin(phi)*np.sin(i),
                np.sin(x[0]) - np.cos(phi)*np.sin(i),
                np.cos(x[1])*np.cos(x[0]) - np.cos(i)]
    i_rad = i*np.pi/180
    phi_rad = phi*np.pi/180
    res = least_squares(func, np.array([60*np.pi/180,40*np.pi/180]), args=(i_rad,phi_rad))
    ring_i, ring_l = res.x
    print("Check ring orientation: ", func(np.array([ring_i,ring_l]),i_rad,phi_rad))
    return np.array([ring_i,ring_l])
    

def parametersweepGeom(ring_i_arr: np.ndarray,
                       fou_file_ring: str,
                       fou_file_planet: str,
                       orbit_i: float,
                       ring_l: float,
                       ring_ri: float,
                       ring_re: float,
                       tau_ring: float = 0.4,
                       interpr: str = "spline",
                       reference_plane: str = "Detector", # "Detector" or "Planetary"
                       Ns: int = 30,
                       Nb: int = 0,
                       Np: int = 10000,
                       Nr: int = 10000):
    
    # Announce start of function
    print (f"\n\n start run with: orbit i = {orbit_i}, ring l = {ring_l}") 
    
    # Generate save location if necessary
    if not os.path.isdir(f"/home/allard/Data/Geom_series_new/Orbit_i_{orbit_i}/Ring_L_{ring_l}/Ring_i_{ring_i_arr[0]}"):      
        os.makedirs(f"/home/allard/Data/Geom_series_new/Orbit_i_{orbit_i}/Ring_L_{ring_l}/Ring_i_{ring_i_arr[0]}")
    save_location = f"/home/allard/Data/Geom_series_new/Orbit_i_{orbit_i}/Ring_L_{ring_l}/Ring_i_{ring_i_arr[0]}/"

    # Save log file since no printing comes out of multiprocessing
    sys.stdout = open(save_location + "logfile.out", "w")
    
    # Make sure all output is printed
    Verbose.VERBOSITY=VERB_ALL
    
    # Calculate starting position of observer and star
    gamma, beta_obs, lamb_obs, lamb_star = Util.calcStartingPosition(orbit_i,ring_i_arr[0],ring_l)
      
    # Initialise the system
    pixx_sys = System()
    s=pixx_sys.add(kind="Star",physics=dict(radius=Consts.rsun/pixx_sys.ul),optics=dict(limb_coeffs=[0.65]))
    p=pixx_sys.add(kind="Planet", primary=s,
                   radius=Consts.rsaturn/pixx_sys.ul,
                   orbit=dict(a=1, e=0.0),
                   physics=dict(radius=Consts.rsaturn/pixx_sys.ul),
                   optics=dict(nspangles=Np))
    r=pixx_sys.add(kind="Ring", primary=p,
                   physics=dict(fi=ring_ri, fe=ring_re, i=gamma),
                   optics=dict(nspangles=Nr))
    
    RP=pixx_sys.ensamble_system(extension=extension,
                                fname_planet=fou_file_planet,
                                fname_ring=fou_file_ring)
    RP.behavior["interp_method_ring"] = interpr
    RP.reference_plane = reference_plane
    
    lamb_initial = lamb_star
    lamb_final = lamb_initial + 360*Consts.deg
    lambs = np.linspace(lamb_initial,lamb_final,361)
        
    # Start series looping over the given ring inclinations
    for jj,r_i in enumerate(ring_i_arr):
        # Re-initialise the system for a different ring inclination
        if jj > 0:
            gamma, beta_obs, lamb_obs, lamb_star = Util.calcStartingPosition(orbit_i,r_i,ring_l)
            RP.i = gamma
            RP.changeObserver([lamb_obs,beta_obs])
            RP.Ns = Ns
            RP.Np = Np
            RP.Nr = Nr
            RP.Nb = Nb
            RP.updateProperties()
            lamb_initial = lamb_star
            lamb_final = lamb_initial + 360*Consts.deg
            lambs = np.linspace(lamb_initial,lamb_final,361)
            
        print (f"\n\n start run with: orbit i = {orbit_i}, ring l = {ring_l}, ring i = {r_i}") 
        
        # Generate save location if necessary
        if not os.path.isdir(f"/home/allard/Data/Geom_series_new/Orbit_i_{orbit_i}/Ring_L_{ring_l}/Ring_i_{r_i}"):      
            os.makedirs(f"/home/allard/Data/Geom_series_new/Orbit_i_{orbit_i}/Ring_L_{ring_l}/Ring_i_{r_i}")
        save_location = f"/home/allard/Data/Geom_series_new/Orbit_i_{orbit_i}/Ring_L_{ring_l}/Ring_i_{r_i}/"
        
        # Save log file since no printing comes out of multiprocessing
        sys.stdout = open(save_location + "logfile.out", "w")
        
        # Initialise the starting position
        RP.changeObserver([lamb_obs,beta_obs])
        RP.changeStellarPosition(lamb_initial)
        RP._updateGeometricalFactors()
        RP._updateIncomingStellarFlux()
        RP._updateObservedFacetAreas()
        
        # Save images showing the starting position of planet, ring and star
        ecl_fig,obs_fig,star_fig = RP.plotRingedPlanet(showstar=False,showfig=False,showtitle=False,axis=False)
        ecl_fig.savefig(save_location + f"fig_with_oi_{orbit_i}_rl_{ring_l}_ri_{r_i}_ecl.png", dpi=300)
        obs_fig.savefig(save_location + f"fig_with_oi_{orbit_i}_rl_{ring_l}_ri_{r_i}_obs.png", dpi=300)
        star_fig.savefig(save_location + f"fig_with_oi_{orbit_i}_rl_{ring_l}_ri_{r_i}_star.png", dpi=300)
        plt.close()
        
        # Make lists
        Stot  = []
        Sp    = []
        Sr    = []
        Ptot  = []
        Pp    = []
        Pr    = []
        alpha = []

        # Start the orbit
        for lamb in lambs:
            RP.changeStellarPosition(lamb)
            print("True anomaly: ", (lamb-lamb_initial)/Consts.deg)
            RP._updateGeometricalFactors()
            RP._updateIncomingStellarFlux()
            RP._updateObservedFacetAreas()
            RP.updateReflection(taur=tau_ring)
            
            # Save the relevant data
            Stot  += [RP.Stot]
            Sp    += [RP.Stotp]
            Sr    += [RP.Stotr]
            Ptot  += [RP.Ptot]
            Pp    += [RP.Ptotp]
            Pr    += [RP.Ptotr]
            alpha += [np.arccos(RP.alphaps)/Consts.deg]
            
        true_anomaly = list((lambs-lamb_initial)/Consts.deg)
        parameters = {"Fourier planet": RP.fname_planet, "Fourier ring": RP.fname_ring,
                      "Orbit inclinatie": orbit_i, "Ring roll": ring_l,
                      "Ring inclinatie": r_i, "Ring ri": RP.fi, "Ring re": RP.fe,
                      "Observer inclinatie": RP.eobs_ecl[1]/Consts.deg, 
                      "Observer longitude": RP.eobs_ecl[0]/Consts.deg,
                      "Ring i_ecl": RP.i/Consts.deg,
                      "Nr": RP.Nrt,
                      "Np": RP.Np,
                      "Ring opacity": tau_ring,
                      "Ref plane": RP.reference_plane}
        save_dict = {"lambda": true_anomaly, "alpha": alpha, "Stot": Stot,
                     "Sp": Sp, "Sr": Sr, "Ptot": Ptot, "Pp": Pp, "Pr": Pr,
                     "Param": parameters}
        
        # Pickle the data, if file already exists it will be overwritten
        with open(save_location + f"data_with_oi_{orbit_i}_rl_{ring_l}_ri_{r_i}.pkl", "wb") as f:
            pickle.dump(save_dict, f)
            
def parameterSweep(fou_file_ring: str,
                   fou_file_planet: str,
                   orbit_i: float,
                   ring_i: float,
                   ring_l: float,
                   ring_ri: float,
                   ring_re: float,
                   name: str,
                   value,
                   tau_ring: float = 0.4,
                   interpr: str = "spline",
                   reference_plane: str = "Detector", # "Detector" or "Planetary"
                   Ns: int = 30,
                   Nb: int = 0,
                   Np: int = 10000,
                   Nr: int = 10000,
                   a: float = 1,
                   r_s: float = Consts.rsun,
                   r_p: float = Consts.rsaturn,
                   e: float = 0.0,
                   theta_end: float = 360,
                   n_theta: int = 361,
                   allow_non_uni: bool = False,
                   normalize: bool = True,
                   lambq_offset: float = 0.0,
                   limb_coeffs: list = [0.65]):
    
    # Announce start of function
    print (f"\n\n start run with: {name} = {value}") 
    
    # Generate save location if necessary
    if not os.path.isdir(f"/home/allard/Data/{name}_Series/{name}_{value}"):      
        os.makedirs(f"/home/allard/Data/{name}_Series/{name}_{value}")
    save_location = f"/home/allard/Data/{name}_Series/{name}_{value}/"

    # Save log file since no printing comes out of multiprocessing
    sys.stdout = open(save_location + "logfile.out", "w")
    
    # Make sure all output is printed
    Verbose.VERBOSITY=VERB_ALL
    
    # Calculate starting position of observer and star
    gamma, beta_obs, lamb_obs, lamb_star = Util.calcStartingPosition(orbit_i,ring_i,ring_l)
      
    # Initialise the system
    pixx_sys = System()
    s=pixx_sys.add(kind="Star",physics=dict(radius=r_s/pixx_sys.ul),optics=dict(limb_coeffs=limb_coeffs))
    p=pixx_sys.add(kind="Planet", primary=s,
                   radius=r_p/pixx_sys.ul,
                   orbit=dict(a=a, e=e),
                   physics=dict(radius=r_p/pixx_sys.ul),
                   optics=dict(nspangles=Np))
    r=pixx_sys.add(kind="Ring", primary=p,
                   physics=dict(fi=ring_ri, fe=ring_re, i=gamma),
                   optics=dict(nspangles=Nr))
    
    RP=pixx_sys.ensamble_system(extension=extension,
                                fname_planet=fou_file_planet,
                                fname_ring=fou_file_ring)
    RP.behavior["interp_method_ring"] = interpr
    RP.behavior["allow_non_uni"] = allow_non_uni
    RP.reference_plane = reference_plane
    
    lamb_initial = lamb_star
    lamb_final = lamb_initial + theta_end*Consts.deg
    lambs = np.linspace(lamb_initial,lamb_final,n_theta)
    
    # Determine starting position in eccentric orbit, default is apocenter
    if e >= 0.05:
        lq = np.linspace(0,2*np.pi,2000)
        d_max = 0
        lambq_max = 0
        for ii,l in enumerate(lq):
            RP.lambq = l
            RP.changeStellarPosition(lamb_initial)
            d = RP.rstar
            if d >= d_max:
                d_max = d
                lambq_max = l
        RP.lambq = lambq_max + lambq_offset*np.pi/180     
    
    # Initialise the starting position
    RP.changeObserver([lamb_obs,beta_obs])
    RP.changeStellarPosition(lamb_initial)
    RP._updateGeometricalFactors()
    RP._updateIncomingStellarFlux()
    RP._updateObservedFacetAreas()

    # Save images showing the starting position of planet, ring and star
    ecl_fig,obs_fig,star_fig = RP.plotRingedPlanet(showstar=True,showfig=False)
    ecl_fig.savefig(save_location + \
                    f"fig_with_{name}_{value}_and_oi_{orbit_i}_rl_{ring_l}_ri_{ring_i}_rin_{ring_ri}_rout_{ring_re}_ecl.png", dpi=300)
    obs_fig.savefig(save_location + \
                    f"fig_with_{name}_{value}_and_oi_{orbit_i}_rl_{ring_l}_ri_{ring_i}_rin_{ring_ri}_rout_{ring_re}_obs.png", dpi=300)
    star_fig.savefig(save_location + \
                     f"fig_with_{name}_{value}_and_oi_{orbit_i}_rl_{ring_l}_ri_{ring_i}_rin_{ring_ri}_rout_{ring_re}_star.png", dpi=300)
    plt.close()

    # Make lists
    Stot  = []
    Sp    = []
    Sr    = []
    Ptot  = []
    Pp    = []
    Pr    = []
    alpha = []
    vorbit= []
    rstar = []
    rstaro= []

    print("\n########################################")
    start_msg = f" Starting orbit simulation with: \n fou_file_planet = {RP.fname_planet} ,"+\
                f"\n fou_file_ring = {RP.fname_ring} ,"+\
                f"\n orbit inclination = {orbit_i} ,"+\
                f"\n ring roll = {ring_l} ,"+\
                f"\n ring inclination = {ring_i} ,"+\
                f"\n ring ri = {RP.fi} ,"+\
                f"\n ring re = {RP.fe} ,"+\
                f"\n observer i = {RP.eobs_ecl[1]/Consts.deg} ,"+\
                f"\n observer longitude = {RP.eobs_ecl[0]/Consts.deg} ,"+\
                f"\n ring inclination wrt ecl = {RP.i/Consts.deg} ,"+\
                f"\n number of ring spangles = {RP.Nrt} ,"+\
                f"\n number of planet spangles = {RP.Np} ,"+\
                f"\n ring opacity = {tau_ring} , "+\
                f"\n reference plane = {RP.reference_plane} "
    print(start_msg)
    print("########################################\n")
    
    # Start the orbit
    for lamb in lambs:
        RP.changeStellarPosition(lamb)
        print("True anomaly: ", (lamb-lamb_initial)/Consts.deg)
        RP._updateGeometricalFactors()
        RP._updateIncomingStellarFlux()
        RP._updateObservedFacetAreas()
        RP.updateReflection(taur=tau_ring,normalize=normalize)
        print("used ring opacity: ", RP.taur)
        
        # Save the relevant data
        Stot  += [RP.Stot]
        Sp    += [RP.Stotp]
        Sr    += [RP.Stotr]
        Ptot  += [RP.Ptot]
        Pp    += [RP.Ptotp]
        Pr    += [RP.Ptotr]
        alpha += [np.arccos(RP.alphaps)/Consts.deg]
        vorbit+= [RP.vorbit]
        rstar += [RP.rstar]
        rstaro+= [RP.rstar_obs]

    true_anomaly = list((lambs-lamb_initial)/Consts.deg)
    parameters = {"Fourier planet": RP.fname_planet, "Fourier ring": RP.fname_ring,
                  "Orbit inclinatie": orbit_i, "Ring roll": ring_l,
                  "Ring inclinatie": ring_i, "Ring ri": RP.fi, "Ring re": RP.fe,
                  "Observer inclinatie": RP.eobs_ecl[1]/Consts.deg, 
                  "Observer longitude": RP.eobs_ecl[0]/Consts.deg,
                  "Ring i_ecl": RP.i/Consts.deg,
                  "Nr": RP.Nrt,
                  "Np": RP.Np,
                  "Ring opacity": tau_ring,
                  "Ref plane": RP.reference_plane,
                  "Planet radius": RP.Rp}
    save_dict = {"lambda": true_anomaly, "alpha": alpha, "Stot": Stot,
                 "Sp": Sp, "Sr": Sr, "Ptot": Ptot, "Pp": Pp, "Pr": Pr,
                 "vorbit": vorbit, "rstar": rstar, "rstar_obs": rstaro,
                 "Param": parameters}

    # Pickle the data, if file already exists it will be overwritten
    with open(save_location + f"data_with_{name}_{value}.pkl", "wb") as f:
        pickle.dump(save_dict, f)
        
def opticalThicknessTest(fou_file_num,orbit_i_arr,ring_i,ring_l,reflection=False):
    #Global parameters
    Ns=30
    Nb=0
    Np=500
    Nr=2000
    
    print ('\n\n start file: ', fou_file_num)  
    
    gamma, beta_obs, lamb_obs, lamb_star = Util.calcStartingPosition(orbit_i_arr[0],ring_i,ring_l)
    fou_ring = "./fou_files/Ring/fou_ring_"+fou_file_num+"_0_8.dat"
    
    sys_test = System()
    s=sys_test.add(kind="Star",physics=dict(radius=Consts.rsun/sys_test.ul),optics=dict(limb_coeffs=[0.65]))
    p=sys_test.add(kind="Planet", primary=s,
                   radius=Consts.rsaturn/sys_test.ul,
                   orbit=dict(a=3, e=0.0),
                   physics=dict(radius=Consts.rsaturn/sys_test.ul),
                   optics=dict(nspangles=Np))
    r=sys_test.add(kind="Ring", primary=p,
                   physics=dict(fi=7, fe=9.25, i=gamma),
                   optics=dict(nspangles=Nr))
    RP_test=sys_test.ensamble_system(extension=extension,
                                     fname_planet=Misc.get_data("fou_lambert.dat"),
                                     fname_ring=fou_ring)
    RP_test.behavior["interp_method_ring"] = "spline"
    
    Rrs = np.zeros((len(orbit_i_arr),3))
    Prs = np.zeros(len(orbit_i_arr))
    illum_angle = np.zeros(len(orbit_i_arr))
    view_angle = np.zeros(len(orbit_i_arr))
    
    if not reflection:
        for jj,o_i in enumerate(orbit_i_arr):
            if jj > 0:
                gamma, beta_obs, lamb_obs, lamb_star = Util.calcStartingPosition(o_i,ring_i,ring_l)
                RP_test.i = gamma
                RP_test.changeObserver([lamb_obs,beta_obs])
                RP_test.Ns = Ns
                RP_test.Np = Np
                RP_test.Nr = Nr
                RP_test.Nb = Nb
                RP_test.updateProperties()

            RP_test.changeObserver([lamb_obs,beta_obs])
            RP_test.changeStellarPosition(lamb_star)
            RP_test.updateOpticalFactors()
            RP_test.updateReflection()

            if (np.inner(RP_test.nstar_equ,RP_test.nr_equ) < 0) ^ (np.inner(RP_test.nobs_equ,RP_test.nr_equ) < 0):
                Rrs[jj] = RP_test.Stotr
                Prs[jj] = RP_test.Ptotr

            illum_angle[jj] = RP_test.etars[0]
            view_angle[jj] = RP_test.zetars[0]
    elif reflection:
        for jj,o_i in enumerate(orbit_i_arr):
            if jj > 0:
                gamma, beta_obs, lamb_obs, lamb_star = Util.calcStartingPosition(o_i,ring_i,ring_l)
                RP_test.i = gamma
                RP_test.changeObserver([lamb_obs,beta_obs])
                RP_test.Ns = Ns
                RP_test.Np = Np
                RP_test.Nr = Nr
                RP_test.Nb = Nb
                RP_test.updateProperties()

            RP_test.changeObserver([lamb_obs,beta_obs])
            RP_test.changeStellarPosition(lamb_star+np.pi)
            RP_test.updateOpticalFactors()
            RP_test.updateReflection()

            if (np.inner(RP_test.nstar_equ,RP_test.nr_equ) < 0) ^ (np.inner(RP_test.nobs_equ,RP_test.nr_equ) < 0):
                Rrs[jj] = np.array([0,0,0])
                Prs[jj] = 0
            else:
                Stokesr = pixx.reflection(RP_test.Nrt, RP_test.phidiffrs, RP_test.betars,
                                          abs(RP_test.etars), abs(RP_test.zetars),
                                          RP_test.nmugsr,RP_test.nmatr,RP_test.nfour,RP_test.xmur,RP_test.rfour,
                                          np.ones(RP_test.Nrt)*RP_test.normr*RP_test.afr,
                                          RP_test.behavior["interp_method_ring"]
                                         )
                Sr = Stokesr[:,:-1]
                Stotr = np.sum(Sr,axis=0)/(np.pi*(RP_test.Rp**2)) 
                if abs(Stotr[0]) < 1e-6:
                    Ptotr = 0.0
                elif abs(Stotr[2]) < 1e-6:
                    Ptotr = -Stotr[1]/Stotr[0]
                else:
                    Ptotr = np.sqrt(Stotr[1]**2 + Stotr[2]**2)/Stotr[0]
                
                Rrs[jj] = Stotr
                Prs[jj] = Ptotr

            illum_angle[jj] = RP_test.etars[0]
            view_angle[jj] = RP_test.zetars[0]
    
    save_dict = {"Orbit_i": orbit_i_arr, "Illum": illum_angle, "Flux": Rrs, "Degree": Prs, "View": view_angle}
    if reflection:
        save_location = "/home/allard/Data/Optical_thickness_reflection"
        if not os.path.isdir(save_location):      
            os.makedirs(save_location)
        save_name = save_location + f"/opt_thickness_{fou_file_num}.pkl"
        with open(save_name, "wb") as f:
            pickle.dump(save_dict, f)
    else:
        with open(f"/home/allard/Data/Optical_thickness/opt_thickness_{fou_file_num}.pkl", "wb") as f:
            pickle.dump(save_dict, f)

def singleRun(fou_file_ring: str,
              fou_file_planet: str,
              orbit_i: float,
              ring_i: float,
              ring_l: float,
              ring_ri: float,
              ring_re: float,
              name: str,
              save: bool = False,
              tau_ring: float = 0.4,
              interpr: str = "spline",
              reference_plane: str = "Detector", # "Detector" or "Planetary"
              Ns: int = 30,
              Nb: int = 0,
              Np: int = 10000,
              Nr: int = 10000,
              a: float = 1,
              r_s: float = Consts.rsun,
              r_p: float = Consts.rsaturn,
              e: float = 0.0,
              theta_end: float = 360,
              theta_int: float = 0.0,
              n_theta: int = 361,
              allow_non_uni: bool = False,
              normalize: bool = True,
              lambq_offset: float = 0.0,
              limb_coeffs: list = [0.65],
              transit: bool = False):
    
    # Generate save location if necessary
    if not os.path.isdir(f"/home/allard/Data/Single_runs/{name}"):      
        os.makedirs(f"/home/allard/Data/Single_runs/{name}")
    save_location = f"/home/allard/Data/Single_runs/{name}/"

    # Calculate starting position of observer and star
    gamma, beta_obs, lamb_obs, lamb_star = Util.calcStartingPosition(orbit_i,ring_i,ring_l)
      
    # Initialise the system
    pixx_sys = System()
    s=pixx_sys.add(kind="Star",physics=dict(radius=r_s/pixx_sys.ul),optics=dict(limb_coeffs=limb_coeffs))
    p=pixx_sys.add(kind="Planet", primary=s,
                   radius=r_p/pixx_sys.ul,
                   orbit=dict(a=a, e=e),
                   physics=dict(radius=r_p/pixx_sys.ul),
                   optics=dict(nspangles=Np))
    r=pixx_sys.add(kind="Ring", primary=p,
                   physics=dict(fi=ring_ri, fe=ring_re, i=gamma),
                   optics=dict(nspangles=Nr))
    
    RP=pixx_sys.ensamble_system(extension=extension,
                                fname_planet=fou_file_planet,
                                fname_ring=fou_file_ring)
    RP.behavior["interp_method_ring"] = interpr
    RP.behavior["allow_non_uni"] = allow_non_uni
    RP.reference_plane = reference_plane
    
    if abs(theta_int) < 0.01:
        lamb_initial = lamb_star
        lamb_final = lamb_initial + theta_end*Consts.deg
        lambs = np.linspace(lamb_initial,lamb_final,n_theta)
    else:
        lamb_initial = lamb_star - theta_int*Consts.deg
        lamb_final = lamb_star + theta_end*Consts.deg
        lambs = np.linspace(lamb_initial,lamb_final,n_theta)
    
    # Determine starting position in eccentric orbit, default is apocenter
    if e >= 0.05:
        lq = np.linspace(0,2*np.pi,2000)
        d_max = 0
        lambq_max = 0
        for ii,l in enumerate(lq):
            RP.lambq = l
            RP.changeStellarPosition(lamb_initial)
            d = RP.rstar
            if d >= d_max:
                d_max = d
                lambq_max = l
        RP.lambq = lambq_max + lambq_offset*np.pi/180  
        
    # Initialise the starting position
    RP.changeObserver([lamb_obs,beta_obs])
    RP.changeStellarPosition(lamb_initial)
    RP._updateGeometricalFactors()
    RP._updateIncomingStellarFlux()
    RP._updateObservedFacetAreas()

    # Save images showing the starting position of planet, ring and star
    if save:
        ecl_fig,obs_fig,star_fig = RP.plotRingedPlanet(showstar=True,showfig=False)
        ecl_fig.savefig(save_location + \
                        f"fig_with_{name}_and_oi_{orbit_i}_rl_{ring_l}_ri_{ring_i}_rin_{ring_ri}_rout_{ring_re}_ecl.png", dpi=300)
        obs_fig.savefig(save_location + \
                        f"fig_with_{name}_and_oi_{orbit_i}_rl_{ring_l}_ri_{ring_i}_rin_{ring_ri}_rout_{ring_re}_obs.png", dpi=300)
        star_fig.savefig(save_location + \
                         f"fig_with_{name}_and_oi_{orbit_i}_rl_{ring_l}_ri_{ring_i}_rin_{ring_ri}_rout_{ring_re}_star.png", dpi=300)
        plt.close()

    # Make lists
    Stot  = []
    Sp    = []
    Sr    = []
    Ptot  = []
    Pp    = []
    Pr    = []
    alpha = []
    
    if transit:
        ts = []
        T  = []
        Tp = []
        Tr = []
    
    # Start the orbit
    for lamb in lambs:
        RP.changeStellarPosition(lamb)
        print("True anomaly: ", (lamb-lamb_initial)/Consts.deg)
        RP._updateGeometricalFactors()
        RP._updateIncomingStellarFlux()
        RP._updateObservedFacetAreas()
        RP.updateReflection(taur=tau_ring, normalize=normalize)
        
        if transit:
            RP.updateTransit()
            ts += [RP.t*RP.CU.UT]
            T  += [-RP.Tip.sum() - RP.Tir.sum() + RP.Stot[0]/1e6]
            Tp += [-RP.Tip.sum()]
            Tr += [-RP.Tir.sum()]
            
        # Save the relevant data
        Stot  += [RP.Stot]
        Sp    += [RP.Stotp]
        Sr    += [RP.Stotr]
        Ptot  += [RP.Ptot]
        Pp    += [RP.Ptotp]
        Pr    += [RP.Ptotr]
        alpha += [np.arccos(RP.alphaps)/Consts.deg]

    true_anomaly = list((lambs-lamb_initial)/Consts.deg)
    parameters = {"Fourier planet": RP.fname_planet, "Fourier ring": RP.fname_ring,
                  "Orbit inclinatie": orbit_i, "Ring roll": ring_l,
                  "Ring inclinatie": ring_i, "Ring ri": RP.fi, "Ring re": RP.fe,
                  "Observer inclinatie": RP.eobs_ecl[1]/Consts.deg, 
                  "Observer longitude": RP.eobs_ecl[0]/Consts.deg,
                  "Ring i_ecl": RP.i/Consts.deg,
                  "Nr": RP.Nrt,
                  "Np": RP.Np,
                  "Ring opacity": tau_ring,
                  "Ref plane": RP.reference_plane}
    if transit:
        ts = np.array(ts)
        ts = (ts-ts[0])/Consts.day
        save_dict = {"lambda": true_anomaly, "alpha": alpha, "Stot": Stot, "Sp": Sp, 
                     "Sr": Sr, "Ptot": Ptot, "Pp": Pp, "Pr": Pr, "Param": parameters, 
                     "Time": ts, "Ttot": T, "Tp": Tp, "Tr": Tr}
    else:
        save_dict = {"lambda": true_anomaly, "alpha": alpha, "Stot": Stot,
                     "Sp": Sp, "Sr": Sr, "Ptot": Ptot, "Pp": Pp, "Pr": Pr,
                     "Param": parameters}
    
    # Pickle the data, if file already exists it will be overwritten
    if save:
        with open(save_location + f"data_with_{name}.pkl", "wb") as f:
            pickle.dump(save_dict, f)
        return save_location + f"data_with_{name}.pkl"
    else:
        return "Nothing"
        
################################################################
##################### PLOT TOOLS ###############################
################################################################

def setup(ax,maticksizex,xlim0,xlim1,maticksizey,ylim0,ylim1,notext=False):  
    ax.xaxis.set_major_locator(ticker.MultipleLocator(maticksizex))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(maticksizey))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    if notext:
        ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', direction='in', length=10, width=1.00, right=True, top=True)
    ax.tick_params(which='minor', direction='in', length=5, width=0.75, right=True, top=True)
    ax.set_xlim([xlim0,xlim1])
    ax.set_ylim([ylim0,ylim1])
    ax.patch.set_alpha(0.0)

def add_subplot_axes(ax,rect,facecolor='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax
    
################################################################
################### TRANSIT CODE ###############################
################################################################

def transitFit(x,fitx,fity,run_num,save_everything=False):
    """
    x = [b,ring_i,ring_l,r,a,e,fi,fe,lincoef,quadcoef,optical_th]
    x[0] = b ; impact parameter
    x[1] = ring_i ; ring_inclination
    x[2] = ring_l ; ring roll
    x[3] = r ; planetary radius, in terms of the radius of jupiter
    x[4] = a ; semi-major axis of the orbit
    x[5] = e ; eccentricity of the orbit
    x[6] = fi ; inner radius of ring, in terms of planetary radius
    x[7] = fe ; outer radius of ring, in terms of planetary radius
    x[8] = lincoef ; linear limb-darkening coefficient
    x[9] = quadcoef ; quadratic limb-darkening coefficient
    x[10] = optical_th ; optical thickness of the ring
    
    fitx = x-array of observations
    fity = y-array of observations
    run_num = run number, manually added up
    """
    optical_thickness_values = np.array([0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.4,0.6,0.8,
                                         1.0,1.2,1.4,1.6,1.8,2.0,4.0,6.0,8.0,10.0,20.0])
    optical_thickness_names = np.array(["0_001","0_002","0_005","0_01","0_02","0_05","0_1","0_2","0_4","0_6","0_8",
                                        "1_0","1_2","1_4","1_6","1_8","2_0","4_0","6_0","8_0","10_0","20_0"])
    idx = (np.abs(optical_thickness_values - 10**x[10])).argmin()
    fname_ring = f"./fou_files/Ring/fou_ring_{optical_thickness_names[idx]}_0_3.dat"
    optical_thickness = optical_thickness_values[idx]
    
    pixx_sys = System()
    
    aR = x[4]*(1-x[5])*pixx_sys.ul/(0.75*Consts.rsun) # In terms of stellar radius
    orbit_i = np.arccos(x[0]/aR)/Consts.deg

    gamma, beta_obs, lamb_obs, lamb_star = Util.calcStartingPosition(orbit_i,x[1],x[2])

    s=pixx_sys.add(kind="Star",physics=dict(radius=0.75*Consts.rsun/pixx_sys.ul),optics=dict(limb_coeffs=[x[8],x[9]]))
    p=pixx_sys.add(kind="Planet", primary=s, 
                   radius=x[3]*Consts.rjupiter/pixx_sys.ul,
                   orbit=dict(a=x[4], e=x[5]),
                   physics=dict(radius=x[3]*Consts.rjupiter/pixx_sys.ul),
                   optics=dict(nspangles=5000))
    r=pixx_sys.add(kind="Ring", primary=p,
                   physics=dict(fi=x[6], fe=x[7], i=gamma),
                   optics=dict(nspangles=5000))
    
    RP=pixx_sys.ensamble_system(extension=extension, 
                                fname_planet="./fou_files/Planet/fou_bmsca60.0_asurf0.5.dat",
                                fname_ring=fname_ring)
    RP.behavior["interp_method_ring"] = "spline"
    thetas=RP.thetas

    lamb_initial=lamb_star - thetas - 4*RP.fe*RP.thetap
    lamb_final=lamb_star + thetas + 4*RP.fe*RP.thetap
    lambs = np.linspace(lamb_initial,lamb_final,200)
    
    lq = np.linspace(0,2*np.pi,200)
    d_max = 0
    lambq_max = 0
    for ii,l in enumerate(lq):
        RP.lambq = l
        RP.changeStellarPosition(lamb_initial)
        d = RP.rstar
        if d >= d_max:
            d_max = d
            lambq_max = l
            
    RP.lambq = lambq_max
    
    # Initialise the starting position
    RP.changeObserver([lamb_obs,beta_obs])
    RP.changeStellarPosition(lamb_initial)
    RP._updateGeometricalFactors()
    RP._updateIncomingStellarFlux()
    RP._updateObservedFacetAreas()

    T     = []
    ts    = []
    if save_everything:
        Stot  = []
        Sp    = []
        Sr    = []
        Ptot  = []
        Pp    = []
        Pr    = []
        Tp    = []
        Tr    = []
        alpha = []

    for lamb in lambs:
        RP.changeStellarPosition(lamb)
        RP._updateGeometricalFactors()
        RP._updateIncomingStellarFlux()
        RP._updateObservedFacetAreas()
        RP.updateReflection(taur=optical_thickness,normalize=False)
        ts    += [RP.t*RP.CU.UT]
        RP.updateTransit()
        T  += [-RP.Tip.sum() - RP.Tir.sum() + RP.Stot[0]/1e6]
        if save_everything:
            Stot  += [RP.Stot]
            Sp    += [RP.Stotp]
            Sr    += [RP.Stotr]
            Ptot  += [RP.Ptot]
            Pp    += [RP.Ptotp]
            Pr    += [RP.Ptotr]
            alpha += [np.arccos(RP.alphaps)/Consts.deg]
            Tp += [-RP.Tip.sum()]
            Tr += [-RP.Tir.sum()] 
        
    ts = np.array(ts)
    T = np.array(T)
    
    ts=(ts-ts[0])/Consts.day
    
    # Fitness test:
    def func(x,fitx,ts,T,fity):
        return np.sqrt(np.sum((np.interp(fitx,ts+x,T)-fity)**2)/len(fity))
    
    res = minimize(func, 0.38, args = (fitx,ts,T+1,fity))
    
    if save_everything:
        true_anomaly = list((lambs-lamb_star)/Consts.deg)    
        save_dict = {"lambda": true_anomaly, "alpha": alpha, "Stot": Stot,
                     "Sp": Sp, "Sr": Sr, "Ptot": Ptot, "Pp": Pp, "Pr": Pr,
                     "Ttot": T, "Tp": Tp, "Tr": Tr, "Time": ts, "x": x, 
                     "Fitness": res.fun, "Fitx": fitx, "Fity": fity,
                     "Shift": res.x}
        #Pickle the data, if file already exists it will be overwritten
        if not os.path.isdir(f"/home/allard/Data/Transit_fit_checked/run_num{run_num}"):      
            os.makedirs(f"/home/allard/Data/Transit_fit_checked/run_num{run_num}")
        save_location = f"/home/allard/Data/Transit_fit_checked/run_num{run_num}/"

        save_name = save_location + \
                    f"rn{run_num}_fitness%.6f_x1_%.1f_.pkl" % (res.fun,x[1])
        with open(save_name, "wb") as f:
            pickle.dump(save_dict, f)
    else:
        save_dict = {"x": x, "Ttot": T, "Time": ts, "Fitness": res.fun,
                     "Fitx": fitx, "Fity": fity, "Shift": res.x}
        #Pickle the data, if file already exists it will be overwritten
        if not os.path.isdir(f"/home/allard/Data/Transit_fit/run_num{run_num}"):      
            os.makedirs(f"/home/allard/Data/Transit_fit/run_num{run_num}")
        save_location = f"/home/allard/Data/Transit_fit/run_num{run_num}/"

        save_name = save_location + \
                    f"rn{run_num}_fitness%.8f_x1_%.1f_x2_%.2f_fi%.2f_fe%.2f_e%.3f_.pkl" % (res.fun,x[1],x[2],x[6],x[7],x[5])
        with open(save_name, "wb") as f:
            pickle.dump(save_dict, f)
    
    return res.fun
    
def transitFitMie(x,fitx,fity,run_num,save_everything=False,allow_non_uni=True):
    """
    x = [b,ring_i,ring_l,r,a,e,fi,fe,lincoef,quadcoef,optical_th]
    x[0] = b ; impact parameter
    x[1] = ring_i ; ring_inclination
    x[2] = ring_l ; ring roll
    x[3] = r ; planetary radius, in terms of the radius of jupiter
    x[4] = a ; semi-major axis of the orbit
    x[5] = e ; eccentricity of the orbit
    x[6] = fi ; inner radius of ring, in terms of planetary radius
    x[7] = fe ; outer radius of ring, in terms of planetary radius
    x[8] = lincoef ; linear limb-darkening coefficient
    x[9] = quadcoef ; quadratic limb-darkening coefficient
    x[10] = optical_th ; optical thickness of the ring
    x[11] = particle_r ; radius of the particles
    
    fitx = x-array of observations
    fity = y-array of observations
    run_num = run number, manually added up
    """
    optical_thickness_values = np.array([0.0,0.002,0.01,0.05,0.1,0.2,0.6,
                                         1.0,1.2,1.6,2.0,4.0,8.0])
    optical_thickness_names = np.array(["0_0","0_002","0_01","0_05","0_1","0_2","0_6",
                                        "1_0","1_2","1_6","2_0","4_0","8_0"])
    particle_r_names = np.array(["020","040","080"])
    if int(x[10]) != 0:
        if int(x[11]) == 2:
            fname_ring = f"./fou_files/Ring/fou_ring_{optical_thickness_names[int(x[10])]}_0_3.dat"
        else:
            fname_ring = "./fou_files/Ring/Mie/fou_file_mie_1.5" + \
                         f"_{particle_r_names[int(x[11])]}_4500_{optical_thickness_values[int(x[10])]}_0.3_60gaus.dat"

        optical_thickness = optical_thickness_values[int(x[10])]
    elif int(x[10]) == 0:
        fname_ring = f"./fou_files/Ring/fou_ring_0_001_0_3.dat"
        optical_thickness = 0.0
    
    pixx_sys = System()
    
    aR = x[4]*(1-x[5])*pixx_sys.ul/(0.75*Consts.rsun) # In terms of stellar radius
    orbit_i = np.arccos(x[0]/aR)/Consts.deg

    gamma, beta_obs, lamb_obs, lamb_star = Util.calcStartingPosition(orbit_i,x[1],x[2])

    s=pixx_sys.add(kind="Star",physics=dict(radius=0.75*Consts.rsun/pixx_sys.ul),optics=dict(limb_coeffs=[x[8],x[9]]))
    p=pixx_sys.add(kind="Planet", primary=s, 
                   radius=x[3]*Consts.rjupiter/pixx_sys.ul,
                   orbit=dict(a=x[4], e=x[5]),
                   physics=dict(radius=x[3]*Consts.rjupiter/pixx_sys.ul),
                   optics=dict(nspangles=2000))
    r=pixx_sys.add(kind="Ring", primary=p,
                   physics=dict(fi=x[6], fe=x[7], i=gamma),
                   optics=dict(nspangles=2000))
    
    RP=pixx_sys.ensamble_system(extension=extension, 
                                fname_planet="./fou_files/Planet/fou_bmsca60.0_asurf0.5.dat",
                                fname_ring=fname_ring)
    RP.behavior["allow_non_uni"] = allow_non_uni
    thetas=RP.thetas

    lamb_initial=lamb_star - thetas - 4*RP.fe*RP.thetap
    lamb_final=lamb_star + thetas + 4*RP.fe*RP.thetap
    if save_everything:
        lambs = np.linspace(lamb_initial,lamb_final,400)
    else:
        lambs = np.linspace(lamb_initial,lamb_final,200)
    
    lq = np.linspace(0,2*np.pi,720)
    d_max = 0
    lambq_max = 0
    for ii,l in enumerate(lq):
        RP.lambq = l
        RP.changeStellarPosition(lamb_initial)
        d = RP.rstar
        if d >= d_max:
            d_max = d
            lambq_max = l
            
    RP.lambq = lambq_max
    
    # Initialise the starting position
    RP.changeObserver([lamb_obs,beta_obs])
    RP.changeStellarPosition(lamb_initial)
    RP._updateGeometricalFactors()
    RP._updateIncomingStellarFlux()
    RP._updateObservedFacetAreas()

    T     = []
    ts    = []
    if save_everything:
        Stot  = []
        Sp    = []
        Sr    = []
        Ptot  = []
        Pp    = []
        Pr    = []
        Tp    = []
        Tr    = []
        alpha = []
        tr    = []
        betas = []

    for lamb in lambs:
        RP.changeStellarPosition(lamb)
        RP._updateGeometricalFactors()
        RP._updateIncomingStellarFlux()
        RP._updateObservedFacetAreas()
        RP.updateReflection(taur=optical_thickness,normalize=False)
        ts += [RP.t*RP.CU.UT]
        RP.updateTransit()
        if int(x[10]) == 0:
            T  += [-RP.Tip.sum() + RP.Stotp[0]/1e6]
        else:
            T  += [-RP.Tip.sum() - RP.Tir.sum() + RP.Stot[0]/1e6]
            
        if save_everything:
            Stot  += [RP.Stot]
            Sp    += [RP.Stotp]
            Sr    += [RP.Stotr]
            Ptot  += [RP.Ptot]
            Pp    += [RP.Ptotp]
            Pr    += [RP.Ptotr]
            alpha += [np.arccos(RP.alphaps)/Consts.deg]
            Tp    += [-RP.Tip.sum()]
            Tr    += [-RP.Tir.sum()] 
            tr    += [RP.tr]
            betas += [RP.betas]
        
    ts = np.array(ts)
    T = np.array(T)
    
    ts=(ts-ts[0])/Consts.day
    
    # Fitness test:
    def func(x,fitx,ts,T,fity):
        return np.sqrt(np.sum((np.interp(fitx,ts+x,T)-fity)**2)/len(fity))
    
    res = minimize(func, 0.38, args = (fitx,ts,T+1,fity))
    
    if save_everything:
        true_anomaly = list((lambs-lamb_star)/Consts.deg)    
        save_dict = {"lambda": true_anomaly, "alpha": alpha, "Stot": Stot,
                     "Sp": Sp, "Sr": Sr, "Ptot": Ptot, "Pp": Pp, "Pr": Pr,
                     "Ttot": T, "Tp": Tp, "Tr": Tr, "Time": ts, "x": x, 
                     "Fitness": res.fun, "Fitx": fitx, "Fity": fity,
                     "Shift": res.x, "tr cond": tr, "betas": betas, 
                     "Foufile ring": fname_ring}
        #Pickle the data, if file already exists it will be overwritten
        if not os.path.isdir(f"/home/allard/Data/Transit_fit_checked/run_num{run_num}"):      
            os.makedirs(f"/home/allard/Data/Transit_fit_checked/run_num{run_num}")
        save_location = f"/home/allard/Data/Transit_fit_checked/run_num{run_num}/"

        save_name = save_location + \
                    f"rn{run_num}_fitness%.8f_x1_%.1f_.pkl" % (res.fun,x[1])
        with open(save_name, "wb") as f:
            pickle.dump(save_dict, f)
    else:
        save_dict = {"x": x, "Ttot": T, "Time": ts, "Fitness": res.fun,
                     "Fitx": fitx, "Fity": fity, "Shift": res.x}
        #Pickle the data, if file already exists it will be overwritten
        if not os.path.isdir(f"/home/allard/Data/Transit_fit/run_num{run_num}"):      
            os.makedirs(f"/home/allard/Data/Transit_fit/run_num{run_num}")
        save_location = f"/home/allard/Data/Transit_fit/run_num{run_num}/"

        save_name = save_location + \
                    f"rn{run_num}_fitness%.8f_x1_%.1f_x2_%.2f_fi%.2f_fe%.2f_e%.3f_x10_%.1f.pkl" % (res.fun,x[1],x[2],x[6],x[7],x[5],x[10])
        with open(save_name, "wb") as f:
            pickle.dump(save_dict, f)
    
    return res.fun
        
        
        
        
        
        
        