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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# External required packages
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from pryngles import *

import spiceypy as spy
import numpy as np
import rebound as rb
from tqdm import tqdm


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Class System
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class System(PrynglesCommon):
    """
    High-level interface for building spangled planetary systems.
    Integrates Rebound simulations with the Pryngles spangling pipeline to generate discrete surface facets,
    manage bodies, compute photometric states, and evolve dynamics.

    Parameters
    ----------
    filename : `str`
        Path to a saved system file to load. If provided, units and resetable
        parameters are ignored | **Default:** None
    units : `list` 
        Three-element list of str specifying units for length, mass, and time
        as recognized by Rebound (order: length, mass, time) | **Default:** ['au', 'msun', 'yr2pi']
    resetable : `bool`
        Whether the system state can be snapshotted and reset to initial | **Default:** False

    Attributes
    ----------
    sim : `rebound.Simulation`
        Underlying Rebound simulation instance (None until initialized).
    ul, um, ut : `float`
        Conversion factors from internal units to SI (meters, kilograms, seconds).
    G : `float`
        Gravitational constant in chosen units.
    bodies : `odict`
        Ordered dictionary of :any:`body.Body` instances in the system, keyed by name.
    nbodies : `int`
        Number of bodies in `bodies`.
    nparticles : `int`
        Number of particles in the Rebound simulation.
    n_obs : `np.array(3)`
        Normalized observer direction vector :math:`\hat{n}` in the system's coordinate frame.
    sg : :any:`spangler.Spangler`
        Spangler instance covering all system facets.
    spangle_scatterers : `dict`
        Mapping from :any:`spangler.Spangler` type attribute to :any:`scatterer.Scatterer` classes and options.
        This means that for spangles of the type :any:`consts.SPANGLE_ATMOSPHERIC` Pryngles will 
        instantiate an object of the class :any:`scatterer.LambertianGrayAtmosphere` to compute their scattering.

    Examples
    --------
    >>> # Create a new system with default units
    >>> sys = System(units=['au','msun','yr2pi'])
    >>> 
    >>> # Now you can add bodies to the system
    >>> star = sys.add('Star', m=1.0)
    >>> planet = sys.add('Planet', parent=star, m=1e-3, a=1.0, e=0.1)
    >>>
    >>> # Initialize the simulation
    >>> sys.initialize_simulation()
    >>>
    >>> # Spangle the bodies to compute photometry
    >>> sys.spangle_system()
    >>> 
    >>> # You can visualize the system
    >>> sys.sg.plot2d(include = ['Star', 'Planet'])

    .. image:: images/system_init.png
        :align: center
    """

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Bassic methods
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def __init__(self,
                 filename=None,
                 units=['au','msun','yr2pi'],
                 resetable=False
                ):
        
        if filename:
            self.load_from(filename)
            return
        
        #Rebound simulation
        self.sim=None
        self._simulated=False
        
        #Attributes by default

        # For Polarization purposes
        self.extension = 'cpixx'
        
        #List of bodies in the system
        self.bodies=odict()
        
        #Root of the tree of bodies
        self.root=None
        
        #Center of the light-source in the system
        self.source=None
        self.center_root=np.array([0,0,0])
        
        #Orbital configuration
        self.orbital_configuration=None
        
        #Observer properties
        self.n_obs=np.array([0,0,1])
        self.alpha_obs=0  
        self.center_obs=None
        
        #Check if spangled
        self._spangled=False
        
        #Check if observer has been set
        self._observer_set=False
        self._luz_set=False
        
        #Initialize spangler object
        self.sg=None
        
        #Is the system resetable?
        self._resetable=resetable
        if self._resetable:
            #Create temporary file
            self._snap_file_name = "/tmp/pryngles-system.pkl"
        
        #Update rebound units
        self.update_units(units)
        
        #By default spangle scatterers
        """
        This is the list of the class of scatterers used to calculate the scattering in
        different types of spangles.
        
        The structure of this dictionary is:
        
            key: integer (or enumerator):
                This is the column spangle_type in the spangler DataFrame.
                
            item: tuple (2):
                Component 1: 
                    Class of scatterer.
                Component 2: 
                    Dictionary mapping the initialization properties of the scatterer to columns
                    in the spangler DataFrame.
        
        Example of item:
        
            SPANGLE_ATMOSPHERIC:(LambertianGrayAtmosphere,dict(AS="albedo_gray_spherical"))
            
                This means that for spangles of the type SPANGLE_ATMOSPHERIC Pryngles will 
                instantiate an object of the class LambertianGrayAtmosphere.  This class have a
                single parameter, the spherical albedo AS.  The dictionary means that when 
                instantiating the object the column "albedo_gray_spherical" will be used to 
                initialize the object.                
        """
        self.spangle_scatterers={
            SPANGLE_ATMOSPHERIC:(LambertianGrayAtmosphere,dict(AS="albedo_gray_spherical")),
            SPANGLE_GRANULAR:(LambertianGraySurface,dict(AL="albedo_gray_normal")),
            SPANGLE_LIQUID:(LambertianGraySurface,dict(AL="albedo_gray_normal")),
            SPANGLE_SOLID_ICE:(LambertianGraySurface,dict(AL="albedo_gray_normal")),
            SPANGLE_SOLID_ROCK:(LambertianGraySurface,dict(AL="albedo_gray_normal")),
            SPANGLE_GASEOUS:(BlackBodySurface,dict()),
            SPANGLE_STELLAR:(BlackBodySurface,dict()),
        }
        
    def update_units(self,units):
        """
        Set or change the internal unit conversion factors.

        Parameters
        ----------
        units :  `list`
            Three-element string of Rebound-recognized units: [length, mass, time]. 
            You can see the defined ones in :any:`consts.REBOUND_ORBITAL_PROPERTIES`

        Raises
        ------
        ValueError
            If any unit string is not recognized by Rebound.
        """
        #Check units
        if units[0] not in rb.units.lengths_SI:
            raise ValueError(f"Length unit provided '{units[0]}' is not recognized by Rebound.  Use one of these: {tuple(rb.units.lengths_SI.keys())}")
        if units[1] not in rb.units.masses_SI:
            raise ValueError(f"Mass unit provided '{units[1]}' is not recognized by Rebound.  Use one of these: {tuple(rb.units.masses_SI.keys())}")
        if units[2] not in rb.units.times_SI:
            raise ValueError(f"Time unit provided '{units[2]}' is not recognized by Rebound.  Use one of these: {tuple(rb.units.times_SI.keys())}")
        
        #Units        
        self.units=units
        self._ul,self._um,self._ut=self.units
        #self.sim.units=self.units
        
        #Canonical units of the system
        self.ul=rb.units.convert_length(1,self._ul,"m")
        self.um=rb.units.convert_mass(1,self._um,"kg")

        #Compute the units of time
        sim=rb.Simulation()
        sim.units=self.units
        self.G=sim.G
        self.ut=np.sqrt(self.G*self.ul**3/(self.um*GSI))
        
        #Update system
        self._update_system()
        
    def _get_source(self,body):
        """Get the source of light (stellar body) in the center of a body
        """
        if (body.parent is None) or (body.kind == "Star"):
            return body

        elif body.parent.kind == "Star":
            return body.parent

        else:
            return self._get_source(body.parent)

    def _update_system(self):
        """Update system properties
        """
        self.nbodies=len(self.bodies)
        if self._simulated:
            self.nparticles=len(self.sim.particles)
        
    def _is_spangled(self):
        """Check if system is spangled
        """
        return True if self.sg else False
    
    def reset_state(self):
        """
        Restore Spangler visibility and illumination to initial.
        Calls :meth:`Spangler.reset_state` and clears observer/light flags.
        """
        self.sg.reset_state()
        self._observer_set=False
        self._luz_set=False

    def save_to(self,filename):
        """Save system from file
        
        Parameters
        ----------
        filename : str
            Base filename (without extension) for saving.

        Returns
        --------------
        :
            filename : str
             Rebound state is saved as filename + '.rbin'.
        """
        if self._simulated:
            #Rebound file
            rb_filename=filename+".rbin"

            #Save rebound state
            verbose(VERB_SIMPLE,"Saving rebound simulation")
            self.sim.save_to_file(rb_filename)

            #Since rebound have ctypes it cannot be pickled
            del self.sim
            self._simulated=True

        #Pickle system
        PrynglesCommon.save_to(self,filename)

        if self._simulated:
            #Load again rebound
            self.sim=rb.Simulation(rb_filename)

    def load_from(self,filename):
        """Load system from filename
                
        Parameters
        ----------
        filename : str
            Base filename (without extension) used for pickle and Rebound state.

        Returns
        ------------
        :
            sim : ``rebound.Simulation``
             :data:`~ system.System.sim` attribute loaded from file
        """
        #Load system
        self=PrynglesCommon.load_from(self,filename)

        if self._simulated:
            #Rebound file
            rb_filename=filename+".rbin"

            #Load rebound
            verbose(VERB_SIMPLE,"Loading rebound simulation")
            self.sim=rb.Simulation(rb_filename)

    def _read_Fourier_data(self):

        if self.extension not in ['pixx','cpixx']:
            raise ValueError(f"The extension '{self.extension}' is not recognized (available 'pixx', 'cpixx')")

        fname_planet = Misc.get_data("fou_gasplanet_optical_50.dat")
        fname_ring = Misc.get_data("fou_ring_0_4_0_8.dat")

        self.SCp = StokesScatterer(fname_planet)
        self.nmatp = self.SCp.nmat

        self.SCr = StokesScatterer(fname_ring)
        self.nmatr = self.SCr.nmat
        
    def status(self):
        """
        Print a summary of the current system and particles.

        Note
        ---------
        If not yet initialized, prompts for initialization.
        """
        if self._simulated:
            print(f"System with {self.nbodies} bodies and {self.nparticles} particles (rings and disk are not particles)")
            self.sim.status()
        else:
            print(f"Simulation for this system has not been yet initialized. Use System.initialize_simulation()")


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Tested methods from module file scatterer
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def _update_scatterers(self):
        """
        Instantiate scatterer objects for all spangles lacking one.

        Raises
        ------
        AssertionError
            If spangles have not yet been generated via :data:`~ system.System.spangle_system()`.
        """
        if not self._spangled:
            raise AssertionError("You need to spangle the system before updating the scatterers.")
        
        # This column stores python objects; keep as object dtype.
        if "scatterer" in self.data.columns and self.data["scatterer"].dtype != object:
            self.data["scatterer"] = self.data["scatterer"].astype(object)

        #Update scatterer only for the non-assigned one
        cond = (self.data.scatterer == "")
        mask = self.data.loc[cond]
        for spangle_type, group in mask.groupby('spangle_type'):
            # Get scatterer class and options description
            spangle_scatterer, spangle_options = self.spangle_scatterers[spangle_type]
            # Build options of scatterers from options description
            options = {key: group[value].iloc[0] for key, value in spangle_options.items()}
            # Instantiate object of scatterer and save hash into DataFrame data object
            self.data.loc[group.index, 'scatterer'] = pd.Series([spangle_scatterer(**options)]*len(group), dtype=object, index=group.index)

    def _update_albedos(self):
        """ 
        Compute directional-dependent Lambertian albedo per spangle. 
        It implements :data:`~ scatterer.Scatterer.get_albedo()` method. See our :doc:`scatterer` for the theory behind it.

        Note
        ------
        Applies only to non-stellar spangles, i.e., with no :data:`~ consts.SPANGLE_STELLAR` attribute
        """

        # Creating lambertian_albedo column for directional-dependent albedo for surface/atmosphere scattering
        self.data['lambertian_albedo'] = 0.0

        # Only planetary surfaces are taken into account
        cond = (self.data['spangle_type'] != 6) & (self.data['cos_luz'] >= 0)
    
        # Initialize Lambertian Albedo columns into de DataFrame
        def _as_float(x):
            # Some scatterers may return numpy scalar/0-d arrays; normalize to Python float for pandas.
            return float(np.asarray(x).reshape(-1)[0])

        self.data.loc[cond, "lambertian_albedo"] = self.data.loc[cond].apply(
            lambda sp: _as_float(
                sp["scatterer"].get_albedo(eta=sp["cos_luz"], zeta=0, delta=0, lamb=0.55)
            ),
            axis=1,
        )
        
    def _update_optical_depth(self):
        """  
        Update the optical depth of all bodies in the system by the user input.
        """

        for name, body in self.bodies.items():

            cond_body = (self.data['name'] == name)
            self.data.loc[cond_body, 'tau_gray_optical'] = body.tau_gray_optical

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Tested methods from module file system
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def add(self,kind="Star",parent=None,**props):
        """
        Add a new Body to the System and auto-spangle it.

        Parameters
        ----------------
        kind : `str`
            Type of body to add from :any:`consts.BODY_KINDS` | **Defaults** = `Star`
        parent : Body or None
            Parent body (orbital central object); None for the root star.
        **props : dict
            Properties forwarded to the Body constructor (mass, radius, orbital elements). It came from :any:`consts.BODY_DEFAULTS`

        Returns
        -------
        :
            output : :any:`body.Body`
             The newly created Body instance.

        Raises
        ------
        AssertionError
            If kind is None.
        ValueError
            If kind not in :any:`consts.BODY_KINDS` or name already exists.    
        """
        if kind is None:
            raise AssertionError("You must provide a valid object kind (Star, Planet, Ring).")
    
        if kind not in BODY_KINDS:
            raise ValueError(f"Object kind '{kind}' is not recognized.")
    
        #Legacy
        if 'primary' in props:
            parent=props["primary"]
        if kind=="Observer":
            parent=self.root
        
        #Default parameters
        if self.root:
            if (kind!="Star") and (parent is None):
                parent=self.root
                
            if kind=="Planet":
                if "m" not in props:
                    props["m"]=0.1*parent.m
                if "radius" not in props:
                    props["radius"]=0.5*parent.radius
                if "a" not in props:
                    if "a" in parent.__dict__:
                        props["a"]=0.5*parent.a
                    
        #Create body
        props.update(dict(name_by_kind=True))
        self.__body=eval(f"{kind}(parent=parent,**props)")
        
        if self.__body.name in self.bodies:
            raise ValueError(f"An object with name '{self.__body.name}' has been already added.")
        
        #If we have a root object and no parent has been provided
        if not parent:
            if self.root:
                raise ValueError(f"A root object alread exist in the system ({self.root.name}) and you do not provided a parent body for {self.__body.name}.")
            else:
                self.root=self.__body
                verbose(VERB_SIMPLE,f"Setting the root object as {self.root.name}")
            
        self.bodies[self.__body.name]=self.__body
        
        #Create the shined body tree
        self.__body.shined=[]

        # Set units
        self.__body.units = self.units
        self.__body.ul = self.ul
        self.__body.um = self.um
        self.__body.ut = self.ut
    
        #Update system
        self._update_system()
        
        #Set the source of the object
        if self.__body.source:
            #Check that the source is a body
            if not isinstance(self.__body.source,Body):
                raise ValueError(f"The source of body must be an actual Body.")        
            #Check that the source is among the bodies
            if self.__body.source.name not in self.bodies:
                raise ValueError(f"The source of body {self.__body.name} is not among system bodies {list(self.bodies.keys())}.")
            #Check that the source be a star
            if self.__body.source.kind!="Star":
                raise ValueError(f"The source of body {self.__body.name} must be a Star.  You set {self.__body.source.name} which is a {self.__body.source.kind}.")
            self.__body.source.shined+=[self.__body.name]
        else:
            if self.__body.kind=="Star":
                self.__body.source=self.__body
            elif self.__body.parent==self.root:
                self.__body.source=self.root
            else:
                self.__body.source=self.__body.parent.source
        self.__body.source.shined+=[self.__body.name]
        
        verbose(VERB_SIMPLE,f"Object '{kind}' with name '{self.__body.name}' has been added.")

        return self.__body
    
    
    def initialize_simulation(self,orbital_tree=None,**rebound_options):
        """
        Build and initialize the Rebound simulation from the orbital tree scheme.

        Note
        ----------
        See the :doc:`orbit` for the convenction to construc an orbital tree input

        Parameters
        ----------
        orbital_tree : `list`
            Hierarchical tree of bodies for N-body initialization; see :any:`orbit.OrbitUtil.build_tree`
        
        Returns
        -------
        :
            output : :any:`orbit.Orbit`
             The Orbit object representing the hierarchical system.

        Raises
        ------
        AssertionError
            If a Body is in System but not in orbital tree.
        """

        # See Orbit Module for build_tree() method
        if orbital_tree is None: self.orbital_tree = OrbitUtil.build_tree(self.root)

        else: self.orbital_tree = orbital_tree
        
        #Set the rebound hash of all bodies
        for name,body in self.bodies.items():
            if body.kind=="Ring":
                body.rbhash=body.parent.name
            else:
                body.rbhash=body.name
        
        #Check that all bodies in system is in the orbital tree
        bodies=list(Misc.flatten(self.orbital_tree))
        for name,body in self.bodies.items():
            if body.kind=="Ring":
                continue
            if body not in bodies:
                raise AssertionError(f"Body '{name}' is in System but not in orbital tree.")
            
        #Build hierarchical N-body system
        orbit,pelements=OrbitUtil.build_system(self.orbital_tree,self.units)
        orbit.calculate_orbit()
        
        #Initialize simulation
        self.sim=rb.Simulation()
        self.sim.units=self.units
        
        #Add particles to simulation
        orbit.sim.move_to_com()
        for i,p in enumerate(orbit.sim.particles):
            self.sim.add(
                hash=bodies[i].name,
                m=bodies[i].m,
                x=p.x,y=p.y,z=p.z,
                vx=p.vx,vy=p.vy,vz=p.vz
            )
        self.orbit=orbit
        self._simulated=True
        self._update_system()
        
        return orbit
    
    def remove(self,name):
        """
        Remove a body (and children objects) from the system.

        Parameters
        ----------
        name : str
            Name (hash) of the body to remove.

        Raises
        ------
        ValueError
            If no body with the given name exists.
        """
        
        if name in self.bodies:
            verbose(VERB_SIMPLE,f"Removing object {name} from system")
    
            obj=self.bodies[name]
    
            #Get the list of child hashes before removing (it changes during for)
            child_hashes=list(obj.childs.keys())
            
            #Remove child objects
            for child_hash in child_hashes:
                if child_hash in self.bodies:
                    self.remove(child_hash)
                    
            #Remove object from Rebound simulation
            if obj.kind != "Ring":
                if self._simulated:
                    if self.nparticles:
                        verbose(VERB_SIMPLE,f"Removing particle {name} from simulation")
                        self.sim.remove(hash=name)
            
            #Remove object from childs of its parent
            if obj.parent:
                del obj.parent.childs[name]
            
            #Remove object from bodies
            del self.bodies[name]
    
            #Update system
            self._update_system()
        else:
            raise ValueError(f"No object with hash '{name}' in the system")
    
    def spangle_system(self):
        """
        Generate :any:`spangler.Spangler` instances for all bodies and join them.

        Attributes
        ------------
        sg : :any:`spangler.Spangler`
            It contains the :data:`~ spangler.Spangler` object in wich we sample and discretize the surface of all bodies defined in system in order to compute light-matter interactions
        data : `pd.DataFrame`
            Contains :any:`consts.SPANGLER_COLUMNS` and sets the  default observer/light states.

        Raises
        ------
        AssertionError
            If simulation not yet initialized.
        """
        if not self._simulated:
            raise AssertionError("Before spangling the system you must initialize the simulation: System.initialize_simulation().")

        self._spanglers=dict()
        
        #Add spangles
        for name,body in self.bodies.items() :
            
            #Center object around its position according to rebound
            body.center_ecl=np.array(self.sim.particles[body.rbhash].xyz)

            verbose(VERB_SIMPLE,f"Spangling body '{name}' (kind '{body.kind}')")

            body.center_source=body.source.center_ecl
            if body==self.root:
                self.center_root=body.source.center_ecl

            body.spangle_body()

            self._spanglers[name]=body.sg
                
        #Join spanglers
        self.sg=Spangler(spanglers=list(self._spanglers.values()))
    
        #An usefule alias
        self.data=self.sg.data
        
        #Set default observer
        self.update_perspective(n_obs=self.n_obs,alpha_obs=self.alpha_obs)
        
        #Save state of the system
        if self._resetable:
            self.save_to(self._snap_file_name)
        
        #Already spangled
        self._spangled=True

        #Update Scatterers for Albedo Computing
        self._update_scatterers()

        #Update Optical Depths
        self._update_optical_depth()
    
    def _set_observer(self,nvec=[0,0,1],alpha=0,center=None):
        """Set the position of the observer
        """
        #Only set observer if it is spangled
        if self._is_spangled():
            
            #At changing the observer, reset state
            self.sg.reset_state()
            
            #Set observer
            self.sg.set_observer(nvec=nvec,alpha=alpha,center=center)
            
            #Update areas of the spangles
            
            #Update system properties
            self.d_obs=self.sg.d_obs
            self.n_obs=self.sg.n_obs.copy()
            self.rqf_obs=self.sg.rqf_obs.copy()
            self.alpha_obs=self.sg.alpha_obs
            self.center_obs=self.sg.center_obs
        
            #Update visibility
            self.sg.update_visibility_state()
            
            #Check that observer has been set
            self._observer_set=True
            
        else:
            raise AssertionError("You must first spangle system before setting observer direction.")
            
    def _set_luz_recursive(self,name,nluz,verbosity=VERB_SIMPLE):
        """Set light source for body and 
        """
        body=self.bodies[name]
        verbose(verbosity,f"Illuminating body {name}, with nluz = {nluz} and center = {body.center_source}")
        self.sg.set_luz(nvec=nluz,center=body.center_source,name=name)
        if body.childs:
            verbose(verbosity,f"\tObject {name} has childs!")
            for child_name in body.childs:
                verbose(verbosity,f"\tCalling recursively set_luz for {child_name}")
                self._set_luz_recursive(child_name,nluz)
        else:
            verbose(verbosity,f"\tObject {name} has no childs!")
                
    def _set_luz(self,verbosity=VERB_SIMPLE):
        """Set illumination in the system.
        
        Update:
            States: illuminated, shadow, hidden_by_luz
        """
        if self._is_spangled():
            
            if not self._observer_set:
                raise AssertionError("You must first set observer before setting light.")
            
            self.bodies_illuminated=[]
            for name,body in self.bodies.items():
              
                if body.kind == "Star":
                    verbose(verbosity,f"Body {body.name} is a star... skipping")
                    continue
                    
                if body.parent.name in self.bodies_illuminated:
                    verbose(verbosity,f"Parent body of {name}, {body.parent.name}, has been already illuminated")
                    continue
                            
                #Get center of body
                center=body.center_ecl
                        
                #Get source and center
                verbose(verbosity,f"Calculating illumination for '{name}' coming from '{body.source.name}' @ {body.center_source}")            
                nluz=body.center_source-center
                body.n_luz,_ = spy.unorm(nluz)

                # Calculates Body's Phase Angle (Source, Observer)
                # body.alpha_angle = np.inner(body.n_luz, self.n_obs)
                        
                if body.kind == "Ring" and body.parent.kind == "Star":
                    verbose(verbosity,f"Parent body of ring, {body.parent.name} is a star. All spangles will be illuminated")
                    self.sg.set_luz(nvec=nluz,center=body.center_source,name=name)
                    cond=(self.sg.data.name==name)
                    self.sg.data.loc[cond,"unset"]=False
                    self.sg.data.loc[cond,"illuminated"]=True
                    self.sg.data.loc[cond,"shadow"]=False
                else:                
                    verbose(verbosity,f"Illuminating body {name} and all its childs")
                    self._set_luz_recursive(name,nluz,verbosity)
                    self.sg.update_illumination_state(included=body.source.shined)
                    self.bodies_illuminated+=[name]
                    
            self._luz_set=True
        else:
            raise AssertionError("You must first spangle system before setting light.")
            
    def update_perspective(self,n_obs=None,alpha_obs=0,center_obs=None):
        """
        Set new observer and illumination perspectives for direction values and recompute photometry.

        Parameters
        ----------
        n_obs : `array_like`
            Observer direction vector.
        alpha_obs : `float`
            Observer roll angle (radians).
        center_obs : `array_like`
            Observer position vector.
        
        Warning
        ----------
        This function may have slow performance
        """
        if n_obs is not None:
            
            #Update observing conditions
            self.n_obs,one=spy.unorm(n_obs)
            self.alpha_obs=alpha_obs
            self.center_obs=center_obs
    
        #Set observer
        self._set_observer(nvec=self.n_obs,alpha=self.alpha_obs,center=center_obs)
        self._set_luz()

        # Update data of each Body instance
        for name, body in self.bodies.items():

            cond = (self.data.name == name)

            src = self.data.loc[cond, body.sg.data.columns]
            # Ensure destination dtypes can hold source values (e.g. python objects like scatterers).
            for col in body.sg.data.columns:
                if col in src.columns and src[col].dtype == object and body.sg.data[col].dtype != object:
                    body.sg.data[col] = body.sg.data[col].astype(object)
                # Also handle numeric strictness: don't try to put floats into int columns.
                if col in src.columns and pd.api.types.is_float_dtype(src[col].dtype) and not pd.api.types.is_float_dtype(body.sg.data[col].dtype):
                    body.sg.data[col] = body.sg.data[col].astype(float)
            body.sg.data.iloc[:, :] = src.to_numpy(copy=True)
        
    
    def update_body(self,body,**props):
        """
        Update a body's properties before spangling.

        Parameters
        ----------
        body : `str` or :data:`~ body.Body`
            Identifier or instance of the body to update.
        **props : dict
            Properties to update (mass, radius, orbital elements).

        Raises
        ------
        AssertionError
            If system already spangled or body not found.
        """
        #Update spangling?
        if self._is_spangled():
            raise AssertionError("After spangling you cannot update the properties of the bodies.  Please rebuild the system")
    
        #Update body properties
        if isinstance(body,Body):
            body.update_body(**props)
        elif body in self.bodies:
            body=self.bodies[body]
            lkind=body.kind.lower()
            exec(f"body.update_{lkind}()")
        else:
            raise AssertionError("You are trying to update a body ({body}) which is not in the system")
            
        #Check if among props there is any property related to position
        if any(k in props for k in REBOUND_ORBITAL_PROPERTIES):
            raise ValueError(f"You cannot update an orbital property {props} without compromising the full simulation. Rebuild the system from scratch.")
    
    def reset(self):
        """
        Reload system state from the last snapshot.
        Works only if ``resetable = True`` on creation.
        """
        if self._resetable:
            self.load_from(self._snap_file_name)
            pass
        else:
            print("System is not resetable. Use resetable = True when defining the System or when you spangle it.")
    
    def integrate(self,*args,**kwargs):
        """
        Advance N-body simulation by a given time step, update center of each body and set positions of the Spangles.

        Parameters
        ----------
        t : `float`
            Time for integrate to it

        Raises
        ------
        AssertionError
            If system not yet spangled.
        
        Note
        ---------
        It doesn´t update the Visibility and Illumination states of the Spangles.
        """
        #Time of integration
        t=args[0]
        verbose(VERB_SIMPLE,"Integrating up to {t}")
        
        if self._spangled:
            
            #Integrate
            self.sim.integrate(*args,**kwargs)
            self.sim.move_to_com()
        
            #Update positions
            for name,body in self.bodies.items():
                
                #Position of the body according
                body.center_ecl=np.array(self.sim.particles[body.rbhash].xyz)
    
                verbose(VERB_VERIFY,f"Updating center of body {name} @ {body.center_ecl}")
                cond=self.sg.data.name==name
                self.sg.data.loc[cond,"center_ecl"]=pd.Series([list(body.center_ecl)]*sum(cond),dtype=object).values
    
            #Update positions (t = t) for rotation
            self.sg.set_positions()
            
        else:
            raise AssertionError("You must first spangle system before setting positions.")

    def integrate_perspective(self, t, n_obs = None, alpha_obs = 0, center_obs = None):
        """
        Rebound Integration of N-Body System and update center of each body and set positions of the Spangles. 
        Also update Visibility and Illumination states of the Spangles


        Parameters
        ----------
        t : float
            Target time for integration.
        n_obs : array_like or None
            Observer direction.
        alpha_obs : float
            Observer roll angle.
        center_obs : array_like or None
            Observer position.

        Warning
        --------------
        This function may have slow performance
        """
        #Integration
        self.integrate(t)

        #Updating Spangles Perspective
        self.update_perspective(n_obs = n_obs, alpha_obs = alpha_obs, center_obs = center_obs)
    
    def ensamble_system(self,lamb=0,beta=0,**physics):
        """
        Legacy :data:`~ legacy.RingedPlanet` assembly helper.

        Returns
        -------
        :
            output : :data:`~ legacy.RingedPlanet`

        Caution
        --------
        It only works on single-planet system for improving performance
        """
        #Check if observer was provided
        if "Observer" in self.bodies:
            lamb=self.bodies["Observer"].lamb
            beta=self.bodies["Observer"].beta
            
        physics_defaults=deepcopy(LEGACY_PHYSICAL_PROPERTIES)
        physics_defaults.update(dict(limb_cs=self.bodies["Star"].limb_coeffs))
        physics_defaults.update(physics)
    
        #--CONSISTENCY--
        self._ringedplanet=dict(
            
            #Behavior
            behavior=dict(shadows=True),
            
            #Units
            CU=CanonicalUnits(UL=self.ul,UM=self.um),
    
            #Basic
            Rstar=self.bodies["Star"].radius,
            Rplanet=self.bodies["Planet"].radius,
    
            Rint=self.bodies["Ring"].fi,
            Rext=self.bodies["Ring"].fe,
            i=self.bodies["Ring"].i,
    
            a=self.bodies["Planet"].a,e=self.bodies["Planet"].e,
    
            #Orbit 
            Mstar=1,x=0,lambq=0,t0=0,kepler=False,
    
            #Observer
            eobs_ecl=np.array([lamb,beta]),
    
            #Sampling
            Np=self.bodies["Planet"].nspangles,
            Nr=self.bodies["Ring"].nspangles,
    
            Nb=0,Ns=30,
    
            #Physical properties
            physics=physics_defaults,
        )
        self.RP=RingedPlanet(**self._ringedplanet)
        return self.RP
    

    def update_StellarFlux(self):
        """
        Compute the Incident Stellar Flux for illuminated spangles that do not belong
        to stars, considering the effective area per spangle and applying the Flux Law

        Attributes
        ----------
        stellar_flux : `pd.Series`
            The computed incident stellar flux, stored in the Spangles Data object
            (``self.data.stellar_flux``).

        Note
        ---------
        - Only illuminated spangles not associated with stars are considered in the computation.
        - The computation follows the Flux Law (:math:`\propto d^{-2}`)
        
        .. math::
            F = \\frac{L}{4 \\pi d^2}
        """
        #Considered Spangles
        body_names = [name for name, body in self.bodies.items() if body.kind != 'Star']

        cond = self.data.name.isin(body_names)*self.data.illuminated

        #Creating stellar_flux attribute
        self.data['stellar_flux'] = np.zeros(self.data.shape[0])

        #Computing Incident Stellar Flux
        # Avoid chained assignment (can be a no-op under pandas Copy-on-Write).
        incident = (
            abs(
                self.data.loc[cond, "asp_luz"].to_numpy(dtype=float)
                * self.data.loc[cond, "cos_luz"].to_numpy(dtype=float)
                / (4 * np.pi * self.data.loc[cond, "d_luz"].to_numpy(dtype=float) ** 2)
            )
        )
        self.data.loc[cond, "stellar_flux"] = incident


    def update_DiffuseReflection(self):
        """
        Compute the Diffuse Reflected Stellar Flux per Spangle that are both visible and illuminated,
        taking into account the illuminated side of the spangles and applying Lambert's Cosine Law.

        Attributes
        ------------
        reflected_flux : `pd.Series`
            The computed reflected flux, stored in the Spangles Data object
            (``self.data.reflected_flux``).

        Note
        -------
        - Only visible and illuminated spangles reflect the incident stellar flux.
        - Diffuse reflection considers the illuminated side of the spangles, where the condition ``cos_obs * cos_luz > 0`` ensures the observer perceives the illuminated side.
        - The computation follows Lambert's Cosine Law (Lommel - Selliger Law is also available, see :any:`scatterer.Scatterer` for details).

        .. math::
            \\frac{F}{F_0} = \\sum_i \\frac{a_{s_i}\\cos \\Lambda_i}{4 \\pi d_i^2}A_{L_i}(\\Lambda_i)\\cos Z_i

        where:

        - :math:`F/F_0` is the reflected flux.
        - :math:`a_{s_i}` is the area of the spangle.
        - :math:`\\Lambda_i` is the angle between the incident light and the surface normal.
        - :math:`d_i` is the distance from the spangle to the light source.
        - :math:`A_{L_i}(\\Lambda_i)` is the Lambertian albedo of the spangle at angle :math:`\\Lambda_i`.
        - :math:`Z_i` is the angle between the observer's line of sight and the surface normal.
        """
        #Considered Spangles
        cond = self.data.illuminated*self.data.visible*(self.data.cos_obs*self.data.cos_luz > 0)*(~self.data.hidden)

        #Creating reflected_flux attribute
        self.data['reflected_flux'] = np.zeros(self.data.shape[0])

        #Compute incident stellar flux
        self.update_StellarFlux()
        
        #Update Lambertian Albedo Values for Integration Step
        self._update_albedos()

        #Computing Diffuse Reflected Light
        # Avoid chained assignment (can be a no-op under pandas Copy-on-Write).
        reflected = (
            self.data.loc[cond, "stellar_flux"].to_numpy(dtype=float)
            * self.data.loc[cond, "lambertian_albedo"].to_numpy(dtype=float)
            * self.data.loc[cond, "cos_obs"].to_numpy(dtype=float)
        )
        self.data.loc[cond, "reflected_flux"] = reflected


    def update_Transit(self):
        """
        This method calculates the stellar flux drop caused by transiting spangles over star-kind
        bodies, accounting for limb-darkening effects and the spangle's optical properties.

        Attributes
        ----------
        transit_flux : `pd.Series`
            The computed stellar flux drop, stored in the Spangles Data object
            (``self.data.transit_flux``).

        See Also
        ------------
        - The computation follows limb-darkening laws, as described in:
        `<https://pages.jh.edu/~dsing3/David_Sing/Limb_Darkening.html>`_.
        - Limb-darkening coefficients are sourced from:
        `<https://pages.jh.edu/~dsing3/LDfiles/LDCs.CoRot.Table1.txt>`_.

        Note
        ----------
        - Only spangles transiting over star-kind bodies are considered.
        - The flux drop is calculated as:

        .. math::
            \\frac{F}{F_0}= \\sum_i a_{s_i} \\cos Z_i \\beta_i(Z_i) \\frac{I(\\mu)}{I_0}

        where:

        - :math:`F/F_0` is the normalized stellar flux drop.
        - :math:`a_{s_i}` is the area of the spangle.
        - :math:`Z_i` is the angle between the observer's line of sight and the surface normal.
        - :math:`\\beta_i(Z_i)` is the attenuation factor for the spangle at angle :math:`Z_i` **[1]**
        .. math::
            \\beta_i(Z_i) = 1 - \\frac{\\tau}{2\\cos Z_i}e^{-\\frac{\\tau}{\\cos Z_i}}
        - :math:`I(\\mu)/I_0` is the intensity of the light at projected distance over stellar disk (see :any:`science.Science.limb_darkening` for details).
        
        **[1]**  French, R.G., Nicholson, P.D., 2000. Icarus 145, 502–523. doi:10. 1006/icar.2000.6357. 

        """
        #Creating transit_flux attribute
        self.data['transit_flux'] = np.zeros(self.data.shape[0], dtype=float)

        #Over all Stars
        for star in self.bodies:

            if self.bodies[star].kind == 'Star':

                #Considered Spangles
                cond = self.data.transit_over_obs.str.fullmatch(rf'^{star}[^&]*&$')*(~self.data.hidden)

                #Source-Spangle Projected Distance
                rhos = self.data.transit_over_obs[cond].str.extract(r':([^:]*)&', expand = False).astype(float)

                #Optical Parameters
                limb_coeffs = self.bodies[star].limb_coeffs
                norm_limb_coeff = self.bodies[star].norm_limb_darkening

                beta_values = 1 - np.exp(-self.data.tau_gray_optical[cond]/abs(self.data.cos_obs[cond]))

                asp = self.data.loc[cond, 'asp_obs']
                cos_obs = abs(self.data.loc[cond, 'cos_obs'])
                star_scale = self.bodies[star].radius

                # limb_darkening = Util.limbDarkening(rhos, 1, limb_coeff, norm_limb_coeff)
                limb_darkening = Science.limb_darkening(rhos, cs = limb_coeffs , N = norm_limb_coeff)

                #Computing Stellar Flux Drop
                flux_drop = (beta_values * cos_obs * limb_darkening * (asp / star_scale**2)).to_numpy(dtype=float)
                # Avoid chained assignment (can be a no-op under pandas Copy-on-Write).
                self.data.loc[cond, "transit_flux"] = self.data.loc[cond, "transit_flux"].to_numpy(dtype=float) + flux_drop


    def update_ThermalEmission(self, bandwidth = (1.1e-6, 1.7e-6)):
        """
        Compute the Thermal Emission Flux per Spangle that are both visible and illuminated,
        taking into account the temperature of each spangle based on its temperature model.

        Parameters
        ----------
        bandwidth : `tuple`
            Wavelength range (in meters) over which to integrate the Planck function for thermal emission

        Raises
        ------
        AttributeError
            If a body lacks a temperature model ('T_model'). Define one using the Planet.set_temperature_model() method.

        Attributes
        ------------
        thermal_flux : `pd.Series`
            The computed thermal emission flux, stored in the Spangles Data object
            (``self.data.thermal_flux``).  

        Note
        -------
        - Only visible spangles not associated with stars are considered in the computation.
        - The thermal emission is calculated by integrating the Planck function over the specified bandwidth

        [1]

        .. math::
            \\frac{F}{F_0} = \\sum_i \\epsilon_i A_{s_i} \\cos Z_i \\frac{\\int_{\\lambda_{min}}^{\\lambda_{max}} B(\\lambda, T_i) d\\lambda}{F_{star}}

        where:
        - :math:`F/F_0` is the thermal emission flux.
        - :math:`\\epsilon_i` is the emissivity of the spangle.
        - :math:`A_{s_i}` is the area of the spangle.
        - :math:`Z_i` is the angle between the observer's line of sight and the surface normal.
        - :math:`B(\\lambda, T_i)` is the Planck function at wavelength :math:`\\lambda` and temperature :math:`T_i`.
        - :math:`F_{star}` is the total flux emitted by the star over the same bandwidth for normalization.

        References
        -------------
        Temperature Models are taken and adapted from the SPIDERMAN code for tidally locked exoplanets

        [1] Tom Louden, Laura Kreidberg, SPIDERMAN: an open-source code to model phase curves and secondary eclipses, Monthly Notices of the Royal Astronomical Society, Volume 477, Issue 2, June 2018, Pages 2613–2627, https://doi.org/10.1093/mnras/sty558
        """
        #Considered Spangles
        body_names = [name for name, body in self.bodies.items() if (body.kind != 'Star') and (body.kind != 'Ring')]

        # Updating Temperature Distribution for each body
        for body in body_names:
            verbose(VERB_VERIFY,f"Computing Thermal Emission for body {body}")

            if not hasattr(self.bodies[body], 'T_model'):
                raise AttributeError(f"Body '{body}' lacks a temperature model ('T_model'). Cannot compute thermal emission, please define one by Planet.set_temperature_model() method.")

            self.bodies[body].update_temperature()
            self.data.loc[self.data.name == body, 'Tem'] = self.bodies[body].sg.data['Tem'].values

        cond = self.data.name.isin(body_names)*self.data.visible*(~self.data.hidden)

        #Creating thermal_flux attribute
        self.data['thermal_flux'] = np.zeros(self.data.shape[0])

        # Star Parameters
        T_star = self.bodies[self.root.name].T_eff
        R_star = self.bodies[self.root.name].radius

        # Spangle Parameters
        asp = self.data.loc[cond, 'asp_obs'].values
        cos_obs = self.data.loc[cond, 'cos_obs'].values

        # Spangle Emissivity
        epsilon = self.data.loc[cond, 'emmisivity'].values

        #  Spangle Blackbody Temperature of Emission
        T_emission = self.data.loc[cond, 'Tem'].values

        # Star Flux for Normalization
        lambda_min, lambda_max = bandwidth
        flux_star = Science.integrate_planck_flux(T_star, lambda_min, lambda_max)*np.pi*R_star**2

        # Computing Thermal Emission Flux per Spangle
        flux_thermal = np.zeros_like(T_emission)

        for i, T in enumerate(T_emission):

            flux_thermal[i] = Science.integrate_planck_flux(T, lambda_min, lambda_max)

        self.data.loc[cond, 'thermal_flux'] = epsilon*asp*cos_obs*flux_thermal/flux_star

    
    def update_Polarization(self):

        """  
        Compute the polarized scattered light of the system using Stokes parameters.

        This method computes the total polarized flux produced by planetary bodies and rings
        in the system by evaluating the Stokes parameters (F, Q, U, V) per-spangle.
        The resulting Stokes quantities and the degree of polarization are stored 
        in the Spangles DataFrame and can be used for polarized lightcurve calculations.

        The Stokes parameters are computed using a preloaded Stokes scattering model associated
        with each body, and are normalized relative to the stellar radius and the projected
        planetary or ring area.

        Attributes
        ----------
        stokes_F : `pd.Series`
            Total scattered flux per spangle (Stokes I).
        stokes_Q : `pd.Series`
            Linear polarization component Q per spangle.
        stokes_U : `pd.Series`
            Linear polarization component U per spangle.
        stokes_V : `pd.Series`
            Circular polarization component V per spangle.
        stokes_P : `pd.Series`
            Degree of polarization per spangle.

        All attributes are stored as columns in the system-wide Spangles Data object
        (`System.data`) and in each individual body Spangler DataFrame.

        Notes
        -----
        - Only bodies of kind ``Planet`` and ``Ring`` are included in the polarization computation.
        - For planets with rings, attenuation factors are applied depending on whether the
        planet is shadowed, hidden by the ring, or indirectly illuminated.
        - Ring polarization accounts for whether the illuminated and visible sides correspond
        to forward or backward scattering.
        - The final degree of polarization is computed from the integrated Stokes parameters as:

            P = sqrt(Q**2 + U**2) / F

        - This method does not return a value; it updates internal data structures in place.

        See Also
        --------
        compute_lightcurve :
            Uses this method when the ``polarization`` effect is requested to generate polarized
            lightcurves.
        StokesScatterer.calculate_stokes :
            Low-level routine used to compute Stokes parameters for individual spangles.
        """

        # Set Stokes Parameters
        stokes_cols = ['stokes_F','stokes_Q','stokes_U','stokes_V','stokes_P']
        # Keep these as float columns; pandas is strict about int<-float assignment.
        self.data[stokes_cols] = np.zeros((self.data.shape[0], 5), dtype=float)

        # Star radius
        R_star = self.root.radius

        # Over each body
        for name, body in self.bodies.items():
            
            # For storing attributes
            cond_name = self.data.name == name
            
            # Pass for Star
            if body.kind not in ['Planet','Ring']:
                continue

            elif body.kind == 'Planet':

                # Radius for Normalization
                R_planet = body.radius
                
                # Ring object (Childs could be also Moons)
                ring_names = [
                    child_name
                    for child_name in body.childs
                    if self.bodies[child_name].kind == "Ring"
                ]
                if not ring_names:
                    raise ValueError(
                        f"Polarization for Planet '{body.name}' requires a Ring "
                        "child; none found."
                    )
                ring_name = ring_names[0]
                ring_body = self.bodies[ring_name]
                
                # Optical Depth of Ring and Attenuation factors
                tau_r = self.bodies[ring_name].taur
                cos_obs_ring = ring_body.sg.data.cos_obs.mean()
                cos_luz_ring = ring_body.sg.data.cos_luz.mean()
                A_luz = np.exp(-tau_r / np.abs(cos_luz_ring))
                A_obs = np.exp(-tau_r / np.abs(cos_obs_ring))

                # Reflection
                qreflection = 1

                #----------------------------------------------------
                # CONDITIONS
                #----------------------------------------------------

                # Visible and Illuminated
                cond_active = body.sg.data.visible & body.sg.data.illuminated

                # Hidden by Ring
                cond_hidden = body.sg.data.hidden_by_obs.str.fullmatch(rf'^{ring_name}[^&]*&$')

                # Visible but Indirectly Illuminated (in Shadow)
                cond_indirect = body.sg.data.visible & body.sg.data.shadow
                
                # Illuminated but Hidden by Rings
                cond_unseen = cond_hidden & body.sg.data.illuminated

                # Hidden and in Indirectly Illuminated (in Shadow)
                cond_both = cond_hidden & body.sg.data.shadow

                # (Visible or Hidden)&(Illuminated or Shadowed)
                cond_full = cond_active | cond_indirect | cond_unseen | cond_both

                # Check if the rings are seen edge-on and illuminated edge-on
                angle_eps = 1e-3 # Cutoff angle in deg for shadowing

                # Ring is Visible
                vcheck = abs(np.arccos(cos_obs_ring)*180/np.pi - 90.0) > angle_eps
                # Ring is Illuminated
                icheck = abs(np.arccos(cos_luz_ring)*180/np.pi - 90.0) > angle_eps

                # Apply attenuation due to ring's optical depth
                attenuation = np.ones(len(cond_full))

                # Hidden and in Indirectly Illuminated (in Shadow)
                if vcheck and icheck and cond_both.any():
                    attenuation[cond_both] *= A_luz * A_obs


                if vcheck and icheck and cond_unseen.any() and cond_indirect.any():
                    attenuation[cond_unseen] *= A_obs
                    attenuation[cond_indirect] *= A_luz
                
                # Visible but Indirectly Illuminated (in Shadow)
                elif icheck and cond_indirect.any():
                    attenuation[cond_indirect] *= A_luz

                # Illuminated but Hidden by Rings
                elif vcheck and cond_unseen.any():
                    attenuation[cond_unseen] *= A_obs

            elif body.kind == 'Ring': 

                # Radius for Normalization
                R_planet = body.parent.radius

                # Visible and Illuminated (not Shadowed)
                cond_full = body.sg.data.visible & body.sg.data.illuminated

                # Ring Parameters
                cos_obs_ring = body.sg.data.cos_obs.mean()
                cos_luz_ring = body.sg.data.cos_luz.mean()

                # Non Attenuation
                attenuation = np.ones(len(cond_full))

                # If using bilinear interpolation there is a risk of singularities at small angles
                if body.physics["interp_method"] == "bilinear":

                    # Reject cases with small angles
                    angle_eps2 = 5e-2 # Cutoff angle in deg for bilinear
                    # Visible
                    vcheck = abs(np.arccos(cos_obs_ring)*180/np.pi - 90.0) < angle_eps2
                    # Illuminated
                    icheck = abs(np.arccos(cos_luz_ring)*180/np.pi - 90.0) < angle_eps2

                    if vcheck or icheck:
                        cond_full[:] = False
                             
                # Check if the back or front of the ring is visible
                qreflection = 1
                if (cos_luz_ring < 0) ^ (cos_obs_ring < 0):                    
                    qreflection = 0  # Scattering

            #----------------------------------------------------
            # Geometrical Parameters
            #----------------------------------------------------
            azim_diff = body.sg.data.loc[cond_full, 'azim_obs_luz'].to_numpy(dtype = np.float64)
            betas_angle = body.sg.data.loc[cond_full, 'beta_loc'].to_numpy(dtype = np.float64)
            cos_obs_abs = abs(body.sg.data.loc[cond_full, 'cos_obs'].to_numpy(dtype = np.float64))
            cos_luz_abs = abs(body.sg.data.loc[cond_full, 'cos_luz'].to_numpy(dtype = np.float64))
            area_sp = body.sg.data.loc[cond_full, 'asp'].to_numpy(dtype = np.float64)/R_star**2

            # Numerical Sensivity of Stokes Method
            if cos_obs_abs.std() < 0.005:
                cos_obs_abs = np.full_like(cos_obs_abs, cos_obs_abs.mean())

            if cos_luz_abs.std() < 0.005:
                cos_luz_abs = np.full_like(cos_luz_abs, cos_luz_abs.mean())

            # ----------------------------------------------------
            # Compute Stokes Parameters
            #----------------------------------------------------
            Stokes = np.zeros((len(cond_full), 4))
            Stokes[cond_full] = body.Stokes.calculate_stokes(azim_diff, betas_angle, cos_luz_abs, cos_obs_abs, area_sp, qreflection) * attenuation[cond_full, None] * R_star**2/(np.pi*R_planet**2)

            stokes_total = np.sum(Stokes, axis=0) 

            # Calculate degree of polarization
            stokes_P = 0
            if abs(stokes_total[0]) < 1e-6:
                stokes_P = 0.0
            elif abs(stokes_total[2]) < 1e-6:
                stokes_P = -stokes_total[1]/stokes_total[0]
            else:
                stokes_P = np.sqrt(stokes_total[1]**2 + stokes_total[2]**2)/stokes_total[0]

            # Append Degree of Polarization to Stokes flux array
            Stokes = np.column_stack((Stokes/Consts.ppm, np.full(Stokes.shape[0], stokes_P/cond_full.sum())))

            # Update Spangle DataFrame
            # Ensure destination dtype is float (Stokes values are continuous).
            for col in stokes_cols:
                if col not in body.sg.data.columns:
                    body.sg.data[col] = 0.0
                if not pd.api.types.is_float_dtype(body.sg.data[col].dtype):
                    body.sg.data[col] = body.sg.data[col].astype(float)
            body.sg.data.loc[:, stokes_cols] = 0.0
            body.sg.data.loc[cond_full, stokes_cols] = Stokes[cond_full]

            self.data.loc[self.data.index[cond_name][cond_full.values], stokes_cols] = Stokes[cond_full]


    def compute_lightcurve(self, times,
                           bodies = None,
                           bandwidth = (500e-9, 600e-9),
                           effects = ['transit'], 
                           observer = None,
                           signal = None, 
                           ):
        """
        Compute the lightcurve of the system. It integrates the contributions from specified effects
        (reflection, transit, emission) over given times, considering the observer's perspective.

        Parameters
        ----------
        times : `np.ndarray`
            Array of times at which to compute the lightcurve.
        bodies : `list` or None
            List of body names to include in the lightcurve computation. If None, all bodies except the root (:data:`~ body.Star`) are included.
        bandwidth : `tuple`
            Wavelength range (in meters) over which to integrate the Planck function for thermal emission.
        effects : `list`
            List of effects to include in the lightcurve computation. Options are 'reflection', 'transit', and 'emission'.
        observer : `tuple` or None
            Observer direction in ecliptic coordinates (lambda_ecl, beta_ecl) in radians. If None, the current observer direction is used.
        signal : `dict` or None
            Dictionary of detector parameters to simulate observational signal. If None, no signal simulation is performed.

        Attributes
        ------------
        lightcurve : `dict`
            The computed lightcurve data, including times, flux, model details, effects, bodies, observer information, bandwidth, and simulated signal (if applicable).
        detector : :data:`~body.Detector`
            The detector instance used for simulating the observational signal, if applicable.

        
        """

        if not self._spangled:
            raise AssertionError("You must Spangle the system before calling compute_lightcurve(). Please call System.spangle_system() first.")


        times = np.asarray(times, dtype=float)
        if times.ndim != 1:
            raise ValueError(f"Array Dimension {times.ndim}. Times must be a 1D array.")
        
        effects = tuple(effects)
        allowed_effects = {"reflection", "transit", "emission", "polarization"}
        unknown_effects = set(effects) - allowed_effects
        if unknown_effects:
            raise ValueError(f"Unrecognized effects: {unknown_effects}.\nAllowed effects: {sorted(allowed_effects)}")
        

        if bodies is None:
            bodies = list(self.bodies.keys())
            bodies.remove(self.root.name)
        else:
            bodies = list(bodies)

        if observer is not None:
            lambda_ecl, beta_ecl = observer
            n_obs = Science.direction(lambda_ecl, beta_ecl)
            self.update_perspective(n_obs = n_obs)

        effects_registry = {
            "reflection": (lambda: self.update_DiffuseReflection(), "reflected_flux", +1.0),
            "transit": (lambda: self.update_Transit(), "transit_flux", -1.0),
            "emission": (lambda: self.update_ThermalEmission(bandwidth), "thermal_flux", +1.0),
            "polarization": (lambda: self.update_Polarization(), ['stokes_F', 'stokes_P'], +1.0),
        }

        # ----------------------------
        # Output DataFrame MultiIndex
        # ----------------------------
        columns = pd.MultiIndex.from_product([bodies, 
                                              list(effects_registry.keys())+['scattering']],
                                          names=["body", "effect"])

        df_output = pd.DataFrame(index=pd.Index(times, name="time"), 
                                 columns=columns, 
                                 dtype=float)

        iterator = tqdm(enumerate(times), total = len(times), desc = 'Computing Lightcurve ')
        for i, t in iterator:

            self.integrate_perspective(t)

            for effect in effects:
                func_effect, data_column, sgn = effects_registry[effect]
                func_effect()

                for body in bodies:

                    cond_body = (self.data.name == body)
                    flux_val = np.sum(self.data.loc[cond_body, data_column].values, axis = 0)

                    # Updates Polarized Flux and Degree of Polarization
                    if effect == 'polarization':
                        
                        df_output.loc[t, (body, 'reflection')] = 0
                        df_output.loc[t, (body, 'scattering')] = sgn * flux_val[0]
                        df_output.loc[t, (body, effect)] = sgn * flux_val[1]

                    else: 
                        df_output.loc[t, (body, effect)] = sgn * flux_val

        total_flux = df_output.drop(columns="polarization", level="effect").sum(axis=1).values + 1.0  # Normalized Flux

        lightcurve = {"times": times,
                      "total_flux": total_flux,
                      "effects": effects,
                      "bodies": bodies,
                      "observer": {"n_obs": self.n_obs,
                                   "direction": np.rad2deg(self.rqf_obs[1:])
                                 },
                      "bandwidth": bandwidth,
                     }

        for effect in effects:
            if effect == 'polarization':
                lightcurve[effect] = df_output.loc[:, (bodies, effect)]
                lightcurve['scattering'] = df_output.loc[:, (bodies, 'scattering')]
            else:
                lightcurve[effect] = df_output.loc[:, (bodies, effect)]
        
        self.lightcurve = lightcurve

        if signal is not None:

            if type(signal) is not dict:
                raise ValueError("Signal parameter must be a dictionary of Detector parameters.")

            detector = Detector(**signal)
            detector.set_source(self.root)

            signal_times, signal_flux, signal_error = detector.generate_signal(times*self.ut, total_flux)

            self.detector = detector

            lightcurve["signal"] = {"times": signal_times,
                                    "signal_flux": signal_flux,
                                    "signal_error": signal_error}

        return lightcurve


