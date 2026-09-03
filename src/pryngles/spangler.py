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

#Aliases

import pandas as pd
import random

#Specialized plotting methods
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib import animation
from celluloid import Camera # getting the camera
import itertools
from tqdm import tqdm



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Class Spangler
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Spangler(PrynglesCommon):
    """
    Represents a collection of spangles associated with one or more astrophysical objects.

    A `Spangler` object discretizes the surface of bodies into "spangles" — discrete units used to
    model light-matter interactions, such as photometry, in the `Pryngles` package. It supports
    two initialization modes: creating a new spangler for a single object or joining multiple
    existing spanglers.

    Parameters
    ----------
    spanglers : list of Spangler, optional
        List of `Spangler` objects to join into a single spangler. If provided, the object
        aggregates the spangles from all listed spanglers. Default is an empty list.
    nspangles : int, optional
        Number of spangles to generate for a single object. Must be positive. Default is 1.
    name : str, optional
        Name identifying the body associated with the spangles. If None, a random 16-bit hash
        is generated. Default is None.
    n_equ : array-like (3,), optional
        Unit vector normal to the equatorial plane of the body in the equatorial system.
        Default is [0, 0, 1].
    alpha_equ : float, optional
        Roll angle of the x-axis in the equatorial system, in radians. Currently not implemented.
        Default is 0.
    center_equ : array-like (3,), optional
        Position of the spangler’s center in the equatorial system, in length units.
        Default is [0, 0, 0].
    center_ecl : array-like (3,), optional
        Position of the spangler’s center in the ecliptic system, in length units.
        Default is [0, 0, 0].
    w : float, optional
        Angular velocity of the body’s rotation, in radians per unit time. Default is 0.
    q0 : float, optional
        Initial longitude at time t=0, in radians. Default is 0.

    Attributes
    ----------
    nspangles : int
        Total number of spangles in the spangler.
    data : pandas.DataFrame
        DataFrame containing detailed information about each spangle, including positions,
        orientations, and states across reference systems. See :data:`pryngles.consts.SPANGLER_COLUMNS`
        for column descriptions.
    name : str or list of str
        Name(s) of the body (or bodies, if joined) associated with the spangles.
    shape : str
        Geometry of the spangler ("vanilla", "circle", "ring", "sphere", or "Join" for joined spanglers).
    sample : Sampler or None
        Associated `Sampler` object used to generate spangle positions, if applicable.
    n_obs : np.ndarray (3,)
        Unit vector towards the observer in the ecliptic system.
    n_luz : np.ndarray (3,)
        Unit vector towards the light source in the ecliptic system.
    M_equ2ecl : dict
        Transformation matrices from equatorial to ecliptic coordinates, keyed by body name.
    qhulls : dict
        Convex hulls of spangles for intersection calculations, keyed by body name.

    Raises
    ------
    AssertionError
        If a provided spangler in `spanglers` is not an instance of `Spangler`.
    ValueError
        If duplicate names are found when joining spanglers.

        
    Examples
    ------------
    >>> # Let's create a planet and spangle it!!
    >>>
    >>> # First create your body
    >>> earth = pr.Body(kind = "Planet",
    ...                 defaults = pr.PLANET_DEFAULTS,
    ...                 parent = None,
    ...                 name = "Earth",
    ...                 radius = 1.0)
    >>>
    >>> ring = pr.Ring(parent=earth, fi = 1.5, fe = 2)  
    >>>
    >>> # Now spangle it
    >>> earth.spangle_body()
    >>> ring.spangle_body()  
    >>>
    >>> # Join the bodies
    >>> ringedplanet = pr.Spangler(spanglers=[earth.sg, ring.sg])
    >>>
    >>> # Now you can visualize it                 
    >>> ringedplanet.plot2d() 

    .. image:: images/spangler_example.png
        :align: center
    """

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Bassic methods
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    def __init__(self,
                 #Initialization using specific options
                 #Initialization with a list of spanglers
                     spanglers=[],
                 #Basic
                     nspangles=1,
                     name=None,
                     n_equ=SPANGLER_COLUMNS["n_equ"],
                     alpha_equ=SPANGLER_COLUMNS["alpha_equ"],
                     center_equ=SPANGLER_COLUMNS["center_equ"],
                     center_ecl=SPANGLER_COLUMNS["center_ecl"],
                 #Optional
                     w=SPANGLER_COLUMNS["w"],
                     q0=SPANGLER_COLUMNS["q0"],
                ):
        
        #Common attributes
        self.n_obs=np.array([0,0,1])
        self.n_luz=np.array([0,0,1])
        self.d_obs=self.d_luz=1
        self.sample=None
        self.geometry=-1
        
        #Direction of vantages point in spherical coordinates
        self.rqf_obs=Science.spherical(self.n_obs)
        self.rqf_luz=Science.spherical(self.n_luz)
        self.center_luz=None
        self.center_obs=None
        
        #Transformation matrices from equatorial to ecliptic coordinates
        self.M_equ2ecl=dict()
        self.M_ecl2equ=dict()
        
        #Convex hulls of spanglers
        self.qhulls=dict()
        
        #Required for plotting
        self.fig2d=None
        self.ax2d=None
        self.fig3d=None
        self.ax3d=None
        
        #Create a spanglers with a list of other spanglers
        if len(spanglers)>0:
            verbose(VERB_SIMPLE,f"Joining {len(spanglers)} spanglers")
            self._join_spanglers(spanglers)
            
        #Create a spangler with the desired options
        else:
            #Attributes
            self.nspangles=nspangles
            self.shape="vanilla" #No geometry defined
            
            #Default property values
            self._defaults=deepcopy(SPANGLER_COLUMNS)

            if not name:
                #Generate a random hash for object
                self.name=str(random.getrandbits(16))
                verbose(VERB_VERIFY,f"Generating random hash {self.name}")
            else:
                self.name=name
                
            self._defaults.update(dict(name=self.name))
            
            #Update other parameters
            self._defaults.update(
                dict(w=w, q0=q0, 
                     n_equ=n_equ, alpha_equ=alpha_equ,
                     center_equ=center_equ, center_ecl=center_ecl)
            )

            #Create Spangler dataframe
            if self.nspangles>0:
                
                #Create a simple DataFrame with the default values
                self.data=pd.DataFrame([list(self._defaults.values())]*self.nspangles,
                                       columns=self._defaults.keys())

                self.M_equ2ecl[name], _ = Science.rotation_matrix(n_equ,alpha_equ)
        
            else:        
                verbose(VERB_SIMPLE,f"Creating a blank Spangler")
                #Creat a blank DataFrame
                self.data=pd.DataFrame(columns=self._defaults.keys())
        
    def reset_state(self):
        """
        Resets the state of all spangles to their initial values.

        This method clears all visibility and illumination states, marking them as unset,
        and resets intersection-related fields.

        Returns
        -------
        None

        Notes
        -----
        Modifies the `data` DataFrame in-place, resetting columns listed in
        :data:`pryngles.consts.SPANGLER_SOURCE_STATES` and
        :data:`pryngles.consts.SPANGLER_VISIBILITY_STATES` to False, setting "unset" to True,
        and clearing "hidden_by_*" and "transit_over_*" fields.
        """
        self.data[list(SPANGLER_SOURCE_STATES)+list(SPANGLER_VISIBILITY_STATES)]=False
        self.data["unset"]=True
        for coords in "int","obs","luz":
            self.data["hidden_by_"+coords]=""
            self.data["transit_over_"+coords]=""

    def set_scale(self,scale):
        """
        Adjusts the scale of all length and area attributes of the spangles.

        Parameters
        ----------
        scale : float
            Scaling factor. Lengths are multiplied by `scale`, areas by `scale**2`, and
            vector components by `scale`.

        Returns
        -------
        None

        Notes
        -----
        Updates the `data` DataFrame in-place, affecting columns in
        :data:`pryngles.consts.SPANGLER_LENGTHS`, :data:`pryngles.consts.SPANGLER_AREAS`,
        and :data:`pryngles.consts.SPANGLER_VECTORS`. Sets the `scale` attribute.
        """
        self.scale=scale
        self.data[SPANGLER_LENGTHS]*=self.scale
        self.data[SPANGLER_AREAS]*=self.scale**2
        for vector in SPANGLER_VECTORS:
            self.data[vector]=[np.array(v)*scale for v in self.data[vector]]
        
    def _join_spanglers(self,spanglers):
        """
        Combines multiple `Spangler` objects into a single spangler.

        Parameters
        ----------
        spanglers : list of Spangler
            List of `Spangler` objects to join.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If any element in `spanglers` is not a `Spangler` instance.
        ValueError
            If duplicate names are found among the spanglers.

        Notes
        -----
        Modifies the current instance by aggregating the `data` DataFrames, merging
        transformation matrices, and updating attributes like `nspangles` and `name`.
        The resulting `shape` is set to "Join".
        """
        self.name=[]
        for spangler in spanglers:
            # if not isinstance(spangler,Spangler):
            #     raise AssertionError(f"One of the spangler is not an Spangler instance")
                
            # if spangler.name in self.name:
            #     raise ValueError(f"Hash '{spangler.name}' already included in spangler '{self.name}'")
                
            self.name+=[spangler.name]

        #When joining there is no single geometry
        self.shape="Join"
        
        #Set of spanglers
        self.spanglers=spanglers

        #Concatenate data
        datas=[spangler.data for spangler in spanglers]
        self.data=pd.concat(datas,ignore_index=True)

        self.M_equ2ecl=dict()
        for spangler in spanglers:
            self.M_equ2ecl.update(spangler.M_equ2ecl)

        #Join properties
        self.nspangles=len(self.data)
        
    def get_mem_usage(self,info=False):
        """
        Calculates the memory usage of the spangler’s DataFrame.

        Parameters
        ----------
        info : bool, optional
            If True, prints detailed memory usage information. Default is False.

        Returns
        -------
        float
            Memory usage in megabytes.

        Examples
        --------
        >>> sp = Spangler(nspangles=1000)
        >>> sp.get_mem_usage()
        0.123  # Example output, actual value depends on data
        >>> sp.get_mem_usage(info=True)  # Prints detailed info
        """
        mem_usage=self.data.memory_usage(deep=True).sum()/1024**2
        if info:
            print(f"Basic information:\n")
            self.data.info(memory_usage="deep")
            print(f"\nDetailed size:\n")
            print(self.data.memory_usage(deep=True).to_string())
            print(f"\nTotal size: {mem_usage:.1g} Mb")

        return mem_usage
        
    def set_positions(self,
                      n_equ=[],alpha_equ=0,
                      center_equ=[],center_ecl=[],
                      t=None
                     ):
        """
        Updates the positions and orientations of spangles across reference systems.

        Parameters
        ----------
        n_equ : array-like (3,), optional
            Normal vector to the equatorial plane. If provided, updates transformation matrices.
            Default is an empty list.
        alpha_equ : float, optional
            Roll angle of the x-axis in the equatorial system, in radians (not implemented).
            Default is 0.
        center_equ : array-like (3,), optional
            Center position in the equatorial system, in length units. Default is an empty list.
        center_ecl : array-like (3,), optional
            Center position in the ecliptic system, in length units. Default is an empty list.
        t : float, optional
            Time in unit time. If provided, updates equatorial coordinates based on rotation.
            Default is None.

        Returns
        -------
        None

        Notes
        -----
        Updates the `data` DataFrame columns including "center_equ", "center_ecl", "n_equ",
        "q_equ", "x_equ", "y_equ", "z_equ", "ns_equ", "x_ecl", "y_ecl", "z_ecl", "ns_ecl",
        "wx_ecl", and "wy_ecl". Transformation matrices in `M_equ2ecl` are updated if `n_equ`
        is provided.
        """
        verbose(VERB_VERIFY,f"Setting positions")

        # Pandas >= 2.2 is strict about assigning floats into int columns.
        # These coordinates are continuous, so keep them as float dtype.
        for col in ("x_ecl", "y_ecl", "z_ecl"):
            if col not in self.data.columns:
                self.data[col] = np.nan
            if not pd.api.types.is_float_dtype(self.data[col].dtype):
                self.data[col] = self.data[col].astype(float)

        #Update normal vectors
        qupdate=False

        #Update equatorial coordinates by rotation
        if t is not None:
            verbose(VERB_VERIFY,f"Updating rotations at t = {t}")

            self.data["q_equ"]=[q+q0+w*t for q,w,q0 in zip(self.data.q_equ,self.data.w,self.data.q0)]
            self.data[["x_equ","y_equ","z_equ"]]=                [Science.cartesian(r) for r in np.array(self.data[["r_equ","q_equ","f_equ"]])]

            qupdate=True

        #If equatorial positions have been changed
        if qupdate and self.sample:
 
            #Update spangles orientations
            verbose(VERB_VERIFY,f"Generating normal vectors")

            #If the spangler has been poputaled update normals
            if self.sample:
                self.data["ns_equ"]=pd.Series(
                    list(
                        self.sample.update_normals(self.data[["x_equ","y_equ","z_equ"]])
                    ),dtype=object
                )

        #Convert from equatorial to ecliptic
        verbose(VERB_VERIFY,f"Converting to equatorial")
        groups = self.data.groupby('name', sort=False)
        for name, group in groups:

            M_equ2ecl = self.M_equ2ecl[name]
            g_indices = group.index
            
            # Transform positions: batch M @ (r + c_equ) + c_ecl
            r_equ = np.array(group[['x_equ', 'y_equ', 'z_equ']])  # (G, 3)
            c_equ = np.stack(group['center_equ'].values)  # (G, 3)
            c_ecl = np.stack(group['center_ecl'].values)  # (G, 3)
            self.data.loc[g_indices, ['x_ecl', 'y_ecl', 'z_ecl']] = (M_equ2ecl @ (r_equ + c_equ).T).T + c_ecl

            # #Update orientation of the spangle
            ns_equ = np.stack(group['ns_equ'].values)  # (G, 3)
            self.data.loc[g_indices, 'ns_ecl'] = pd.Series(((M_equ2ecl @ ns_equ.T).T).tolist(), index=g_indices, dtype=object)


        #Update matrix of the transformation from ecliptic to local (horizontal) reference frame of the spangle

        #Search all spangles pointing towards ez or -ez
        ns_ecl = np.stack(self.data.ns_ecl.values)  # Shape: (N, 3)
        cond = pd.Series(np.abs(np.dot(ns_ecl, [0, 0, 1])) != 1.0, index = self.data.index)
        if cond.any():
            verbose(VERB_VERIFY,f"Setting local vectors based on ns: {sum(cond)}")
            
            #wy = ez x ns because with this definition seen from above the system is oriented as usual
            index = self.data.index[cond]
            ns_ecl_masked = ns_ecl[cond.values]
            
            # wy_ecl: unorm(cross(ez, ns))[0] (vectorized)
            cross_ez_ns = np.cross([0,0,1], ns_ecl_masked)
            wy = np.divide(cross_ez_ns, np.linalg.norm(cross_ez_ns, axis=1)[:, np.newaxis])
            self.data.loc[index, "wy_ecl"] = pd.Series(wy.tolist(), index=index)
            
            # wx_ecl: cross([wy, 0, 0], ns)  (vectorized)
            wx = np.cross(wy, ns_ecl_masked)
            self.data.loc[index, "wx_ecl"] = pd.Series(wx.tolist(), index=index, dtype=object)

            cond=~cond

        else:
            cond = pd.Series([True]*self.nspangles)

        #Spangles pointing towards ez or -ez
        if cond.any():
            verbose(VERB_VERIFY,f"Setting local matrix based on ex: {sum(cond)}")
            
            indices_cond = self.data[cond].index
            self.data.loc[indices_cond, 'wx_ecl'] = pd.Series([[1.0, 0.0, 0.0]]*len(indices_cond), index=indices_cond, dtype=object)

            # wy_ecl: unorm(cross(ns_ecl, wx))[0] — vectorized
            crosses = np.cross(np.stack(self.data.loc[indices_cond, 'ns_ecl'].values), [1.0, 0.0, 0.0])  # (M, 3)
            wy = np.divide(crosses, np.linalg.norm(crosses, axis=1)[:, np.newaxis])
            self.data.loc[indices_cond, 'wy_ecl'] = pd.Series(wy.tolist(), index=indices_cond)
            
        #Update velocities
        #Not implemented yet

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Tested methods from module file spangler
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def populate_spangler(self,
                          shape="circle",preset=False,spangle_type=SPANGLE_SOLID_ROCK,
                          scale=1,seed=0,**shape_args):
        
        """
        Populates the spangler with spangles based on a specified geometry.

        Parameters
        ----------
        shape : str, optional
            Geometry of the spangler. Options are "circle", "ring", or "sphere". Default is "circle".
        preset : bool, optional
            If True, uses a precomputed sample from :class:`pryngles.sampler.Sampler`. Default is False.
        spangle_type : int, optional
            CousinsType of spangle, as defined in :mod:`pryngles.consts` (e.g, :data:`pryngles.consts.SPANGLE_SOLID_ROCK`).
            Default is `SPANGLE_SOLID_ROCK`.
        scale : float, optional
            Scaling factor for the spangle sizes and positions. Default is 1.
        seed : int, optional
            Seed for random number generation. If 0, no seed is set. Default is 0.
        **shape_args : dict
            Additional arguments for the `Sampler` geometry generation (e.g., `ri` for "ring").

        Returns
        -------
        None

        Notes
        -----
        Updates the `data` DataFrame with spangle positions, normals, and areas. For "ring" shapes,
        adds hidden spangles at the inner border. Calls :meth:`set_positions` to update coordinates.
        """
        #Check if preset
        if preset:
            verbose(VERB_VERIFY,f"Populating spangler from preset for {shape}")
            preset=(shape,shape_args)
            self.sample=Sampler(preset=preset,N=self.nspangles,seed=seed)   
        else:
            verbose(VERB_VERIFY,f"Generating spangler from scratch")
            self.sample=Sampler(N=self.nspangles,seed=seed)
            exec(f"self.sample.gen_{shape}(**shape_args)")
    
        self.shape=shape
        self.data["geometry"]=self.sample.geometry
        self.data["spangle_type"]=spangle_type
    
        if self.sample.geometry in [SAMPLER_GEOMETRY_SPHERE]:
            #Purge sample if it is in 3d
            verbose(VERB_VERIFY,f"Purging 3d sample")
            self.sample.purge_sample()
            self.nhidden=0
            
        elif self.shape == "ring":
            #Number of hidden points
            """
            The number of hidden points for a ring is choosen in such a way that the ratio between t
            he area of the circle sector to the area of the circle segment is larger than (1-epsilon)
            
            Ag / As = (r^2 sin(teta)/2)/(r^2 teta/2) > (1-epsilon)
            
            Expanding sin(teta) we get:
            
            1 - teta^2/6 > 1 - epsilon
            """
            nhidden=int(2*np.pi/(6*SPANGLER_EPS_BORDER)**0.5)
    
            #Add hidden spangles to ring inner borders
            pp_border=np.zeros((nhidden,3))
            ss_border=np.zeros((nhidden,3))
            ns_border=np.zeros((nhidden,3))
            for i,theta in enumerate(np.linspace(0,2*np.pi,nhidden)):
                pp_border[i]=[self.sample.ri,theta,0]
                ss_border[i]=[self.sample.ri*np.cos(theta),
                              self.sample.ri*np.sin(theta),
                              0]
                ns_border[i]=[0,0,1]
            self.sample.pp=np.vstack((self.sample.pp,pp_border))
            self.sample.ss=np.vstack((self.sample.ss,ss_border))
            self.sample.ns=np.vstack((self.sample.ns,ns_border))
            self.sample.N+=nhidden
            self.nhidden=nhidden
                    
        #Check if number of samples is not equal to that of spangles defined when the spangler was created
        if self.sample.N!=self.nspangles:
            verbose(VERB_SYSTEM,f"Sample size {self.sample.N} is different from spangles {self.nspangles}. Adjusting.")
    
            #Difference between sampler number of points and number of spanglers
            dif=self.sample.N-self.nspangles
            
            if dif>0:
                #Add spangles
                verbose(VERB_SYSTEM,f"Adding {dif} entries to DataFrame")
                for i in range(dif):
                    df=pd.DataFrame([self.data.iloc[-1]])
                    self.data=pd.concat([self.data,df],ignore_index=True)
            else:
                #Remove spangles
                verbose(VERB_SYSTEM,f"Removing {-dif} entries to DataFrame")
                self.data.drop(range(self.nspangles+dif,self.nspangles),inplace=True)
                
            self.nspangles=self.sample.N
        
        #Area
        self.data["asp"]=self.sample.aes*scale**2
        self.data["dsp"]=2*(self.data["asp"]/np.pi)**0.5
        
        #Update scale
        self.data["scale"]=scale
    
        #Store positions in DataFrame
        self.data[["x_equ","y_equ","z_equ"]]=self.sample.ss*scale
        self.data[["r_equ","q_equ","f_equ"]]=self.sample.pp
        self.data["r_equ"]*=scale
    
        #Update normal vectors
        self.data["ns_equ"]=pd.Series(list(self.sample.ns),dtype=object)
            
        #Hide border points in case of ring
        if shape == "ring":
            self.data.loc[self.nspangles-self.nhidden:self.nspangles,"hidden"]=True
            
        #Update positions
        # self.set_positions()
        
    def _update_column_order(self):
        """
        Reorders the columns of the `data` DataFrame for consistency.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If any column in `data` is not present in :data:`pryngles.consts.SPANGLER_KEY_ORDERING`.

        Notes
        -----
        Modifies the `data` DataFrame in-place based on the order defined in
        :data:`pryngles.consts.SPANGLER_KEY_ORDERING`.
        """
        for col in self.data.columns:
            if col not in SPANGLER_KEY_ORDERING:
                raise AssertionError(f"Column {col} not present in key ordering.  Please checl SPANGLER_KEY_ORDERING")
        
        #Order columns
        self.data=self.data.loc[:,SPANGLER_KEY_ORDERING]
         
    
    def plot3d(self,
               coords="ecl",
               only=None,
               center_at=None,
               not_plot=[],
               fsize=5,
               factor=1.2,
               statemark=0,
               show_directions=False
              ):
        """
        Visualizes the spangles in a 3D plot.

        Parameters
        ----------
        coords : str, optional
            Coordinate system to plot. Options are "equ", "ecl", "obs", "luz", "int". Default is "ecl".
        only : str, optional
            Name of the single object to plot. If provided, overrides other filters. Default is None.
        center_at : str, optional
            Name of the object to center the plot around. Default is None.
        not_plot : list of str, optional
            Names of objects to exclude from the plot. Default is an empty list.
        fsize : float, optional
            Size of the figure in inches (square figure). Default is 5.
        factor : float, optional
            Scaling factor for axis limits relative to the maximum range. Default is 1.2.
        statemark : float, optional
            Fraction of spangles to mark with their state (0 to 1). If 0, no marks are shown. Default is 0.
        show_directions : bool, optional
            If True, displays normal vectors and intersection direction vectors. Default is False.

        Returns
        -------
        None

        Notes
        -----
        Sets the `fig3d` and `ax3d` attributes. Spangle colors reflect their illumination and visibility states.
        """
        bgcolor='k'
    
        #Plot only a given object
        if only:
            not_plot=list(self.data.name.unique())
            if only not in not_plot:
                raise ValueError(f"Spangler '{only}' not among available spanglers ({not_plot})")
            else:
                not_plot.remove(only)
                center_at=only
    
        #Check if plot is in the ecliptic system
        qecl=True
        if 'ecl' not in coords:
            qecl=False
    
        scoords=coords
        coords=[f"x_{scoords}",f"y_{scoords}",f"z_{scoords}"]
        
        #Center
        cond=(self.data.name==center_at)
        x_cen,y_cen,z_cen=self.data[cond][coords].mean() if sum(cond)>0 else np.array([0,0,0])
        
        #Figure
        fig=plt.figure(figsize=(fsize,fsize))
        fig.patch.set_facecolor(bgcolor)
        ax=fig.add_subplot(111,projection='3d',facecolor=bgcolor)
        ax.axis("off")
        
        #Spangles
        for i in range(self.nspangles):
    
            #Avoid plotting 
            name=self.data.loc[i,"name"]
            if name in not_plot:
                continue
            
            #Reference transparency of spangles
            alpha_base=0.5
    
            #Avoid hidden spangles
            if self.data.loc[i,"hidden"]:
                continue
    
            spangle_type=self.data.loc[i,"spangle_type"]
    
            #Define the color according to illumination or shadow
            state=""
            color_hls=SPANGLES_DARKNESS_COLOR #Default color: gray
    
            #Define color according to illumination or shadow
            if self.data.loc[i,"shadow"]:
                #Inside a shadow
                state+="S."
                color_hls=SHADOW_COLOR_LUZ #Gray
            
            if self.data.loc[i,"illuminated"]:
                #Illuminated
                state+="I."
                color_hls=SPANGLE_COLORS[spangle_type] #Planet color
                
            #Modify color according to visibility, transmission or darkness
            if not self.data.loc[i,"illuminated"]:
                #In darkness
                state+="D."
                color=Plot.rgb(color_hls) #No color modification
                
            if self.data.loc[i,"transmit"]:
                #Transmitting
                state+="T."
                color=Plot.rgb(color_hls) #No color modification
    
            if not self.data.loc[i,"visible"]:
                #Not visible
                state+="N."
                color=Plot.rgb([color_hls[0],
                                color_hls[1]/2, #Reduce level to half
                                color_hls[2]
                               ])
            else:
                #Invisible
                state+="V."
                color=Plot.rgb(color_hls) #No color modification
                
            if self.data.loc[i,"unset"]:
                state+="U."
                color=Plot.rgb([0,0.5,0])
                
            #Define alpha according to albedo
            alpha=alpha_base*self.data.albedo_gray_normal[i]
    
            center=[self.data[coords[0]][i]-x_cen,self.data[coords[1]][i]-y_cen,self.data[coords[2]][i]-z_cen]
            radius=self.data.dsp[i]/2
            zDir=self.data[f"ns_{scoords}"][i]
    
            #verbose(VERB_DEEP,i,center,radius,zDir)
            Plot.circle3d(ax,
                          center=center,
                          radius=radius,
                          zDir=zDir,
                          color=color,alpha=alpha,lw=0)
            if statemark:
                if np.random.rand()>1-statemark:
                    ax.text(center[0],center[1],center[2],state,fontsize=6,color='w')
            
        #Aspect
        ax.set_box_aspect([1,1,1])
    
        #Zoom around center
        cond=(self.data.name==center_at)
        cond=cond if sum(cond)>0 else np.array([True]*self.nspangles)
    
        #Not 
        cond=cond&(~self.data.name.isin(not_plot))
        
        #Range
        maxval=1.0*np.abs(self.data[cond][coords].to_numpy()-[x_cen,y_cen,z_cen]).max()
        ax.set_xlim(-maxval,maxval)
        ax.set_ylim(-maxval,maxval)
        ax.set_zlim(-maxval,maxval)
        
        #Decoration
        xmin,xmax=factor*np.array(list(ax.get_xlim()))
        ymin,ymax=factor*np.array(list(ax.get_ylim()))
        zmin,zmax=factor*np.array(list(ax.get_zlim()))
    
        #Axis
        ax.plot([xmin,xmax],[0,0],[0,0],'w-',alpha=0.3)
        ax.plot([0,0],[ymin,ymax],[0,0],'w-',alpha=0.3)
        ax.plot([0,0],[0,0],[zmin,zmax],'w-',alpha=0.3)
        ax.text(xmax,0,0,rf"$x_{{{scoords}}}$",color='w',alpha=0.5,fontsize=8)
        ax.text(0,ymax,0,rf"$y_{{{scoords}}}$",color='w',alpha=0.5,fontsize=8)
        ax.text(0,0,zmax,rf"$z_{{{scoords}}}$",color='w',alpha=0.5,fontsize=8)
        
        #Plot n_obs and n_luz vector only in the case of ecliptic system
        increase=1.05*factor*maxval
        if qecl:
            ax.quiver(+self.n_luz[0]*increase,+self.n_luz[1]*increase,+self.n_luz[2]*increase,
                      -self.n_luz[0]*increase,-self.n_luz[1]*increase,-self.n_luz[2]*increase,
                      color='y',alpha=0.7)
            ax.text(self.n_luz[0]*increase,self.n_luz[1]*increase,self.n_luz[2]*increase,
                    r"$n_{luz}$",color='w',alpha=0.7,fontsize=8,ha='left',va='bottom')
            ax.quiver(+self.n_obs[0]*increase,+self.n_obs[1]*increase,+self.n_obs[2]*increase,
                      -self.n_obs[0]*increase,-self.n_obs[1]*increase,-self.n_obs[2]*increase,
                      color='c',alpha=0.7)        
            ax.text(self.n_obs[0]*increase,self.n_obs[1]*increase,self.n_obs[2]*increase,
                    r"$n_{obs}$",color='c',alpha=0.7,fontsize=8,ha='right',va='top')
            ax.view_init(30,60)
        else:
            r_obs,t_obs,f_obs=Science.spherical(self.n_obs)
            ax.view_init(f_obs*Consts.rad,t_obs*Consts.rad)
            
        #Show vectors
        if show_directions:
            
            cond=cond&(~self.data.hidden)
            
            if scoords=="ecl":
                nstr="n_int_"+scoords
            else:
                nstr="n_"+scoords
            
            #It is important to stress that "vectors" should be directed towards the light-source 
            vectors=np.array(list(self.data.loc[cond,nstr]))
            
            normals=np.array(list(self.data.loc[cond,"ns_"+scoords]))
            
            ax.scatter(self.data.loc[cond,"x_"+scoords],self.data.loc[cond,"y_"+scoords],self.data.loc[cond,"z_"+scoords],
                       marker="*",c='w',s=10)
            ax.quiver(self.data.loc[cond,"x_"+scoords],self.data.loc[cond,"y_"+scoords],self.data.loc[cond,"z_"+scoords],
                      vectors[:,0],vectors[:,1],vectors[:,2],color='w',label="From intersection")
            ax.quiver(self.data.loc[cond,"x_"+scoords],self.data.loc[cond,"y_"+scoords],self.data.loc[cond,"z_"+scoords],
                      normals[:,0],normals[:,1],normals[:,2],color='y',label="Normal")
            
            leg=ax.legend(loc='lower right',facecolor='k',ncol=3,prop={'size':8},bbox_to_anchor=(0.5,-0.01, 0.5, 0.5))
            frame=leg.get_frame()
            frame.set_edgecolor("k")
            for text in leg.get_texts():
                text.set_color("w")
    
        #Title
        ax.set_title(f"Spangler {self.shape}, N = {self.nspangles}",
                     color='w',fontsize=10)
        Plot.pryngles_mark(ax)
        
        fmark=""
        if statemark:
            fmark=f", I/D: Illum./Dark, V/N: Visible/Invisible, S: Shadow, T: Transmit"
        
        #Scale
        ax.text2D(0,0,f"Axis scale: {maxval*factor:.2g} {fmark}",
                fontsize=7,color='w',
                transform=ax.transAxes)
    
        fig.tight_layout()
        self.fig3d=fig
        self.ax3d=ax
    
    def set_intersect(self,
                      nvec=[0,0,1],
                      alpha=0,
                      center=None,
                      name=None,
                     ):
        """
        Sets the positions and orientations of spangles in the intersection reference system.

        Parameters
        ----------
        nvec : array-like (3,), optional
            Vector towards the intersection vantage point in the ecliptic system. Default is [0, 0, 1].
        alpha : float, optional
            Roll angle of the x-axis in the intersection system, in radians. Default is 0.
        center : array-like (3,), optional
            Position of the vantage point in the ecliptic system. If None, assumes infinite distance.
            Default is None.
        name : str, optional
            Name of the specific object to apply the transformation to. If None, applies to all.
            Default is None.

        Returns
        -------
        tuple
            - cond : np.ndarray
                Boolean mask indicating which spangles were updated.
            - n_int : np.ndarray (3,)
                Normalized vector towards the vantage point.
            - d_int : float
                Distance to the vantage point (infinite if `center` is None).

        Notes
        -----
        Updates the `data` DataFrame with intersection coordinates and states, and sets attributes
        like `n_int`, `d_int`, and `M_ecl2int`.
        """
        
        verbose(VERB_SIMPLE,
                f"Setting intersect using nvec = {nvec}, alpha = {alpha} center = {center}, name = {name}")
        
        verbose(VERB_VERIFY,f"Generating intersection matrices from pvec = {nvec}")
    
        #Unitary observer vector
        n_int,norm=spy.unorm(nvec)
        alpha_int=alpha
        
        #Store n_int and d_int for update state purposes
        self.rqf_int=Science.spherical(n_int)
        self.n_int=n_int
        
        #Distance to center of intersection
        if center is None:
            self.infinite=True
            d_int=np.inf
            center=np.array([0,0,0])
        else:
            self.infinite=False
            d_int=np.linalg.norm(center)
            center=np.array(center)
        self.d_int=d_int
    
        #Transformation matrices
        M_int2ecl,self.M_ecl2int=Science.rotation_matrix(n_int,alpha_int)
        self.M_int2ecl=M_int2ecl
        
        #Depending on body
        cond = np.full(self.nspangles, True)
        if name:
            cond = (self.data.name==name)
    
        #If no point is of type name
        if not cond.any():
            return
            
        #Update positions in the intersection reference frame
        # Pandas >= 2.2 is strict about assigning floats into int columns.
        # Intersection-frame coordinates are continuous, so keep them as float dtype.
        for col in ("x_int", "y_int", "z_int"):
            if col not in self.data.columns:
                self.data[col] = np.nan
            if not pd.api.types.is_float_dtype(self.data[col].dtype):
                self.data[col] = self.data[col].astype(float)

        r_ecl = self.data.loc[cond, ["x_ecl", "y_ecl", "z_ecl"]].to_numpy()
        self.data.loc[cond, ["x_int", "y_int", "z_int"]] = (self.M_ecl2int @ (r_ecl - center).T).T
        
        #Center of the object in the observer reference system
        groups = self.data[cond].groupby('name')

        for group_name, group in groups:
            M_equ2ecl = self.M_equ2ecl[group_name]
            c_ecl = np.array(group['center_ecl'].iloc[0])
            c_equ = np.array(group['center_equ'].iloc[0])
            c_int = np.matmul(self.M_ecl2int, c_ecl + np.matmul(M_equ2ecl, c_equ) - center)
            self.data.loc[group.index, "center_int"] = pd.Series([c_int] * len(group), dtype=object, index=group.index)

            # #Pseudo-cylindrical coordinates in the observer system
            r_int = (self.data.loc[group.index, ["x_int","y_int","z_int"]]).to_numpy()
            # Ensure continuous coordinates are float columns for pandas strictness.
            for col in ("rho_int", "az_int", "cosf_int"):
                if col not in self.data.columns:
                    self.data[col] = np.nan
                if not pd.api.types.is_float_dtype(self.data[col].dtype):
                    self.data[col] = self.data[col].astype(float)
            self.data.loc[group.index, ["rho_int","az_int","cosf_int"]] = Science.pcylindrical(r_int - c_int)
        
        #According to distance to intersetcion point generate z_cen_int
        if self.infinite:
            self.data.loc[cond,"z_cen_int"]=-np.inf
        else:
            z_cen_int = np.stack(self.data.loc[cond, "center_int"])[:, 2]
            self.data.loc[cond,"z_cen_int"] = z_cen_int
    
        #Compute distance to intersection of each spangle and the 
        if self.infinite:
            #Distance to all points is assumed infinite
            self.data.loc[cond,"n_int"]=pd.Series([[0,0,1]]*sum(cond),dtype=object)
            self.data.loc[cond,"d_int"]=np.inf
            self.data.loc[cond,"n_int_ecl"]=pd.Series([list(n_int)]*sum(cond),dtype=object)
        else:
            #Distance to origin of coordinates in the int system where the center is located
            r_int = self.data.loc[cond, ["x_int", "y_int", "z_int"]].to_numpy()
            d_int_arr = np.linalg.norm(-r_int, axis=1)
            n_int_arr = -r_int/d_int_arr[:, None]
            self.data.loc[cond, "n_int"] = pd.Series(n_int_arr.tolist(), dtype=object, index=self.data[cond].index)
            self.data.loc[cond, "d_int"] = d_int_arr
            n_int_ecl = (M_int2ecl @ n_int_arr.T).T
            self.data.loc[cond, "n_int_ecl"] = pd.Series(n_int_ecl.tolist(), dtype=object, index=self.data[cond].index)
   
        #Azimuth of the direction of the intersection vector in the tangent plane of the spangle
        wy_ecl = np.stack(self.data.loc[cond, "wy_ecl"])
        wx_ecl = np.stack(self.data.loc[cond, "wx_ecl"])
        n_int_ecl = np.stack(self.data.loc[cond, "n_int_ecl"])
        dot_wy_n = np.sum(wy_ecl * n_int_ecl, axis=1)
        dot_wx_n = np.sum(wx_ecl * n_int_ecl, axis=1)
        if "azim_int" not in self.data.columns:
            self.data["azim_int"] = np.nan
        if not pd.api.types.is_float_dtype(self.data["azim_int"].dtype):
            self.data["azim_int"] = self.data["azim_int"].astype(float)
        self.data.loc[cond, "azim_int"] = np.arctan2(dot_wy_n, dot_wx_n)

        #Update spangles orientations
        ns_ecl = np.stack(self.data.loc[cond, "ns_ecl"])
        ns_int = (self.M_ecl2int @ ns_ecl.T).T
        self.data.loc[cond, "ns_int"] = pd.Series(ns_int.tolist(), dtype=object, index=self.data[cond].index)
        

        #Cosine of the direction of the intersection vector and the normal to the spangle
        # Store cosines as float: pandas >=2 raises LossySetitemError when assigning
        # floats into an int-typed column, so coerce cos_int to float first.
        if "cos_int" not in self.data.columns:
            self.data["cos_int"] = np.nan
        if not pd.api.types.is_float_dtype(self.data["cos_int"].dtype):
            self.data["cos_int"] = self.data["cos_int"].astype(float)

        if self.infinite:
            #In this case n_int is a global variable
            cos_int = np.sum(ns_ecl * n_int, axis=1)
            self.data.loc[cond, "cos_int"] = cos_int
        else:
            #In this case n_int is a per-spangle variable
            n_int_cond = np.stack(self.data.loc[cond, "n_int"])
            cos_int = np.sum(ns_int * n_int_cond, axis=1)
            self.data.loc[cond, "cos_int"] = cos_int


        #Set areas
        self.data.loc[cond,"asp_int"]=self.data.loc[cond,"asp"]
        
        return cond,n_int,d_int

    def _normalize_areas(self):
        """
        Normalizes the projected areas of spangles in intersection, observer, and light-source frames.

        Returns
        -------
        None

        Notes
        -----
        Adjusts `asp_int`, `asp_obs`, and `asp_luz` columns in the `data` DataFrame based on the
        expected geometric area versus the sum of facet areas, considering only face-on spangles.
        """
        """
        Set normalization factor of area projection in Intersection/Observer/Light-Source Frames
        Also corrects for the area projection for flux computation  (Expected Area/Facet-Based Area)

        Attribute Modified:

            asp_int/obs/luz = Projected area of individual Spangle in Intersection/Observer/Light-Source Frame

        Computation:

        - Only face-on Spangles in line-of-sight are taking into account (cos >= 0)

            asp = Area of Individual Spangle
            cos = Cosine between the line-of-sight and each Spangle
            scale = Body Lenght Scale (Radius)
            r_int = Internal Radius (Only for Ring)

        - Normalization Factor by Areas Relation 

            expected_area = Projected Disk Area -> pi*(1 - r_int^2)*(scale^2)*cos
            facets_area = Sum of face-on Spangles area -> sum(asp*cos)

            norm_factor = expected_area/facets_area
            
        - Area Correction by Perspective

            asp_int/obs/luz = asp*norm_factor

        """

        # Over each Body
        for i, name in enumerate(self.name):

            #Only Face-On Spangles
            cond_name = (self.data.name == name)
            cond = cond_name*(self.data.cos_obs >= 0)*(~self.data.hidden)

            #Only for Ring
            r_int = self.spanglers[i].sample.ri if hasattr(self.spanglers[i].sample, 'ri') else 0

            asp = self.data[cond_name].asp.iloc[0]
            scale = self.data[cond_name].scale.iloc[0]
            cos = self.data[cond].cos_obs.iloc[0] if r_int != 0 else 1

            #Normalization
            expected_area = np.pi*(1 - r_int**2)*scale**2*cos
            norm_factor = expected_area/(asp*self.data[cond].cos_obs).sum()

            #Area Correction
            self.data.loc[cond_name, 'asp_int'] *= norm_factor

    
    def _calc_qhulls(self):
        
        """
        Computes convex hulls for spangles in the intersection configuration.

        Returns
        -------
        None

        Notes
        -----
        Populates the `qhulls` attribute with convex hulls for each object, handling cases with holes
        (e.g., rings). Used internally by intersection state updates.
        """
        
        #Convex hulls
        for name in Misc.flatten([self.name]):
    
            self.qhulls[name]=[]
            cond_obj=(self.data.name==name)
            center=list(self.data[cond_obj].center_int.iloc[0])
            zord=min(self.data[cond_obj].z_int)
    
            if (self.data[cond_obj].hidden).sum()==0:
    
                #Convex hull of whole objects
                cond_hull=(cond_obj)&(~self.data[cond_obj].hidden)
                verbose(VERB_SIMPLE,"Hull points (whole object):",sum(cond_hull))
                qhull=Science.get_convexhull(self.data[cond_hull][["x_int","y_int"]])
                vhull=qhull.volume if qhull else 0
                
                self.qhulls[name]+=[dict(
                    name=name,
                    hulltype="cen",
                    center=center,
                    zord=zord,
                    qhull=qhull,
                    vhull=vhull,
                )]
    
            else:
                #Convex hull of objects with a hole (eg. rings)
    
                #Plane of rings
                cond_hidden=(cond_obj)&(self.data[cond_obj].hidden)
                hidden=self.data[cond_hidden][["x_int","y_int","z_int"]].values
                nhidden=len(hidden)
                p1,p2,p3=hidden[0],hidden[int(nhidden/3)],hidden[2*int(nhidden/3)]
                plane=Plane(p1,p2,p3)
    
                #Convex hull of hidden points (the hole)
                cond_hull=(cond_obj)&(self.data[cond_obj].hidden)
                verbose(VERB_SIMPLE,"Hull points (hidden):",sum(cond_hull))
                qhull=Science.get_convexhull(self.data[cond_hull][["x_int","y_int"]])
                vhull=qhull.volume if qhull else 0
    
                self.qhulls[name]+=[dict(
                    name=name,
                    hulltype="hidden",
                    center=center,
                    zord=zord,
                    qhull=qhull,
                    vhull=vhull,
                    plane=plane
                )]
    
                #Convex hull of no hidden points
                cond_hull=(cond_obj)&(~self.data[cond_obj].hidden)
                verbose(VERB_SIMPLE,"Hull points (visible ring):",sum(cond_hull))
                qhull=Science.get_convexhull(self.data[cond_hull][["x_int","y_int"]])
                vhull=qhull.volume if qhull else 0
    
                self.qhulls[name]+=[dict(
                    name=name,
                    hulltype="plane",
                    center=center,
                    zord=zord,
                    qhull=qhull,
                    vhull=vhull,
                    plane=plane
                )]
               
    
    def set_observer(self,nvec=[0,0,1],alpha=0,center=None):
        """
        Sets the positions and orientations of spangles in the observer reference system.

        Parameters
        ----------
        nvec : array-like (3,), optional
            Vector towards the observer in the ecliptic system. Default is [0, 0, 1].
        alpha : float, optional
            Roll angle of the x-axis in the observer system, in radians. Default is 0.
        center : array-like (3,), optional
            Position of the observer in the ecliptic system. If None, assumes infinite distance.
            Default is None.

        Returns
        -------
        None

        Notes
        -----
        Updates the `data` DataFrame with observer coordinates and sets the "visible" state.
        Calls :meth:`set_intersect` internally.
        """
        verbose(VERB_SIMPLE,f"Setting observer")
        
        #Set observer
        cond,self.n_obs,self.d_obs=self.set_intersect(nvec,alpha,center)
        
        #Set properties
        self.alpha_obs=alpha
        self.rqf_obs=Science.spherical(self.n_obs)
        self.center_obs=center.copy() if center else center
        
        self.data.loc[cond,"visible"]=False
        #self.data.loc[cond,SPANGLER_COL_OBS]=self.data.loc[cond,SPANGLER_COL_INT].values
        #"""
        # Pandas is strict about assigning float arrays into int columns.
        # These observer-frame columns can contain continuous floats and/or object-like vectors,
        # so we coerce them to object dtype before bulk assignment.
        for col in SPANGLER_COL_OBS:
            if col not in self.data.columns:
                self.data[col] = pd.Series([None] * len(self.data), dtype=object, index=self.data.index)
            elif self.data[col].dtype != object:
                self.data[col] = self.data[col].astype(object)

        self.data.loc[cond,SPANGLER_COL_OBS]=pd.DataFrame(self.data.loc[cond,SPANGLER_COL_INT].values,
                                                          columns=SPANGLER_COL_OBS,
                                                          index=self.data[cond].index)
        

        # Beta angle computation (Reference Plane rotation angle)
        # Appendix D arXiv:2404.16606v1
        groups = self.data[cond].groupby('name')

        for group_name, group in groups:

            # Normal vector of each spangle
            ns_obs = np.stack(group['ns_obs'].values)

            if group['spangle_type'].iloc[0] == 4: # Ring Spangle
                
                # Cosine of the angle between normal vector and observer vector
                cos_obs = group['cos_obs'].iloc[0]

                # X-Z angle
                sigma = np.arctan2(ns_obs[0,2], ns_obs[0,0])

                # Beta angle
                cos_beta =  np.cos(sigma)/np.sin(np.arccos(cos_obs))
                betas = np.full_like(ns_obs[:,0], fill_value = np.arccos(cos_beta))

                # Check Rotation Direction
                if ns_obs[0,1] < 0:
                    betas = np.pi - betas

            else: # Planetary or Stellar Spangle
                betas = np.arctan(ns_obs[:,1]/ns_obs[:,0])
                betas[ns_obs[:,0]*ns_obs[:,1] < 0] += np.pi

            if "beta_loc" not in self.data.columns:
                self.data["beta_loc"] = pd.Series([None] * len(self.data), dtype=object, index=self.data.index)
            elif self.data["beta_loc"].dtype != object:
                self.data["beta_loc"] = self.data["beta_loc"].astype(object)
            self.data.loc[group.index, "beta_loc"] = pd.Series(betas.tolist(), dtype=object, index=group.index)
        
        #Update states
        self.data.unset=False
        
        #Condition for visibility
        """
        & ! Hidden
        & z_cen_obs+scale < 0: spangle is observable from the observer vantage point-
            (
                | cos_obs > 0: spangle it is towards the observer
                | Spangle type is semitransparent
            )
        """
        cond=    (~self.data.hidden)&    ((self.data.z_cen_obs+self.data.scale)<0)&    (        (self.data.cos_obs>0)|        (self.data.spangle_type.isin(SPANGLES_SEMITRANSPARENT))
        )
        self.data.loc[cond,"visible"]=True
        
    def set_luz(self,nvec=[0,0,1],alpha=0,center=None,name=None):
        """
        Sets the positions and orientations of spangles in the light-source reference system.

        Parameters
        ----------
        nvec : array-like (3,), optional
            Vector towards the light source in the ecliptic system. Default is [0, 0, 1].
        alpha : float, optional
            Roll angle of the x-axis in the light-source system, in radians. Default is 0.
        center : array-like (3,), optional
            Position of the light source in the ecliptic system. If None, assumes infinite distance.
            Default is None.
        name : str, optional
            Name of the specific object to apply the light direction to. If None, applies to all.
            Default is None.

        Returns
        -------
        None

        Notes
        -----
        Updates the `data` DataFrame with light-source coordinates and sets "illuminated" and
        "transmit" states. Requires :meth:`set_observer` to be called first for accurate transmission.
        """
       
        verbose(VERB_SIMPLE,f"Setting light-source")
     
        #Set intersect of all points in order to prepare the update luz
        cond,self.n_luz,self.d_luz=self.set_intersect(nvec,alpha,center,name=None) 
        verbose(VERB_SIMPLE,f"Number of points: {sum(cond)}")
        
        #Depending on body choose which spangles to change
        cond = pd.Series([True] * self.nspangles, index=self.data.index)
        if name:
            cond=(self.data.name==name)
        
        #Set the light source direction in spherical coordinates
        self.rqf_luz=Science.spherical(self.n_luz)
        
        #Set the default value of the states to change in False
        self.data.loc[cond,"illuminated"]=False
        self.data.loc[cond,"transmit"]=False
        
        #Conditions
        #self.data.loc[cond,SPANGLER_COL_LUZ]=deepcopy(self.data.loc[cond,SPANGLER_COL_INT].values)
        # Same rationale as in set_observer: allow float/object assignment safely.
        for col in SPANGLER_COL_LUZ:
            if col not in self.data.columns:
                self.data[col] = pd.Series([None] * len(self.data), dtype=object, index=self.data.index)
            elif self.data[col].dtype != object:
                self.data[col] = self.data[col].astype(object)
        self.data.loc[cond,SPANGLER_COL_LUZ]=pd.DataFrame(self.data.loc[cond,SPANGLER_COL_INT].values,
                                                          columns=SPANGLER_COL_LUZ,
                                                          index=self.data[cond].index)
        
        #Set relative azimuth [-pi,pi]
        azim_obs_luz = (self.data.loc[cond,"azim_obs"]-self.data.loc[cond,"azim_luz"]).to_numpy(dtype=float)
        # pi Shift and domain in [-pi,pi]
        if "azim_obs_luz" not in self.data.columns:
            self.data["azim_obs_luz"] = np.nan
        if not pd.api.types.is_float_dtype(self.data["azim_obs_luz"].dtype):
            self.data["azim_obs_luz"] = self.data["azim_obs_luz"].astype(float)
        self.data.loc[cond,"azim_obs_luz"] = np.arctan2(np.sin(azim_obs_luz + np.pi), np.cos(azim_obs_luz + np.pi))

        #Update states
        self.data.loc[cond,"unset"]=False
        
        #Condition for illumination
        """
        & ! Hidden
        & z_cen_luz+scale < 0: spangle is in front of the light-source.
            (
                | geometry = circle : 2d spangles are always illuminated
                | spangle_type = stellar: stellar spangles are always illuminated
                | cos_luz > 0: spangle it is towards the light source
            )
        """
        cond=    cond&    (~self.data.hidden)&    ((self.data.z_cen_luz+self.data.scale)<0)&    (        (self.data.geometry==SAMPLER_GEOMETRY_CIRCLE)|        (self.data.cos_luz>0)|        (self.data.spangle_type==SPANGLE_STELLAR)|        (self.data.spangle_type.isin(SPANGLES_SEMITRANSPARENT))
        )
        self.data.loc[cond,"illuminated"]=True
    
        #Conditions for transmission:
        """
        & No hidden
        (
            & Spangle type is semitransparent
            & cos_obs . cos_luz < 0: observer and light source are in opposite sides
        )
        
        ATTENTION: TRANSMISSION IS ONLY PROPERLY SET IF OBSERVER HAVE BEEN PREVIOUSLY SET.
        """
        cond=    cond&    (~self.data.hidden)&    (     (self.data.spangle_type.isin(SPANGLES_SEMITRANSPARENT))&     ((self.data.cos_luz*self.data.cos_obs)<=0)
        )
        self.data.loc[cond,"transmit"]=True
        
    def plot2d(self,
                coords="obs",
                center_at=None,
                include=[],
                exclude=[],
                axis=True,
                fsize=5,
                newfig=True,
                show_azim=False,
                highlight=None,
                maxval=None,
                bgdark = True,
                # temperature = False,
                ):
        """
        Visualizes the spangles in a 2D plot.

        Parameters
        ----------
        coords : str, optional
            Coordinate system to plot. Options are "equ", "ecl", "int", "obs", "luz". Default is "obs".
        center_at : str, optional
            Name of the object to center the plot around. Default is None.
        include : list of str, optional
            Names of objects to exclusively include in the plot. Default is an empty list.
        exclude : list of str, optional
            Names of objects to exclude from the plot. Default is an empty list.
        axis : bool, optional
            If True, displays coordinate axes. Default is True.
        fsize : float, optional
            Size of the figure in inches (square figure). Default is 5.
        newfig : bool, optional
            If True, creates a new figure. If False, reuses the existing one (for animations).
            Default is True.
        show_azim : bool, optional
            If True, displays azimuth directions for observer and light source. Default is False.
        highlight : tuple, optional
            Tuple of (condition, scatter_options) to highlight specific spangles. Default is None.
        maxval : float, optional
            Maximum value for plot range. If None, computed automatically. Default is None.
        bgdark : bool, optional
            If True, uses a dark background. Default is True.

        Returns
        -------
        tuple
            - x_cen : float
                X-coordinate of the plot center.
            - y_cen : float
                Y-coordinate of the plot center.

        Notes
        -----
        Sets the `fig2d` and `ax2d` attributes. Spangle colors and sizes reflect their states.
        """
        
        #Global properties of the plot
        if bgdark:
            bgcolor='k'
            textcolor = 'w'
        else:
            bgcolor='w'
            textcolor = 'k'

        fig_factor=fsize/5

        #Create figure and axes
        if "fig2d" not in self.__dict__ or newfig:
            fig=plt.figure(figsize=(fsize,fsize))
            fig.patch.set_facecolor(bgcolor)
            ax=fig.add_subplot(111,facecolor=bgcolor)

            #Keep figure and axe
            self.fig2d=fig
            self.ax2d=ax

        #Convert list 
        include_string=[]
        for body in include:
            if isinstance(body,PrynglesCommon):
                include_string+=[body.name]
            else:
                include_string+=[body]
        include=include_string

        exclude_string=[]
        for body in exclude:
            if isinstance(body,PrynglesCommon):
                exclude_string+=[body.name]
            else:
                exclude_string+=[body]
        exclude=exclude_string
        
        if isinstance(center_at,PrynglesCommon):
            center_at=center_at.name

        #Plot only a set of objects
        if len(include)>0:
            
            #List of spanglers names
            exclude=list(self.data.name.unique())
            
            for name in include:
                if name not in exclude:
                    raise ValueError(f"Spangler '{name}' not among available spanglers ({exclude})")
                else:
                    exclude.remove(name)
                    
            #Center at the first object in the list
            center_at=include[0]

        #Center of plot
        cond=(self.data.name==center_at)
        x_cen,y_cen,z_cen=self.data[cond][[f"x_{coords}",f"y_{coords}",f"z_{coords}"]].mean() if sum(cond)>0 else np.array([0,0,0])

        #Select plotting bodies
        cond_included=(~self.data.hidden)&(~self.data.name.isin(exclude))
        num_included=sum(cond_included)
        if num_included==0:
            raise AssertionError(f"No body remain after removing {exclude}")
        data=self.data[cond_included]
        
        #Calculate range of plot
        cond_maxval=(~data.hidden)&(~data.name.isin(exclude))
        cond_maxval=cond_maxval if sum(cond_maxval)>0 else np.array([True]*num_included)        
        if not maxval:
            maxval=1.2*np.abs(np.array(data[cond_maxval][[f"x_{coords}",f"y_{coords}"]])-np.array([x_cen,y_cen])).max()

        #Function to determine the size of the spangles
        size_factor=1/2.5
        size_points=lambda dsp,cos_obs:size_factor*(dsp[cond])*abs(cos_obs)**0.5
        
        ##########################################################
        #Plotting properties according to state
        ##########################################################
        #Default colors and sizes
        colors=np.array(['#000000']*num_included)
        sizes=np.array([0.0]*num_included)
        marker='o'
        #All
        """
        cond=[True]*len(data)
        sizes[cond]=0
        """

        #Illuminated
        cond=(data.visible)&(data.illuminated)
        verbose(VERB_SIMPLE,f"Visible and illuminated: {cond.sum()}")
        colors[cond]=[Plot.rgb([SPANGLE_COLORS[stype][0],
                                SPANGLE_COLORS[stype][1]*min((cos_luz*cos_obs+0.3),1),
                                SPANGLE_COLORS[stype][2]],
                                to_hex=True) for stype,cos_luz,cos_obs in zip(data[cond].spangle_type,
                                                                            abs(data[cond].cos_luz),
                                                                            abs(data[cond].cos_obs))
                        ] #Object color
        sizes[cond]=size_points(data.dsp[cond],data.cos_obs[cond])

        #Not illuminated
        cond=(data.visible)&(~data.illuminated)
        verbose(VERB_SIMPLE,f"Visible and not illuminated: {cond.sum()}")
        colors[cond]=Plot.rgb(SPANGLES_DARKNESS_COLOR,to_hex=True)
        sizes[cond]=size_points(data.dsp[cond],data.cos_obs[cond])
        
        #In shadow
        cond=(data.visible)&(data.shadow)
        verbose(VERB_SIMPLE,f"Visible and not illuminated: {cond.sum()}")
        colors[cond]=Plot.rgb(SHADOW_COLOR_LUZ,to_hex=True)
        sizes[cond]=size_points(data.dsp[cond],data.cos_obs[cond])

        if coords!="obs":
            #Not visible
            cond=(~data.visible)&(data[f"z_{coords}"]>0)
            colors[cond]=Plot.rgb(SHADOW_COLOR_OBS,to_hex=True)
            sizes[cond]=size_points(data.dsp[cond],data.cos_obs[cond])

        #Transmitting
        cond=(data.visible)&(data.transmit)&(data.illuminated)
        verbose(VERB_SIMPLE,f"Visible, illuminated and transmitting: {cond.sum()}")
        colors[cond]=[Plot.rgb([SPANGLE_COLORS[stype][0],
                                SPANGLE_COLORS[stype][1]*min((cos_luz*cos_obs+0.3),1)/2,
                                SPANGLE_COLORS[stype][2]],
                                to_hex=True) for stype,cos_luz,cos_obs in zip(data[cond].spangle_type,
                                                                            abs(data[cond].cos_luz),
                                                                            abs(data[cond].cos_obs))
                        ] #Object color
        sizes[cond]=size_points(data.dsp[cond],data.cos_obs[cond])
        
        ##########################################################
        #Dot sizes scaled according to figure size
        ##########################################################
        ppd=72./self.ax2d.figure.dpi
        trans=self.ax2d.transData.transform
        dot_size=lambda x:int(((trans((1,x/maxval))-trans((0,0)))*ppd)[1]**2)
        
        ##########################################################
        #Plotting properties according to state
        ##########################################################
        st=[max(dot_size(s),0.1) if s>0 else 0 for s in sizes]
        sargs=dict(c=colors,s=st,marker=marker,zorder=-100)
        self.ax2d.scatter(data[f"x_{coords}"]-x_cen,data[f"y_{coords}"]-y_cen,**sargs)
        
        #Ranges
        self.ax2d.set_xlim(-maxval,maxval)
        self.ax2d.set_ylim(-maxval,maxval)
        
        factor=1
        xmin,xmax=factor*np.array(list(self.ax2d.get_xlim()))
        ymin,ymax=factor*np.array(list(self.ax2d.get_ylim()))
        
        ##########################################################
        #Show azim
        ##########################################################
        if show_azim:
            #Choose which directions to show
            cond=(self.data["cos_"+coords]>=0)&(~self.data.hidden)&(cond_included)&(self.data["spangle_type"]!=SPANGLE_STELLAR)

            #Options of arrows showing direction
            quiver_args=dict(scale=15,scale_units='width',
                                width=0.005,alpha=0.6,zorder=+1000,headwidth=0)
            
            #Quiver plot of azimuth for light
            azx=[mh.cos(x) for x in self.data[cond].azim_luz]
            azy=[mh.sin(x) for x in self.data[cond].azim_luz]
            
            
            self.ax2d.quiver(self.data[cond]["x_"+coords]-x_cen,self.data[cond]["y_"+coords]-y_cen,
                                azx,azy,color='m',label="Az.luz",**quiver_args)

            #Quiver plot of azimuth for observer
            azx=[mh.cos(x) for x in self.data[cond].azim_luz]
            azy=[mh.sin(x) for x in self.data[cond].azim_luz]
            self.ax2d.quiver(self.data[cond]["x_"+coords]-x_cen,self.data[cond]["y_"+coords]-y_cen,
                                azx,azy,color='y',label="Az.obs",**quiver_args)

            #Quiver plot of elevation for light
            tx=np.sqrt(1-self.data[cond].cos_int**2).values
            ty=self.data[cond].cos_int.values
            self.ax2d.quiver(self.data[cond]["x_"+coords]-x_cen,self.data[cond]["y_"+coords]-y_cen,
                                tx,ty,color='c',label="Elev.luz",**quiver_args)

            #Legend decoration
            leg=self.ax2d.legend(loc='lower right',facecolor=bgcolor,ncol=3,prop={'size':8},
                                    bbox_to_anchor=(0.5, -0.05, 0.5, 0.5))
            frame=leg.get_frame()
            frame.set_edgecolor(bgcolor)
            for text in leg.get_texts():
                text.set_color(textcolor)
            axis=False

        ##########################################################
        #Highlight spangles
        ##########################################################
        if highlight:
            if len(highlight)<2:
                raise AssertionError("Highlight should include conditions and scatter options")
            
            def_args_scatter=dict(c=textcolor,s=0.1,marker='*')
            cond_highlight,args_scatter=highlight
            def_args_scatter.update(args_scatter)
            self.ax2d.scatter(self.data[cond_highlight&cond_included]["x_"+coords]-x_cen,
                                self.data[cond_highlight&cond_included]["y_"+coords]-y_cen,
                                **def_args_scatter)
            
        ##########################################################
        #Show axis and other labels
        ##########################################################
        if newfig and axis:
            self.ax2d.plot([xmin,xmax],[0,0], textcolor + '-',alpha=0.3)
            self.ax2d.plot([0,0],[ymin,ymax], textcolor + '-',alpha=0.3)
            self.ax2d.text(xmax,0,fr"$x_{{{coords}}}$",color=textcolor,alpha=0.5,fontsize=8*fig_factor)
            self.ax2d.text(0,ymax,fr"$y_{{{coords}}}$",color=textcolor,alpha=0.5,fontsize=8*fig_factor)

            #Scale
            center_text=""
            if center_at:
                center_text=f", Center at '{center_at}'"
            self.ax2d.text(0,0,f"Axis scale: {maxval*factor:.2g}{center_text}",
                        fontsize=8*fig_factor,color=textcolor,
                        transform=self.ax2d.transAxes)

        ##########################################################
        #Decorate plot
        ##########################################################
        if newfig:
            #Title
            label_obs=""
            lamb=0
            phi=0
            if coords=="obs":
                lamb=self.rqf_obs[1]*Consts.rad
                phi=self.rqf_obs[2]*Consts.rad
                coords_label=f"($\lambda$,$\\beta$) : ({lamb:.1f}$^\circ$,{phi:.1f}$^\circ$)"
            elif coords=="luz":
                lamb=self.rqf_luz[1]*Consts.rad
                phi=self.rqf_luz[2]*Consts.rad
                coords_label=f"($\lambda$,$\\beta$) : ({lamb:.1f}$^\circ$,{phi:.1f}$^\circ$)"
            elif coords=="int":
                lamb=self.rqf_int[1]*Consts.rad
                phi=self.rqf_int[2]*Consts.rad
            coords_label=f"($\lambda$,$\\beta$) : ({lamb:.1f}$^\circ$,{phi:.1f}$^\circ$)"

            if coords=="ecl":
                coords_label=""

            label_obs=f"{coords} {coords_label}"
            self.ax2d.text(0.5,1.01,f"{label_obs}",
                            transform=self.ax2d.transAxes,ha='center',
                            color=textcolor,fontsize=10*fig_factor)

            self.ax2d.axis("off")
            Plot.pryngles_mark(self.ax2d)
        
        ##########################################################
        #Adjust sizes
        ##########################################################
        self.ax2d.axis("equal")
        self.fig2d.tight_layout()
        
        return x_cen,y_cen
            
    
    def update_intersection_state(self,excluded=[],included=[]):
        """
        Updates the intersection states of spangles.

        Parameters
        ----------
        excluded : list of str, optional
            Names of objects to exclude from intersection calculations. Default is an empty list.
        included : list of str, optional
            Names of objects to include in intersection calculations. If empty, includes all not excluded.
            Default is an empty list.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If no intersection vantage point has been set (i.e., `qhulls` is empty) or if all objects
            are excluded.

        Notes
        -----
        Updates the "intersect" state and related fields in the `data` DataFrame based on convex hulls.
        """   
        #Update qhulls using the latest intersection state
        self._calc_qhulls()
        
        #Check if at least one qhull has been computed
        if len(self.qhulls) == 0:
            raise AssertionError("You must set an intersection vantage point.")
    
        #List of objects in spangler
        names=list(Misc.flatten([self.name]))
        
        if len(included):
            excluded=[n for n in names if n not in included]
        verbose(VERB_SIMPLE,f"Exclusion list: {excluded}")
        
        #Objects included when performing intersection calculation
        cond_included=self.data.name.apply(lambda x:x not in excluded)
        if cond_included.sum()==0:
            raise AssertionError("You have excluded all objects when calculating intersetions.")
        else:
            verbose(VERB_SIMPLE,f"Points included in calculation: {cond_included.sum()}")
        
        #Under the current circumstances all this spangles are intersecting 
        cond=    (~self.data.hidden)&    (     (self.data.cos_int>0)|     (self.data.spangle_type.isin(SPANGLES_SEMITRANSPARENT))    )&    (cond_included)
        
        self.data.loc[cond,"intersect"]=True
        self.data.hidden_by_int=""
        self.data.transit_over_int=""
            
        #Loop over objects producing intersection
        for name in names:
            
            #If body is excluded
            if name in excluded:
                verbose(VERB_SIMPLE,f"Skipping {name} from intersection computation")
                continue
            
            #Points in present body
            cond=(self.data.name==name)
            geometry=self.data[cond].geometry.iloc[0]
            scale=self.data[cond].scale.iloc[0]
            
            #If this body is not in the field-of-view, avoid computation
            z_cen_int=self.data[cond].z_cen_int.iloc[0]
            scale=self.data[cond].scale.iloc[0]
            if (z_cen_int+scale)>=0:
                continue
    
            #By default for all objects
            inhull_not_in_hole=[True]
            
            verbose(VERB_SIMPLE,f"Calculating intersections for '{name}'")
            
            #Hull area of the body
            ahull=0
            for i,hull in enumerate(self.qhulls[name]):
                
                qhull=hull["qhull"]
                if qhull is None:
                    verbose(VERB_SIMPLE,f"No hull for '{name}'")
                    continue
                vhull=hull["vhull"]
                
                htype=hull["hulltype"]
                xcen,ycen,zcen=hull["center"]
                zord=hull["zord"]
                
                verbose(VERB_SIMPLE,f"Hull {i+1} for '{name}' of type '{htype}'")
    
                #Evaluate conditions
                inhull = np.asarray(Science.points_in_hull(self.data[["x_int","y_int"]], qhull), dtype=bool)
                inhull = inhull & np.asarray((~cond) & (cond_included), dtype=bool)
                below=np.array([False]*self.nspangles)
                above=np.array([False]*self.nspangles)
                
                if htype=="hidden":
    
                    #Holes
                    inhull_not_in_hole=(~inhull)
                    verbose(VERB_SIMPLE,f"Points outside hidden hull for '{name}': {sum(inhull_not_in_hole)}")
                    hull["notinhole"]=sum(inhull_not_in_hole)
    
                    continue
                    
                else:
                    #Body
                    verbose(VERB_SIMPLE,f"Points not in hole for '{name}:{htype}': {sum(inhull_not_in_hole)}")
        
                    #Area of the body
                    ahull+=vhull
                    
                    #Spangles to evaluate
                    cond_vis=(self.data.cos_int>0)|(self.data.spangle_type.isin(SPANGLES_SEMITRANSPARENT))
                    cond_int=(~self.data.hidden)&(self.data.name!=name)&(cond_vis)
    
                    if htype=="cen":
                        not_in_hole = np.asarray(inhull_not_in_hole, dtype=bool)
                        cond_int_arr = np.asarray(cond_int, dtype=bool)
                        z_int = self.data["z_int"].to_numpy(dtype=float)
                        below = not_in_hole & inhull & cond_int_arr & (z_int <= zcen)
                        above = not_in_hole & inhull & cond_int_arr & (z_int > zcen)
                        
                    elif htype=="plane":
                        #Not in hole, inhull, not hidden, not in object and intersect
                        cond_full = np.asarray(inhull_not_in_hole, dtype=bool) & inhull & np.asarray(cond_int, dtype=bool)
                        verbose(VERB_SIMPLE,"Fulfilling all conditions:",sum(cond_full))
                        
                        plane=hull["plane"]
                        below[cond_full]=[plane.is_below(r,[0,0,1]) for r in self.data[cond_full][["x_int","y_int","z_int"]].values]
                        above[cond_full]=~below[cond_full]
                    else:
                        raise ValueError("Type of hull '{htype}' not recognized")
                
                #Store information
                verbose(VERB_SIMPLE,f"Points in hull for '{name}:{htype}': {sum(inhull)}")
                verbose(VERB_SIMPLE,f"Points below '{name}:{htype}': {sum(below)}")
                
                #Set visibility
                self.data.loc[below,"intersect"]=False
                self.data.loc[below,"hidden_by_int"]=self.data.loc[below,"hidden_by_int"]+f"{name}:{zord:.3e}&"
                self.data.loc[below,"occult"] = True
                    
                #Compute distance to center for transiting spangles
                self.data.loc[above,"string_int"]=[f"{name}:{zord:.3e}:{((r[0]-xcen)**2+(r[1]-ycen)**2)**0.5/scale:.3e}&"                                                for r in self.data[above][["x_int","y_int"]].values]
                self.data.loc[above,"transit_over_int"]=self.data.loc[above,"transit_over_int"]+self.data.loc[above,"string_int"]
                self.data.loc[above,"transit"] = True
                
                hull["inhull"]=sum(inhull)
                hull["below"]=sum(below)
                
            #Correct areas
        # self._normalize_areas()
            # if geometry != SAMPLER_GEOMETRY_CIRCLE:
            #     cond=(self.data.name==name)&(self.data.cos_int>=0)&(~self.data.hidden)
                
            #     #This what the sum actually is
            #     ahull_expected=(self.data.loc[cond,"asp_int"]*abs(self.data.loc[cond,"cos_int"])).sum()
            #     #This is the expected value
            #     ahull=np.pi*scale**2
                
            #     #Normalization factor to get sum = ahull
            #     norma=ahull/ahull_expected
                
            #     #Final area
            #     self.data.loc[cond,"asp_int"]*=norma
        
    def update_visibility_state(self):
        """
        Updates visibility states and related variables of spangles.

        Returns
        -------
        None

        Notes
        -----
        Calls :meth:`update_intersection_state` and updates "visible", "hidden_by_obs", and
        "transit_over_obs" in the `data` DataFrame.
        """
        self.update_intersection_state()
        self.data[SPANGLER_COL_OBS]=self.data[SPANGLER_COL_INT]
        
        self.data.hidden_by_obs=self.data.hidden_by_obs+self.data.hidden_by_int
        self.data.transit_over_obs=self.data.transit_over_obs+self.data.transit_over_int
        self.data.visible=self.data.visible&self.data.intersect
    
    def update_illumination_state(self,excluded=[],included=[]):
        """
        Updates illumination states and related variables of spangles.

        Parameters
        ----------
        excluded : list of str, optional
            Names of objects to exclude from illumination calculations. Default is an empty list.
        included : list of str, optional
            Names of objects to include in illumination calculations. If empty, includes all not excluded.
            Default is an empty list.

        Returns
        -------
        None

        Notes
        -----
        Calls :meth:`update_intersection_state` and updates "illuminated", "shadow", and related
        fields in the `data` DataFrame.
        """
        self.update_intersection_state(excluded,included)
        self.data[SPANGLER_COL_LUZ]=self.data[SPANGLER_COL_INT]
        
        self.data.hidden_by_luz=self.data.hidden_by_luz+self.data.hidden_by_int
        self.data.transit_over_luz=self.data.transit_over_luz+self.data.transit_over_int
        
        #Update illumination
        self.data.illuminated=self.data.illuminated&self.data.intersect
        
        #Not intersected and spangles in the direction of the light-source are for sure shadowed spangles
        #Condition for visibility
        """
        | shadow
        & ~intersect : spangle does not intersect
            (
                | cos_obs > 0: spangle it is towards the observer
                | Spangle type is semitransparent
            )
        """
        self.data.shadow=    (self.data.shadow)|    (
            (~self.data.intersect)&\
            (
                (self.data.cos_int>0)|
                (self.data.spangle_type.isin(SPANGLES_SEMITRANSPARENT))
            )
        )
        
        #Stellar spangles are always illuminated
        cond=(self.data.spangle_type==SPANGLE_STELLAR)
        self.data.loc[cond,"illuminated"]=True
        
        #In stellar spangles cos_luz = cos_obs for not having strange visual representations
        self.data.loc[cond,"cos_luz"]=self.data.loc[cond,"cos_obs"]