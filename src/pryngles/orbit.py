##################################################################
#                                                                #
# .#####...#####...##..##..##..##...####...##......######...####..#
# .##..##..##..##...####...###.##..##......##......##......##.....#
# .#####...#####.....##....##.###..##.###..##......####.....####..#
# .##......##..##....##....##..##..##..##..##......##..........##.#
# .##......##..##....##....##..##...####...######..######...####..#
# ................................................................#
#                                                                #
# PlanetaRY spanGLES                                             #
#                                                                #
##################################################################
# License http://github.com/seap-udea/pryngles-public            #
##################################################################

# --------------------------------------------------
# External required packages
# --------------------------------------------------

import numpy as np
import rebound as rb
from anytree import NodeMixin, RenderTree

from pryngles import *


# --------------------------------------------------
# Class Orbody
# --------------------------------------------------
class Orbody(PrynglesCommon, NodeMixin):
    """
    Represents an orbital body in a hierarchical N-body system.

    Parameters
    --------------
    name : `str`
        Name of the body.
    parent : :data:`~ orbit.Orbody`
        Parent body in the orbital hierarchy. Defaults to None.
    **elements : `float`
        Orbital elements as keyword arguments. Must be valid properties defined in
        :data:`~ consts.REBOUND_ORBITAL_PROPERTIES`.

    Raises
    -------------
    ValueError
        If an invalid orbital element is provided.
    """

    def __init__(self, name=None, parent=None, **elements):

        # Basic attributes
        self.name = "body" if name is None else name
        self.parent = parent

        # Add initial elements to attributes
        for element in elements:
            if element in REBOUND_ORBITAL_PROPERTIES:
                self.__dict__[element] = elements[element]
            else:
                raise ValueError(f"Element {element} not identified.")
        self.elements = elements

    def show_tree(self):
        print(RenderTree(self))


# --------------------------------------------------
# Class Orbit
# --------------------------------------------------
class Orbit(PrynglesCommon):
    """
    Represents a two-body orbital system, interfacing with the ``rebound`` framework.

    This class constructs a hierarchical N-body system by combining two-body subsystems,
    allowing for simulation of complex orbital dynamics.

    Parameters
    -----------------
    m1-m2 : `float`, :data:`~ orbit.Orbit`
        | If ``float``, it represents the mass of the body.
        | If :data:`~ orbit.Orbit` instance, it represents a children system,
          i.e., a 2-body subsystem by the hierarchical N-body approximation.

    elements : `dict`
        Dictionary with the orbital elements provided for the second body.
        Valid orbital elements are defined in :any:`const.REBOUND_ORBITAL_PROPERTIES`

    R-V : `np.array(3)`
        Cartesian components of center of mass position/velocity vector | **Default** = `[0, 0, 0]`

    Attributes
    --------------
    sim : :any:`rebound.Simulation`
        Rebound simulation corresponding to all particles in the system, i.e., the hierarchical N-body system.
        It contains all the rebound attributes associated to a simulation
    ORBIT_SIMULATIONS : `list`
        List containing all the 2-body susbsytem rebound simulation from wich is constructed the global system

    Raises
    --------------
    ValueError
        If **m1, m2** parameter has a non-valid object type.
    ValueError
        If **elements** has a non-valid parameter from :any:`consts.REBOUND_ORBITAL_PROPERTIES`
    """

    # --------------------------------------------------
    # Bassic methods
    # --------------------------------------------------
    ORBIT_SIMULATIONS = []

    def __init__(self, name=None, units=None, m1=1, m2=1, R=np.array([0, 0, 0]), V=np.array([0, 0, 0]), **elements):

        # System
        self.name = "system" if name is None else name

        # Global simulation
        self.sim = rb.Simulation()
        if units:
            self.units = units
        else:
            self.units = ["au", "msun", "yr2pi"]
        self.sim.units = self.units

        # Particles
        self.p1 = m1
        self.p2 = m2

        # Periods
        self.Ps = []

        # Check first system
        qmixed = False
        if isinstance(self.p1, Orbit):
            self.m1 = self.p1.Mtot
            qmixed = True
        elif isinstance(self.p1, float) or isinstance(self.p1, int):
            self.m1 = self.p1
        else:
            raise ValueError(
                f"Type of first componente ({type(m1)}) not recognized.  It should be a float or an Orbit instance."
            )

        # Check second system
        if isinstance(self.p2, Orbit):
            self.m2 = self.p2.Mtot
            qmixed = True
        elif isinstance(self.p2, float) or isinstance(self.p2, int):
            self.m2 = self.p2
        else:
            raise ValueError(
                f"Type of Second component ({type(m2)}) not recognized.  It should be a float or an Orbit instance."
            )

        if not qmixed and (sum(R) != 0 or sum(V) != 0):
            raise ValueError("You cannot provide a center of mass position and velocity for a non-mixed system.")

        # Total mass
        self.Mtot = self.m1 + self.m2

        # Add initial elements to attributes
        for element in elements:
            if element in REBOUND_ORBITAL_PROPERTIES:
                self.__dict__[element] = elements[element]
            else:
                raise ValueError(f"Element {element} not identified.")

        # Update states
        self._update_states(R, V)

    def _update_states(self, R=np.array([0, 0, 0]), V=np.array([0, 0, 0])):
        """Updates the position and velocity states of the bodies in the system.

        Parameters
        ------------
        R-V : `np.array(3)`
            Center of mass position/velocity vector | **Defaults** =  `[0, 0, 0]`
        """
        # Create rebound options
        self._rb_options = {k: v for k, v in self.__dict__.items() if k in REBOUND_ORBITAL_PROPERTIES}
        self._rb_options.update(dict(m=0))

        # Create rebound simulation
        sim = rb.Simulation()
        sim.units = self.units
        sim.add(m=self.Mtot)
        sim.add(**self._rb_options)

        # Relative vector
        self.r = np.array(sim.particles[1].xyz)
        self.v = np.array(sim.particles[1].vxyz)
        del sim

        # Calculate positions of components
        self.r1 = R - self.m2 / self.Mtot * self.r
        self.v1 = V - self.m2 / self.Mtot * self.v

        if isinstance(self.p1, Orbit):
            self.p1._update_states(self.r1, self.v1)

        self.r2 = R + self.m1 / self.Mtot * self.r
        self.v2 = V + self.m1 / self.Mtot * self.v
        if isinstance(self.p2, Orbit):
            self.p2._update_states(self.r2, self.v2)

        # Create a simulation of this system
        self.sub_sim = rb.Simulation()
        self.sub_sim.units = self.units
        self.sub_sim.add(
            m=self.m1, x=self.r1[0], y=self.r1[1], z=self.r1[2], vx=self.v1[0], vy=self.v1[1], vz=self.v1[2]
        )
        self.sub_sim.add(
            m=self.m2, x=self.r2[0], y=self.r2[1], z=self.r2[2], vx=self.v2[0], vy=self.v2[1], vz=self.v2[2]
        )

        Orbit.ORBIT_SIMULATIONS += [self.sub_sim]

    def calculate_orbit(self, sim=None):
        """Assemble a :any:`rebound.Simulation` Hierarchical N-body system from clustering 2-body systems.

        Parameters
        -------------
        sim: :any:`rebound.Simulation`
            Main simulation to assemble the hierarchical N-body system.
            This is used for recursion purposes | **Default** = `None`

        Return
        -----------
        :
            self: :data:`~ orbit.Orbit`
                Class object with an assembled simulation attribute

        Examples
        --------------------
        >>> # This is a Star-Planet basic system
        >>> orbit = pr.Orbit(m1 = 1, m2 = 1e-3, a = 0.5, e = 0.4)
        >>> orbit.calculate_orbit()
        >>> orbit.sim.move_to_com()
        >>>
        >>> # Here, the orbit plot
        >>> rb.OrbitPlot(orbit.sim)

        .. image:: images/planet_moon.png
            :align: center

        >>> # For a more complex system, you can cluster various subsystems
        >>> system1 = pr.Orbit(m1 = 1, m2 = 0.1, a = 1, e = 0.4)
        >>> system2 = pr.Orbit(m1 = 1, m2 = 0.1, a = 1, e = 0.7)
        >>>
        >>> # This is a way you can joint the above systems
        >>> system = pr.Orbit(m1 = system1, m2 = system2, a = 5, e = 0)
        >>> system.calculate_orbit()
        >>> system.sim.move_to_com()

        .. image:: images/two_systems.png
            :align: center
        """
        if sim is None:
            sim = self.sim

        if isinstance(self.p1, Orbit):
            self.p1.calculate_orbit(sim)
        else:
            sim.add(m=self.m1, x=self.r1[0], y=self.r1[1], z=self.r1[2], vx=self.v1[0], vy=self.v1[1], vz=self.v1[2])

        if isinstance(self.p2, Orbit):
            p = self.p2.calculate_orbit(sim)
        else:
            sim.add(m=self.m2, x=self.r2[0], y=self.r2[1], z=self.r2[2], vx=self.v2[0], vy=self.v2[1], vz=self.v2[2])

        for p in sim.particles[1:]:
            self.Ps += [p.P]

        return self

    def get_states(self):
        """Get the state vector for positions and velocities of particles in the system

        Returns
        ----------
        :
            sim : :any:`rebound.Simulation`
                Assembled simulation of the system
            states: `list`
                List of dictionaries having the state vector :math:`[x, y, z, v_x, v_y, v_z]` of
                each particle.

        Examples
        -------------
        >>> import rebound as rb
        >>>
        >>> # Create a basic Planet-Moon system
        >>> orbit = pr.Orbit(m1 = 1, m2 = 1e-3, a = 0.5, e = 0.4)
        >>> orbit.calculate_orbit()
        >>> orbit.sim.move_to_com()
        >>>
        >>> # Get the state vector for each particle
        >>> sim, particle_states = orbit.get_states()
        >>>
        >>> particle_states
        [{'m': 1.0,
        'x': -0.0002997002997002997, 'y': 0.0, 'z': 0.0,
        'vx': 0.0, 'vy': -0.002159167585437652, 'vz': 0.0},
        {'m': 0.001,
        'x': 0.2997002997002997, 'y': 0.0, 'z': 0.0,
        'vx': 0.0, 'vy': 2.1591675854376517, 'vz': 0.0}])
        >>>
        >>> rb.OrbitPlot(sim)

        .. image:: images/planet_moon.png
            :align: center
        """
        states = []
        for p in self.sim.particles:
            states += [dict(m=p.m, x=p.x, y=p.y, z=p.z, vx=p.vx, vy=p.vy, vz=p.vz)]
        return self.sim, states


# --------------------------------------------------
# Class OrbitUtil
# --------------------------------------------------
class OrbitUtil(PrynglesCommon):
    """
    General Celestial Mechanics Utilities Class.
    It provides static methods for building and manipulating hierarchical N-body systems.
    """

    def build_tree(root):
        """
        Construct the Orbital Tree for the hierarchical N-body system.
        It is a recursively method for identifying the children bodies from a primary body,
        excluding bodies of kind :data:`~ body.Ring`.

        Parameters
        -------------
        root : :data:`~ body.Body`
            The main body in the System. It corresponds to a :data:`~ body.Star` object, usually

        Returns
        --------------
        :
            orbital_tree : `list`
                Nested list that represents the hierarchical structure of the tree.
                This tree can be used to draw an organizational chart like the one shown above.

        Examples
        -------------
        >>> # Let's create a Star-Planet-Moon like system
        >>>
        >>> # This is the main (root) body in our system
        >>> star = pr.Star()
        >>>
        >>> # Be sure of the structure of your system when define the parent attribute
        >>> planet = pr.Planet(parent = star, m = 0.1, a = 1, e = 0.2)
        >>> moon = pr.Planet(parent = planet, m = 0.01, a = 0.1, e = 0.5)
        >>>
        >>> # You would expect the next output tree based on the structure of the system
        >>> star.show_tree()
        <pryngles.body.Star object at 0x71343555c340>
        └── <pryngles.body.Planet object at 0x71343555c160>
            └── <pryngles.body.Planet object at 0x71343555c070>

        >>>
        >>> # It is the hierarchical orbital structure of the system
        >>> orbital_tree = pr.OrbitUtil.build_tree(star); orbital_tree
        [<pryngles.body.Star at 0x71343555c340>,
            [<pryngles.body.Planet at 0x71343555c160>,
             <pryngles.body.Planet at 0x71343555c070>]]

        .. image:: images/sun_earth_moon.png
            :align: center
            :scale: 30 %

        >>> # For a two-stars system, follow the convention to build the system
        >>>
        >>> # Build your first system
        >>> star = pr.Star()
        >>> planet = pr.Planet(parent = star, m = 0.1, a = 1, e = 0.2)
        >>> moon = pr.Planet(parent = planet, m = 0.01, a = 0.1, e = 0.5)
        >>>
        >>> # And next, the last system orbits the first star
        >>> star2 = pr.Star(parent = star, m = 1, a = 5, e = 0.3)
        >>> planet2 = pr.Planet(parent = star2, m = 0.1, e = 0.5)
        >>>
        >>> # You would expect the next output tree based on the structure of the system
        >>> star.show_tree()
        <pryngles.body.Star object at 0x71342e63e7d0>
        ├── <pryngles.body.Planet object at 0x7134367c1960>
        │   └── <pryngles.body.Planet object at 0x71342e63e770>
        └── <pryngles.body.Star object at 0x71342e63e7a0>
            └── <pryngles.body.Planet object at 0x71342e63e710>
        >>>
        >>> # You can trust that our method works!!
        >>> pr.OrbitUtil.build_tree(star)
        [[<pryngles.body.Star at 0x71342e63e7d0>,
          [<pryngles.body.Planet at 0x7134367c1960>,
          <pryngles.body.Planet at 0x71342e63e770>]],
        [<pryngles.body.Star at 0x71342e63e7a0>,
        <pryngles.body.Planet at 0x71342e63e710>]]

        .. image:: images/two_stars.png
            :align: center
            :scale: 30 %
        """
        orbital_tree = root

        for children in root.children:
            if children.kind == "Ring":
                continue

            children_tree = OrbitUtil.build_tree(children)

            orbital_tree = [orbital_tree] + [children_tree]

        return orbital_tree

    def build_system(orbital_tree, units=None):
        """
        Builds, recursively, a hierarchichal N-body system from clustering various pairs of :any:`body.Body` objects.

        Parameters
        ----------------
        orbital_tree : `list`
            Nested list with 2-body subsystems wich forms the system.
            Each subsystem is represented as a two (2) :data:`~ body.Body` objects list
            | You can pass it or generate it by the :any:`orbit.OrbitUtil.build_tree` method.

        units : `list`
            List of string containing the units convention used in calculations.
            The order **SHOULD** always be MKS: length, mass, time (in that order).
            The allowed units are defined in our :doc:`consts` and imported from ``rebound`` package.
            | **Default** =  `['au', 'msun', 'yr2pi']`

        Return
        ---------------
        :
            orbit : :data:`~ orbit.Orbit`
                :data:`~ orbit.Orbit` instance class containing the `rebound` simulation of the
                hierarchical N-body system.
            pelements : `dict`
                Dictionary of orbital elements from the primary component (root body).
                It will be useful in case you want to reproduce the dynamics of the root body

        Examples
        ---------------
        >>> # Let's see our sun-Earth-Moon like system
        >>> orbital_tree = [star, [planet, moon]]
        >>>
        >>> # Now you can assemble your system
        >>> orbit = pr.OrbitUtil.build_system(orbital_tree)
        >>> orbit.calculate_orbit()
        >>>
        >>> # You can access to the Rebound Simulation
        >>> orbit.sim.move_to_com()

        .. image:: images/sun_earth_moon_orbit.png
            :align: center

        >>> # Now see it for our two-stars system
        >>> orbital_tree = [[star, [planet, moon]], [star2, planet2]]
        >>> orbit = pr.OrbitUtil.build_system(orbital_tree)
        >>> orbit.calculate_orbit()
        >>> orbit.sim.move_to_com()

        .. image:: images/two_stars_orbit.png
            :align: center
        """

        p1, p2 = orbital_tree

        if type(p1) is list:
            m1, pelements = OrbitUtil.build_system(p1, units)
        else:
            m1 = p1.m
            pelements = p1.elements.copy()

        if type(p2) is list:
            m2, elements = OrbitUtil.build_system(p2, units)
        else:
            m2 = p2.m
            elements = p2.elements

        Orbit.ORBIT_SIMULATIONS = []
        orbit = Orbit(m1=m1, m2=m2, units=units, **elements)

        return orbit, pelements
