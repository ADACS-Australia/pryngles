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
import pytest
from pryngles import *

def test_fun():

    Verbose.VERBOSITY=VERB_ALL

    B=Body("Body",BODY_DEFAULTS,None,m=2,x=2,a=1,name_by_kind=True)

    print(B)
    print(B.m)

    B.update_body(name="B")
    print(B)

    C=Body("Body",BODY_DEFAULTS,B,name="C")
    print(C)
    print(B)

    #Tree structure
    C.show_tree()
    B.show_tree()

    #Test legacy
    B=Body("Body",BODY_DEFAULTS,None,name_by_kind=True,primary=C,orbit=dict(m=2,x=2,a=1))
    print(B.m)

    Verbose.VERBOSITY=VERB_NONE

def test_spangle():

    Verbose.VERBOSITY=VERB_ALL

    #Create body
    B=Body("Body",BODY_DEFAULTS,None,name='B',m=2,x=2)
    B.spangle_body()
    B.sg.plot3d()

    Verbose.VERBOSITY=VERB_NONE

def test_star():

    Verbose.VERBOSITY=VERB_ALL

    S=Star()
    print(S)

    #Check derived properties
    np.testing.assert_allclose(np.isclose([S.wrot],
                                [2*np.pi/BODY_DEFAULTS["prot"]],
                                rtol=1e-7))

    S.update_star(m=2,limb_coeffs=[1,1])
    print(S)

    #Check exception: parent could not be different from None or Body
    with pytest.raises(ValueError): Star(parent="Nada")

    S=Star(nspangles=270,i=45*Consts.deg)
    S.spangle_body()

    print_df(S.sg.data.tail())

    S.sg.set_observer()
    S.sg.set_luz()
    S.sg.plot3d()

    Verbose.VERBOSITY=VERB_NONE

def test_planet():

    Verbose.VERBOSITY=VERB_ALL

    S=Star()

    #Check exception: parent is mandatory for planets
    with pytest.raises(ValueError): Planet()

    P=Planet(parent=S)
    print(P.name)

    #Check derived properties
    np.testing.assert_allclose(np.isclose([P.wrot],
                                [2*np.pi/BODY_DEFAULTS["prot"]],
                                rtol=1e-7))

    #Check a non-existing property
    P.update_planet(vz=0.2)
    print(P)

    #Check exception: parent could not be different from None or Body
    with pytest.raises(AssertionError): Planet(parent="Nada")

    P.update_body(nspangles=250)
    P.spangle_body()
    print_df(P.sg.data.tail())

    P.sg.plot3d()

    Verbose.VERBOSITY=VERB_NONE

def test_ring():

    Verbose.VERBOSITY=VERB_ALL

    #Define first star and planet
    S=Star()
    P=Planet(parent=S)

    with pytest.raises(ValueError): Ring()
    R=Ring(parent=P)

    R.update_ring(fe=3)
    print(R)

    R.update_body(nspangles=250,i=60*Consts.deg,roll=0*Consts.deg)
    R.spangle_body()
    print_df(R.sg.data.tail())
    R.sg.plot3d()

    Verbose.VERBOSITY=VERB_NONE
