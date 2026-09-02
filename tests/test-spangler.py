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
	def test_const(self):
	
	    Verbose.VERBOSITY=VERB_ALL
	
	    for key in SPANGLER_KEY_ORDERING:
	        if key not in SPANGLER_COLUMNS:
	            raise AssertionError(f"Column '{key}' in SPANGLER_KEY_ORDERING not in SPANGLER_COLUMNS")
	
	    for key in SPANGLER_COLUMNS:
	        if key not in SPANGLER_KEY_ORDERING:
	            raise AssertionError(f"Column '{key}' in SPANGLER_COLUMNS not in SPANGLER_KEY_ORDERING")
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_init(self):
	
	    Verbose.VERBOSITY=VERB_ALL
	
	    print("Basic definition:")
	    sg=Spangler(nspangles=1,center_equ=[0,0,0],n_equ=[1,0,0])
	    Misc.print_df(sg.data.head(1))
	    sg.get_mem_usage(True)
	
	    print("\nCenter equ:")
	    sg=Spangler(nspangles=3,center_equ=[0,0,1],n_equ=[0,1,0])
	    Misc.print_df(sg.data.head(1))
	
	    print("\nCenter ecl:")
	    sg=Spangler(nspangles=3,center_ecl=[0,0,1],n_equ=[0,0,1])
	    Misc.print_df(sg.data.head(1))
	
	    print("\nRotation:")
	    sg=Spangler(nspangles=3,w=30*Consts.deg,q0=40*Consts.deg,n_equ=[0,1,1])
	    sg.set_positions(t=1)
	    Misc.print_df(sg.data.head(1))
	
	    print("\nJoin:")
	    sg1=Spangler(name="Body 1",nspangles=3,w=40*Consts.deg,n_equ=[1,1,0])
	    sg2=Spangler(name="Body 2",nspangles=3,w=30*Consts.deg,n_equ=[1,0,1])
	    sg=Spangler(spanglers=[sg1,sg2])
	    sg.set_positions(t=1)
	    Misc.print_df(sg.data)
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_reset(self):
	
	    Verbose.VERBOSITY=VERB_SIMPLE
	
	    sg=Spangler(nspangles=100)
	    sg.reset_state()
	    print_df(sg.data[["unset"]+list(SPANGLER_VISIBILITY_STATES)+list(SPANGLER_SOURCE_STATES)].head())
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_scale(self):
	
	    Verbose.VERBOSITY=VERB_SIMPLE
	
	    sg=Spangler(center_ecl=[1,1,1],center_equ=[1,1,1])
	    print_df(sg.data)
	
	    sg.set_scale(5)
	    print_df(sg.data)
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_pop(self):
	    Verbose.VERBOSITY=VERB_ALL
	
	    #No preset
	    sg=Spangler(nspangles=850,n_equ=[1,0,0])
	    sg.populate_spangler(shape="ring",
	                         spangle_type=SPANGLE_GASEOUS,
	                         scale=2,seed=1,ri=0.2)
	    sg.sample.plot()
	    sg.sample.ax.set_title(f"N={sg.nspangles}")
	    sg.sample.fig.tight_layout()
	    print_df(sg.data.head(3))
	
	    #Using preset
	    sg=Spangler(nspangles=850)
	    sg.populate_spangler(shape="ring",
	                         preset=True,
	                         spangle_type=SPANGLE_SOLID_ROCK,ri=0.2)
	    sg.sample.plot()
	    sg.sample.ax.set_title(f"N={sg.nspangles}")
	    sg.sample.fig.tight_layout()
	    print_df(sg.data.head(3))
	
	    #Sphere
	    sg=Spangler(nspangles=100)
	    sg.populate_spangler(shape="sphere",scale=3,seed=1,preset=True)
	    sg.sample.plot(spangled=dict(color='r',alpha=0.1))
	    sg.sample.ax.set_title(f"N={sg.nspangles}")
	    sg.sample.fig.tight_layout()
	
	    print_df(sg.data.head(3))
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_plot3d(self):
	    global sg
	    Verbose.VERBOSITY=VERB_SIMPLE
	
	    #Sphere
	    sg=Spangler(nspangles=100)
	    sg.populate_spangler(shape="sphere",spangle_type=SPANGLE_SOLID_ICE,preset=0,scale=3)
	    sg.reset_state()
	
	    sg.data.unset=False
	    cond=sg.data.z_ecl>0
	    sg.data.loc[cond,"illuminated"]=True
	    cond=sg.data.x_ecl>0
	    sg.data.loc[cond,"visible"]=True
	    cond=sg.data.y_ecl>0
	    sg.data.loc[cond,"shadow"]=True
	    cond=sg.data.f_equ>45*Consts.deg
	    sg.data.loc[cond,"transmit"]=True
	
	    sg.plot3d(statemark=0.5,coords="ecl")
	
	    #No preset
	    sg=Spangler(nspangles=850,n_equ=[1,1,1])
	    sg.populate_spangler(shape="ring",preset=True,
	                         spangle_type=SPANGLE_GRANULAR,
	                         scale=2,ri=0.2)
	
	    sg.data.unset=False
	    sg.data.illuminated=True
	    sg.data.illuminated=True
	    cond=sg.data.x_ecl>0
	    sg.data.loc[cond,"visible"]=True
	    sg.data.loc[cond,"transmit"]=True
	    sg.plot3d(statemark=0.1)
	
	    #No preset
	    sg=Spangler(nspangles=50,n_equ=[1,1,1])
	    sg.populate_spangler(shape="ring",preset=True,
	                         spangle_type=SPANGLE_GRANULAR,
	                         scale=2,ri=0.2)
	    sg.plot3d(coords="ecl",show_directions=True)
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_setint(self):
	    global sg
	
	    Verbose.VERBOSITY=VERB_SIMPLE
	
	    #No preset
	    sg=Spangler(nspangles=50,name="Ring")
	    sg.populate_spangler(shape="ring",seed=1,
	                         spangle_type=SPANGLE_GRANULAR,
	                         scale=2,ri=0.2)
	    sg.data.illuminated=True
	    sg.data.visible=True
	
	    cond,n_int,d_int=sg.set_intersect(nvec=[1,0,1],center=[0,0,-1],
	                                      name="Ring")
	    sg._calc_qhulls()
	    #sg._plot_qhulls() #Deprecated
	
	    #Plot 3d
	    sg.plot3d(coords="int")
	    plane=sg.qhulls["Ring"][0]["plane"]
	    plane.plot_plane(ax=sg.ax3d,color='c',alpha=0.5)
	
	    #Hulls
	    print(sg.qhulls)
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_setobsluz(self):
	    global sg
	
	    Verbose.VERBOSITY=VERB_SIMPLE
	
	    #Normal
	    nspangles=10
	    sg=Spangler(nspangles=nspangles,n_equ=[1,0,1],name="Planet")
	    sg.populate_spangler(shape="sphere",preset=0,
	                         spangle_type=SPANGLE_SOLID_ROCK,
	                         scale=2)
	
	    print_df(sg.data.loc[~sg.data.hidden,SPANGLER_DEBUG_FIELDS])
	
	    sg.set_observer(nvec=[0,0,+1],center=None)
	    sg.set_luz(nvec=[+1,0,0],center=None)
	
	    sg.plot3d(coords="obs",statemark=1)
	
	    #Semitransparent
	    nspangles=50
	    sg=Spangler(nspangles=nspangles,n_equ=[1,0,1],name="Planet")
	    sg.populate_spangler(shape="sphere",preset=0,
	                         spangle_type=SPANGLE_GASEOUS,
	                         scale=2)
	    sg.set_observer(nvec=[0,0,+1],center=None)
	    sg.set_luz(nvec=[+1,0,0],center=None)
	    sg.plot3d(statemark=1)
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_simplevis(self):
	    global sg
	
	    Verbose.VERBOSITY=VERB_SIMPLE
	
	    plt.close("all")
	    #Ring with semitransparent spangle: all illuminated, all visible, no transmission
	    sg=Spangler(nspangles=100,n_equ=[1,1,1])
	    sg.populate_spangler(shape="ring",ri=0.3,spangle_type=SPANGLE_GRANULAR,preset=True,scale=3)
	    sg.set_observer([0,0,1])
	    sg.set_luz([1,1,-1])
	    sg.plot3d()
	
	    #Ring with semitransparent spangle: all illuminated, all visible, no transmission
	    sg=Spangler(nspangles=100,n_equ=[1,1,1])
	    sg.populate_spangler(shape="ring",ri=0.3,spangle_type=SPANGLE_GRANULAR,preset=True,scale=3)
	    sg.set_observer([0,0,1])
	    sg.set_luz([-1,-1,-1])
	    sg.plot3d()
	
	    #Sphere with solid spangle: only illuminated 
	    sg=Spangler(nspangles=100,n_equ=[1,1,1])
	    sg.populate_spangler(shape="sphere",spangle_type=SPANGLE_SOLID_ROCK,preset=True,scale=3)
	    sg.set_observer([1,0,1])
	    sg.set_luz([0,0,1])
	    sg.plot3d()
	
	    #Sphere with stellar spangle: all illuminated, not all visible
	    sg=Spangler(nspangles=100,n_equ=[1,1,1])
	    sg.populate_spangler(shape="sphere",spangle_type=SPANGLE_STELLAR,preset=True,scale=3)
	    sg.set_observer([1,0,1])
	    sg.set_luz([0,0,1])
	    sg.plot3d()
	
	    #Sphere with semitransparent spangle: all illuminated, all visible
	    sg=Spangler(nspangles=100,n_equ=[1,1,1])
	    sg.populate_spangler(shape="sphere",spangle_type=SPANGLE_GASEOUS,preset=True,scale=3)
	    sg.set_observer([0,0,1])
	    sg.set_luz([1,0,0])
	    sg.plot3d()
	
	    #Two spheres
	    sg1=Spangler(name="Planet 1",nspangles=100,center_equ=[-5,0,0])
	    sg1.populate_spangler(shape="sphere",spangle_type=SPANGLE_SOLID_ICE,preset=True,scale=3)
	
	    sg2=Spangler(name="Planet 2",nspangles=100,center_equ=[+5,0,0])
	    sg2.populate_spangler(shape="sphere",spangle_type=SPANGLE_SOLID_ROCK,preset=True,scale=3)
	
	    sg=Spangler(spanglers=[sg1,sg2])
	
	    sg.set_observer([0,1,0])
	    sg.set_luz(nvec=[1,0,0],center=[0,0,0],name="Planet 1")
	    sg.set_luz(nvec=[-1,0,0],name="Planet 2")
	
	    sg.plot3d()
	    return
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_plot2d(self):
	
	    global sg
	
	    Verbose.VERBOSITY=VERB_SIMPLE
	
	    plt.close("all")
	    sg=Spangler(nspangles=2500,name="123",n_equ=[1,1,1],center_ecl=[0,0,2])
	    sg.populate_spangler(shape="sphere",spangle_type=SPANGLE_SOLID_ROCK,scale=2,seed=1,preset=True)
	    #sg.populate_spangler(shape="sphere",spangle_type=SPANGLE_GASEOUS,scale=2,seed=1,preset=True)
	
	    sg.set_observer(nvec=[1,0,0])
	    sg.set_luz(nvec=[1,1,1])
	    fs=3
	    sg.plot3d(coords="ecl")
	    sg.plot2d(coords="ecl",fsize=fs)
	    sg.plot2d(coords="luz",fsize=fs)
	    sg.plot2d(coords="obs",fsize=fs)
	
	    sg=Spangler(nspangles=50,name="123",n_equ=[1,1,1],center_ecl=[0,1,0])
	    sg.populate_spangler(shape="sphere",spangle_type=SPANGLE_SOLID_ROCK,scale=2,seed=1,preset=True)
	    sg.set_observer(nvec=[1,0,0])
	    sg.set_luz(nvec=[0,1,0],center=[0,6,0])
	    sg.plot2d()
	    sg.plot2d(show_azim=True,fsize=5)
	    sg.plot3d(coords="luz",show_directions=True)
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_join(self):
	
	    Verbose.VERBOSITY=VERB_SIMPLE
	
	    sg1=Spangler(nspangles=1000,name="Ring",n_equ=[1,0,5])
	    sg1.populate_spangler(shape="ring",spangle_type=SPANGLE_GRANULAR,scale=2.5,seed=1,ri=1.5/2.5,boundary=0)
	
	    sg2=Spangler(nspangles=1000,name="Planet",n_equ=[0,0,1])
	    sg2.populate_spangler(shape="sphere",spangle_type=SPANGLE_ATMOSPHERIC,scale=1,seed=1,preset=True)
	
	    sgj=Spangler(spanglers=[sg1,sg2])
	
	    sgj.set_observer([1,0,0.1])
	    sgj.set_luz([0,0,1])
	
	    sgj.plot3d()
	    sgj.plot2d()
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_upint(self):
	    plt.close("all")
	    global sg
	
	    Verbose.VERBOSITY=VERB_NONE
	
	    """Shadow-test
	    """
	    nspangles=500
	    sps=[]
	    sg=Spangler(nspangles=nspangles,name="Star",n_equ=[0,0,1],center_equ=[-7,0,0])
	    sg.populate_spangler(shape="sphere",spangle_type=SPANGLE_STELLAR,scale=3,seed=1,preset=1)
	    sps+=[sg]
	    sg=Spangler(nspangles=nspangles,name="Planet",n_equ=[0,0,1])
	    sg.populate_spangler(shape="sphere",spangle_type=SPANGLE_SOLID_ROCK,scale=1,seed=1,preset=True)
	    sps+=[sg]
	    sg=Spangler(nspangles=nspangles,name="Ring",n_equ=[1,0,-2])
	    sg.populate_spangler(shape="ring",spangle_type=SPANGLE_GRANULAR,scale=2.5,seed=1,ri=1.5/2.5,boundary=0)
	    sps+=[sg]
	    sg=Spangler(nspangles=nspangles,name="Moon",n_equ=[0,0,1],center_equ=[+4.0,0.0,0.0])
	    sg.populate_spangler(shape="sphere",spangle_type=SPANGLE_ATMOSPHERIC,scale=0.3,seed=1,preset=True)
	    sps+=[sg]
	
	    sg=Spangler(spanglers=sps)
	
	    #"""
	    sg.set_observer(nvec=sci.direction(40,0))
	    sg.update_visibility_state()
	    #""";
	
	    #"""
	    sg.set_luz(nvec=sci.direction(0,0))
	    #sg.update_illumination_state(excluded=[])
	    sg.update_illumination_state(included=["Moon","Planet"])
	    #sg.update_illumination_state(included=["Ring","Star"])
	    #sg.update_illumination_state(excluded=["Star"])
	    #""";
	
	    SHADOW_COLOR_LUZ=[90,0.2,1.0]
	    sg.plot3d(center_at="Ring")
	    sg.plot2d(center_at="Ring",maxval=5)
	
	    Verbose.VERBOSITY=VERB_NONE
	
	def test_muluz(self):
	
	    Verbose.VERBOSITY=VERB_NONE
	
	    nspangles=100
	    sps=[]
	
	    sg=Spangler(nspangles=nspangles,name="Planet1",n_equ=[0,0,1],center_ecl=[0,0,0])
	    sg.populate_spangler(shape="sphere",spangle_type=SPANGLE_SOLID_ROCK,scale=1,seed=1,preset=True)
	    sps+=[sg]
	
	    sg=Spangler(nspangles=nspangles,name="Moon1",n_equ=[0,0,1],center_ecl=[2,0,0])
	    sg.populate_spangler(shape="sphere",spangle_type=SPANGLE_SOLID_ROCK,scale=0.5,seed=1,preset=True)
	    sps+=[sg]
	
	    sg=Spangler(spanglers=sps)
	
	    sg.set_observer([1,1,1])
	    sg.update_visibility_state()
	
	    name="Planet1"
	    sg.set_luz(nvec=[1,0,0],name=name)
	    sg.update_illumination_state()
	
	    #"""
	    name="Moon1"
	    sg.set_luz(nvec=[-2,1,0],name=name)
	    sg.update_illumination_state()
	    #"""
	
	    sg.plot3d()
	
	    Verbose.VERBOSITY=VERB_NONE
	
if __name__=="__main__":
   unittest.main(argv=['first-arg-is-ignored'],exit=False)
