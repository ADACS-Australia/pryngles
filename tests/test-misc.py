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

from pryngles import *

def test_misc(self):

    #Get path
    filepath=Misc.get_data("diffuse_reflection_function.data")
    print(filepath)

    #print_df dataframe
    import pandas as pd
    import numpy as np
    df=pd.DataFrame(np.zeros((5,3)),columns=["a","b","c"])
    Misc.print_df(df)

    #Flatten
    print(list(Misc.flatten(["hola"])))
    print(list(Misc.flatten(["hola",["perro","gato"]])))

    #Get methods
    print(Misc.get_methods(Misc))

    #Hash
    d=dict(a=1,b=3,c=np)
    print(Misc.calc_hash(d))
    P=PrynglesCommon()
    print(Misc.calc_hash(P))
    print(Misc.calc_hash(PrynglesCommon))
