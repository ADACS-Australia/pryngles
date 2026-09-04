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
from collections import OrderedDict as odict
from collections.abc import Iterable
import inspect
import os
import gdown
import pandas as pd
from sys import maxsize as HASH_MAXSIZE

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Constants of module miscelaneous
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DATA_INDEX=dict(
    baseurl = 'https://docs.google.com/feeds/download/spreadsheets/Export?exportFormat=csv&key=',
    downurl = 'https://docs.google.com/uc?export=download&id=',
    filename = 'pryngles_data_index.csv',
    fileid = '17rgmzuENn_5jEO2rzDzngJLpZT1P9rPsQr3PNQGFkGs',
    col_filename = 'filename',
    col_fileid = 'fileid',
)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Class Misc
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Misc(object):
    """
    Miscelaneous routines.
    
    This is a set of util routines intended for a diversity of purposes.
    
    Routines included:
    
        get_data(file)
    """

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Data methods
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def get_data(path):
        """
        Get the full path of the `datafile` which is one of the datafiles provided with the package.
        
        Parameters:
            datafile: Name of the data file, string.
            
        Return:
            Full path to package datafile in the python environment.
            
        """
        return os.path.join(ROOTDIR,'data',path);

    def retrieve_data(datafile,path="/tmp/",quiet=False,overwrite=False):
        """Retrieve a data file from public Pryngles repo, https://bit.ly/pryngles-data.

        Parameters:
        
            datafile: string.
                Name of the datafile to retrieve.

            path: string, default = '/tmp/'
                Path where the data files be retrieved.

            quiet: bool, default = False
                Is the downloading process quiet or not. This is a `gdown` option.

        Return:
            List of datafiles retrieve it.        
        """
        # Get the list of files
        filename = path+'/'+DATA_INDEX['filename']
        if not os.path.isfile(filename) or overwrite:
            url = DATA_INDEX['baseurl']+DATA_INDEX['fileid']
            gdown.download(url,filename,quiet=quiet)
        else:
            if not quiet:
                print(f"Index file {filename} already retrieved. For overwrite use overwrite = True.")
            
        files=pd.read_csv(filename,index_col=DATA_INDEX['col_filename'])
        if not quiet:
            print(f"There are {len(files)} files in data repository.")

        if isinstance(datafile,str):
            datafile=[datafile]

        if len(datafile) == 0:
            return list(files.index)
            
        dfiles=[]
        for dfile in datafile:
            # Look for file in the data index
            if dfile in files.index:
                fileid = files.loc[dfile,DATA_INDEX['col_fileid']]
                # Download
                url = DATA_INDEX['downurl']+fileid
                filename = path+'/'+dfile
                if not os.path.isfile(filename) or overwrite:
                    gdown.download(url,filename,quiet=quiet)
                else:
                    if not quiet:
                        print(f"File {filename} already retrieved. For overwrite use overwrite = True.")
                dfiles+=[filename]
            else:
                raise ValueError(f"Datafile {dfile} not available in repository. List of available files:\n{list(files.index)}")

        if not quiet:
            print(f"Files downloaded: {dfiles}")

        return dfiles

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Input/output methos
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def print_df(df):
        """Print DataFrame.
        
        Parameters:
            df: Pandas DataFrame:
                DataFrame to print.
        """
        display(HTML(df.to_html()))
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Array methods
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def flatten(collection):
        """Flatten a list of objects

        Examples:
            list(Misc.flatten(["cosa"]))
            list(Misc.flatten([["cosa"]]))
            list(Misc.flatten([["cosa","perro"]]))
            list(Misc.flatten([[1,"perro"],object,float]))
        """
        for i in collection:
            if isinstance(i, Iterable) and not isinstance(i, (str, bytes)):
                for subc in Misc.flatten(i):
                    yield subc
            else:
                yield i
                
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Programming methods
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def get_methods(my_class):
        """Get a list of the methods for class my_class
        """
        return sorted([member[0] for member in inspect.getmembers(my_class) if '__' not in member[0]])
    
    def calc_hash(obj):
        if type(obj) is dict:
            hash_obj=frozenset(obj.items())
        else:
            hash_obj=obj
        hash_val=str(hash(hash_obj)%((HASH_MAXSIZE+1)*2))
        return hash_val

