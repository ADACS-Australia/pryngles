import inspect
import dill

# Verbosity levels: see help(Verbose)
VERB_NONE = 0
VERB_SIMPLE = 1
VERB_SYSTEM = 2
VERB_VERIFY = 3
VERB_DEEP = 4
VERB_ALL = 100


class Verbose:
    """Verbose print in the package

    Attributes:
        VERBOSITY: int, default = 0:
            Level of verbosity.

            Verbosity levels:
                SIMPLE: Simple messages.
                SYSTEM: System operations.
                VERIFY: Message to verify operations
                DEEP: Deep debugging messages
                ALL: All debugging messages

    Methods:
        print(level,msg):
            Print a message if level<=VERBOSITY.

    Example:

        Verbose.print(1,"Hello world") #No print

        Verbose.print(0,"Hello world") #Print

        Verbose.VERBOSITY=1
        Verbose.print(1,"Hello world") #Print

        Verbose.VERBOSITY=2
        Verbose.print(1,"Hello world") #Print

        Verbose.VERBOSITY=2
        Verbose.print(4,"Hello world") #No print
    """

    VERBOSITY = VERB_ALL

    def print(level, *args):
        if level <= Verbose.VERBOSITY:
            print("  " * level + f"VERB{level}::{inspect.stack()[1][3]}::", *args)


# Alias
verbose = Verbose.print


class PrynglesCommon:
    """Base class of the package.

    All major classes are children of PrynglesCommon class.
    """

    def __init__(self):
        pass

    def save_to(self, filename):
        """Save object to a binary file

        Parameters:
            filename: string:
                Name of the file where the object will be stored.

        Notes:
            Based on https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e.
        """
        Verbose.print(VERB_SYSTEM, f"Saving object to {filename}")
        pikd = open(filename, "wb")
        dill.dump(self, pikd)
        pikd.close()

    def load_from(self, filename):
        """Read object from a binary file.

        Parameters:
            filename: string:
                Name of the file where the object is stored.
        """
        Verbose.print(VERB_SYSTEM, f"Loading object from {filename}")
        pikd = open(filename, "rb")
        data = dill.load(pikd)
        pikd.close()
        Verbose.print(VERB_VERIFY, "Transferring data to new object")
        self.__dict__ = data.__dict__
        return data

    def __str__(self):
        """Show content of an object

        This method determines the default behavior of the command:

            print(object)
        """
        # Remove private attributes
        return str({k: v for k, v in self.__dict__.items() if k[0] != "_"})