import numpy as np
import pandas as pd


class SystemChecks:
    """Shared regression tests for any 'system' fixture."""

    # Numerical values are compared with a tight relative tolerance.
    TOL = dict(rtol=1e-12, atol=1e-12)

    # Check the attributes that are declared in the System docstring
    def test_system_metadata(self, system, num_regression):
        data = {
            "ul" : system.ul,
            "um" : system.um,
            "ut" : system.ut,
            "G" : system.G,
            "nbodies" : system.nbodies,
            "nparticles" : system.nparticles,
            "n_obs_x" : system.n_obs[0],
            "n_obs_y" : system.n_obs[1],
            "n_obs_z" : system.n_obs[2],
        }
        num_regression.check(data, default_tolerance=self.TOL, basename="system_metadata_numeric")

    def test_spangler_data(self, system, dataframe_regression, file_regression):
        # Note: spangler data is a dataframe of mixed types.

        # Special handling of 'beta_loc', which is a numpy array of floats,
        # but gets converted to object dtype in the dataframe, for some reason...
        # Convert column to float.
        system.sg.data['beta_loc'] = system.sg.data['beta_loc'].astype(float)

        # Expand 3-vector columns into separate numeric columns
        cols = system.sg.data.select_dtypes(include=['object'],exclude=['string']).drop(columns=['scatterer']).columns
        suffixes = ['_x', '_y', '_z']
        new_cols = {}
        for col in cols:
            stacked = np.stack(system.sg.data[col].to_numpy())  # shape (n_rows, 3)
            for i, suf in enumerate(suffixes):
                new_cols[f"{col}{suf}"] = stacked[:, i]

        # Recombine the expanded columns with the rest of the DataFrame
        expanded = pd.DataFrame(new_cols, index=system.sg.data.index)
        system.sg.data = pd.concat([system.sg.data.drop(columns=cols), expanded], axis=1)

        # Numerical columns
        dataframe_regression.check(system.sg.data.select_dtypes(include=['number']), default_tolerance=self.TOL, basename="spangler_data_numeric_df")

        # String columns with rows containing non-empty strings (ignoring the 'name' column)
        df = system.sg.data.select_dtypes(include='string')
        other_cols = [c for c in df.columns if c != 'name']
        mask = (df[other_cols] != '').any(axis=1)
        file_regression.check(df[mask].to_csv(), extension=".csv", basename="spangler_data_string_df")

        # TODO: 'scatterer' column will need special handling.

    # Check the attributes that are declared in the Spangler docstring
    def test_spangler_metadata(self, system, data_regression, num_regression):
        # Plain types
        data = {
            "nspangles": system.sg.nspangles,
            "name": system.sg.name,
            "shape": system.sg.shape,
        }
        data_regression.check(data, basename="spangler_metadata")

        # Numeric types (3-vectors)
        data = {
            "n_obs": system.sg.n_obs,
            "n_luz": system.sg.n_luz,
        }
        num_regression.check(data, default_tolerance=self.TOL, basename="spangler_metadata_numeric")

        # Note: M_equ2ecl is a dict of ndarrays (matrices).
        # Could use ndarrays_regression, but that saves files as binary npz; not so good for git.
        # Prefere human-readable csv, so flatten each matrix and use num_regression.
        num_regression.check({key: val.flatten() for key, val in system.sg.M_equ2ecl.items()}, default_tolerance=self.TOL, basename="spangler_M_equ2ecl")

        # TODO: qhulls will need special handling; it's a dict of different types, including a qhull object.

    def test_lightcurve_arrays(self, system, num_regression):
        data = {
            # 1D arrays
            "times": system.lightcurve["times"],
            "total_flux": system.lightcurve["total_flux"],
            "bandwidth": system.lightcurve["bandwidth"],

            # Unpack dict of 3-vectors
            "observer_n_obs": system.lightcurve['observer']['n_obs'],
            "observer_direction": system.lightcurve['observer']['direction'],
        }
        num_regression.check(data, default_tolerance=self.TOL, basename="lightcurve_arrays")

    def test_lightcurve_metadata(self, system, data_regression):
        data = {
            "effects": system.lightcurve["effects"],
            "bodies": system.lightcurve["bodies"],
        }
        data_regression.check(data, basename="lightcurve_metadata")

    def test_lightcurve_effects(self, system, dataframe_regression, num_regression):
        for effect in system.lightcurve['effects']:
            self.check_multi_index_df(system.lightcurve[effect], dataframe_regression, basename=f"lightcurve_{effect}_df")
            if effect == 'polarization':
                self.check_multi_index_df(system.lightcurve['polarization'], dataframe_regression, basename="lightcurve_polarization_df")
        if 'signal' in system.lightcurve:
            num_regression.check(system.lightcurve['signal'], default_tolerance=self.TOL, basename="lightcurve_signal")

    # MultiIndex need to be flattened, otherwise they're read back in as object/str
    def check_multi_index_df(self, multi_index_df, dataframe_regression, basename):
        df = multi_index_df.set_axis(['_'.join(c) for c in multi_index_df.columns], axis=1)
        dataframe_regression.check(df, default_tolerance=self.TOL, basename=basename)
