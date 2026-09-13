import numpy as np
import pandas as pd


class SystemChecks:
    """Shared regression tests for any 'system' fixture."""

    # Numerical values are compared with a tight relative tolerance.
    TOL = dict(rtol=1e-12, atol=1e-12)

    # Check the attributes that are declared in the System docstring
    def test_system_metadata(self, system, num_regression):
        data = {k: getattr(system, k) for k in ["ul", "um", "ut", "G", "nbodies", "nparticles"]}

        # Unpack the n_obs vector into separate x, y, z components
        n_obs = getattr(system, "n_obs")
        data["n_obs_x"] = n_obs[0]
        data["n_obs_y"] = n_obs[1]
        data["n_obs_z"] = n_obs[2]

        num_regression.check(data, default_tolerance=self.TOL, basename="system_metadata_numeric")

    def test_spangler_data(self, system, data_regression, dataframe_regression):
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

        # String columns
        data_regression.check(system.sg.data.select_dtypes(include=['string']).to_dict(orient='records'), basename="spangler_data_string_df")

        # TODO: 'scatterer' column will need special handling.

    # Check the attributes that are declared in the Spangler docstring
    def test_spangler_metadata(self, system, data_regression, num_regression, ndarrays_regression):
        # Plain types
        data_regression.check({k: getattr(system.sg, k) for k in ["nspangles", "name", "shape"]}, basename="spangler_metadata")

        # Numeric types (3-vectors)
        num_regression.check({k: getattr(system.sg, k) for k in ["n_obs", "n_luz"]}, default_tolerance=self.TOL, basename="spangler_metadata_numeric")

        # Note: M_equ2ecl is a dict of ndarrays (matrices)
        ndarrays_regression.check(system.sg.M_equ2ecl, default_tolerance=self.TOL, basename="spangler_M_equ2ecl")

        # TODO: qhulls will need special handling; it's a dict of different types, including a qhull object.

    def test_lightcurve_arrays(self, system, num_regression):

        # 1d arrays
        numeric = {k: system.lightcurve[k] for k in ['times', 'total_flux', 'bandwidth']}

        # Unpack dict of 3-vectors
        numeric.update(
            observer_n_obs=system.lightcurve['observer']['n_obs'],
            observer_direction=system.lightcurve['observer']['direction'],
        )

        num_regression.check(numeric, default_tolerance=self.TOL, basename="lightcurve_arrays")

    def test_lightcurve_metadata(self, system, data_regression):
        dicts = {k: system.lightcurve[k] for k in ['effects', 'bodies']}
        data_regression.check(dicts, basename="lightcurve_metadata")

    def test_lightcurve_effects(self, system, dataframe_regression):
        for effect in system.lightcurve['effects']:
            dataframe_regression.check(system.lightcurve[effect], default_tolerance=self.TOL, basename=f"lightcurve_{effect}_df")
            if effect == 'polarization':
                dataframe_regression.check(system.lightcurve['polarization'], default_tolerance=self.TOL, basename="lightcurve_polarization_df")

    def test_lightcurve_signal(self,system, ndarrays_regression):
        # Note: system.lightcurve['signal'] is a dict of ndarrays
        if 'signal' in system.lightcurve:
            ndarrays_regression.check(system.lightcurve['signal'], default_tolerance=self.TOL, basename="lightcurve_signal")
