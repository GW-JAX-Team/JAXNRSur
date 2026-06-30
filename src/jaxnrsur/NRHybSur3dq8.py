from collections.abc import Sequence

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jaxnrsur import WaveformModel
from jaxnrsur.DataLoader import DataLoader, h5_mode_tuple, h5Group_to_dict, load_data
from jaxnrsur.EIMPredictor import EIMpredictor
from jaxnrsur.Harmonics import SpinWeightedSphericalHarmonics
from jaxnrsur.Spline import CubicSpline, CubicSplineFactorization
from jaxnrsur.typing import FloatLike

_DEFAULT_MODELIST: tuple[tuple[int, int], ...] = (
    (2, 2),
    (2, 1),
    (2, 0),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 2),
    (4, 3),
    (4, 4),
    (5, 5),
)


def _map_params(params: Float[Array, " n_dim"]) -> Float[Array, " n_dim"]:
    """Map physical [q, chi1z, chi2z] to surrogate training space [log(q), chiHat, chi_a].

    NRHybSur3dq8 GP node functions were trained on these remapped parameters.
    See gwsurrogate.new.nodeFunction.NRHybSur3dq8Fit for the reference implementation.
    """
    q, chi1z, chi2z = params[0], params[1], params[2]
    eta = q / (1.0 + q) ** 2
    chi_wtAvg = (q * chi1z + chi2z) / (1.0 + q)
    chiHat = (chi_wtAvg - 38.0 * eta / 113.0 * (chi1z + chi2z)) / (
        1.0 - 76.0 * eta / 113.0
    )
    chi_a = (chi1z - chi2z) / 2.0
    return jnp.array([jnp.log(q), chiHat, chi_a])


def get_T3_phase(
    q: FloatLike, t: Float[Array, " n"], t_ref: float = 1000.0
) -> Float[Array, " n"]:
    """
    Compute the T3 phase correction for the waveform model.

    Args:
        q (FloatLike): Mass ratio.
        t (Float[Array, " n"]): Time array.
        t_ref (float, optional): Reference time. Defaults to 1000.0.

    Returns:
        float: T3 phase correction value.
    """
    eta = q / (1 + q) ** 2
    theta_raw = (eta * (t_ref - t) / 5) ** (-1.0 / 8)
    theta_cal = (eta * (t_ref + 1000) / 5) ** (-1.0 / 8)
    return 2.0 / (eta * theta_raw**5) - 2.0 / (eta * theta_cal**5)


class NRHybSur3dq8DataLoader(DataLoader):
    sur_time: Float[Array, " n_sample"]
    modes: list[dict]

    def __init__(
        self,
        modelist: Sequence[tuple[int, int]] = _DEFAULT_MODELIST,
    ) -> None:
        """
        Initialize the NRHybSur3dq8DataLoader.

        Args:
            modelist (Sequence[tuple[int, int]]): Mode tuples to load.
        """
        data = load_data(
            "https://zenodo.org/records/3348115/files/NRHybSur3dq8.h5?download=1",
            "NRHybSur3dq8.h5",
        )
        self.sur_time = jnp.array(data["domain"])

        self.modes = []
        for i in range(len(modelist)):
            self.modes.append(self.read_single_mode(data, modelist[i]))

    def read_function(self, node_data: h5py.Group) -> dict:
        """
        Read a function group from the HDF5 node data and construct the EIM predictors.

        Args:
            node_data (h5py.Group): HDF5 group containing node function data.

        Returns:
            dict: Dictionary containing predictors, EIM basis, and metadata.

        Raises:
            KeyError: If required data is missing.
            TypeError: If required data has an unexpected HDF5 type.
        """
        n_nodes_data = node_data["n_nodes"]
        if not isinstance(n_nodes_data, h5py.Dataset):
            raise TypeError("n_nodes data must be an HDF5 dataset")

        n_nodes = int(n_nodes_data[()])
        result: dict[str, object] = {"n_nodes": n_nodes}
        predictors = []
        for count in range(n_nodes):
            fit_data = node_data[
                f"node_functions/ITEM_{count}/node_function/DICT_fit_data"
            ]
            if not isinstance(fit_data, h5py.Group):
                raise TypeError("GPR fit data must be an HDF5 group")

            predictors.append(EIMpredictor(h5Group_to_dict(fit_data)))

        name_data = node_data["name"]
        if not isinstance(name_data, h5py.Dataset):
            raise TypeError("Function name data must be an HDF5 dataset")
        name = name_data[()]
        if isinstance(name, bytes):
            result["name"] = name.decode("utf-8")
        elif isinstance(name, str):
            result["name"] = name
        else:
            raise TypeError("Function name must be text")

        eim_basis_data = node_data["ei_basis"]
        if not isinstance(eim_basis_data, h5py.Dataset):
            raise TypeError("EIM basis data must be an HDF5 dataset")

        result["predictors"] = predictors
        result["eim_basis"] = jnp.array(eim_basis_data[()])
        # Stack all EIMpredictor leaves so eqx.filter_vmap can batch the
        # loop over nodes into a single vectorised kernel call.
        result["stacked_predictor"] = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *predictors
        )
        return result

    def read_function_item(self, data: h5py.Group, item: str) -> dict:
        """Read a function group stored under ``item`` in an HDF5 group."""
        function_data = data[item]
        if not isinstance(function_data, h5py.Group):
            raise TypeError(f"{item} must be an HDF5 group")
        return self.read_function(function_data)

    @staticmethod
    def make_empty_function(name: str, length: int) -> dict:
        """
        Create an empty function dictionary for a mode component.

        Args:
            name (str): Name of the function ('re' or 'im').
            length (int): Length of the EIM basis.

        Returns:
            dict: Dictionary representing an empty function.
        """
        return {
            "n_nodes": 1,
            "predictors": [lambda x: 1],
            "eim_basis": jnp.zeros((1, length)),
            "name": name,
            "stacked_predictor": None,  # no real nodes — always evaluates to zero
        }

    def read_single_mode(self, file: h5py.File, mode: tuple[int, int]) -> dict:
        """
        Read a single mode's data from the HDF5 file.

        Args:
            file (h5py.File): HDF5 file object.
            mode (tuple[int, int]): Mode tuple (l, m).

        Returns:
            dict: Dictionary containing mode data.
        """
        result = {}
        data = file[f"sur_subs/{h5_mode_tuple[mode]}/func_subs"]
        if not isinstance(data, h5py.Group):
            raise TypeError("Mode data must be an HDF5 group")
        if mode == (2, 2):
            result["phase"] = self.read_function_item(data, "ITEM_0")
            result["amp"] = self.read_function_item(data, "ITEM_1")
        else:
            if mode[1] != 0:
                result["real"] = self.read_function_item(data, "ITEM_0")
                result["imag"] = self.read_function_item(data, "ITEM_1")
            else:
                local_function = self.read_function_item(data, "ITEM_0")
                if local_function["name"] == "re":
                    result["real"] = local_function
                    result["imag"] = self.make_empty_function(
                        "im", local_function["eim_basis"].shape[1]
                    )
                else:
                    result["imag"] = local_function
                    result["real"] = self.make_empty_function(
                        "re", local_function["eim_basis"].shape[1]
                    )
        result["mode"] = mode
        return result


class NRHybSur3dq8Model(WaveformModel[NRHybSur3dq8DataLoader]):
    data: NRHybSur3dq8DataLoader
    spline_factorization: CubicSplineFactorization
    mode_no22: list[dict]
    harmonics: list[SpinWeightedSphericalHarmonics]
    negative_harmonics: list[SpinWeightedSphericalHarmonics]
    mode_22_index: int
    m_mode: Int[Array, " n_modes-1"]
    negative_mode_prefactor: Int[Array, " n_modes-1"]

    def __init__(
        self,
        modelist: Sequence[tuple[int, int]] = _DEFAULT_MODELIST,
    ) -> None:
        """
        Initialize NRHybSur3dq8Model.

        The model is described in the paper:
        https://journals.aps.org/prd/abstract/10.1103/PhysRevD.99.064045

        Args:
            modelist (Sequence[tuple[int, int]]): Modes to be used.
        """
        self.data = NRHybSur3dq8DataLoader(modelist=modelist)
        self.spline_factorization = CubicSplineFactorization(self.data.sur_time)
        self.harmonics = []
        self.negative_harmonics = []
        negative_mode_prefactor = []
        for mode in modelist:
            if mode != (2, 2):
                self.harmonics.append(
                    SpinWeightedSphericalHarmonics(-2, mode[0], mode[1])
                )
                self.negative_harmonics.append(
                    SpinWeightedSphericalHarmonics(-2, mode[0], -mode[1])
                )
                # h_{l,-m} = (-1)^l * conj(h_{l,m}) for m > 0; no negative partner for m = 0
                if mode[1] > 0:
                    negative_mode_prefactor.append((-1) ** mode[0])
                else:
                    negative_mode_prefactor.append(0)

        self.mode_no22 = [
            self.data.modes[i] for i in range(len(self.data.modes)) if i != 0
        ]
        self.mode_22_index = int(
            jnp.where((jnp.array(modelist) == jnp.array([[2, 2]])).all(axis=1))[0][0]
        )
        self.m_mode = jnp.array(
            [modelist[i][1] for i in range(len(modelist)) if i != self.mode_22_index]
        )
        self.negative_mode_prefactor = jnp.array(negative_mode_prefactor)

    def __call__(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_dim"],
        theta: FloatLike = 0.0,
        phi: FloatLike = 0.0,
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """
        Compute the waveform for given time and source parameters.

        Args:
            time (Float[Array, " n_sample"]): Time grid.
            params (Float[Array, " n_dim"]): Source parameters.
            theta (float, optional): Polar angle. Defaults to 0.0.
            phi (float, optional): Azimuthal angle. Defaults to 0.0.

        Returns:
            tuple: Plus and cross polarizations of the waveform.
        """
        return self.get_waveform_geometric(time, params, theta, phi)

    @property
    def n_modes(self) -> int:
        """
        Get the number of modes in the model.

        Returns:
            int: Number of modes.
        """
        return len(self.data.modes)

    @staticmethod
    def get_eim(
        eim_dict: dict, params: Float[Array, " n_dim"]
    ) -> Float[Array, " n_sample"]:
        """
        Construct the EIM basis given the source parameters.

        Args:
            eim_dict (dict): EIM dictionary containing predictors and basis.
            params (Float[Array, " n_dim"]): Source parameters.

        Returns:
            Float[Array, " n_sample"]: EIM basis evaluated at parameters.
        """
        stacked = eim_dict["stacked_predictor"]
        if stacked is not None:
            node_vals = eqx.filter_vmap(lambda p: p(params))(stacked)
            # Each GPR returns shape (1,) for a single test point; squeeze to scalar per node.
            node_vals = node_vals[:, 0]
        else:
            node_vals = jnp.zeros(eim_dict["n_nodes"])
        return jnp.dot(eim_dict["eim_basis"].T, node_vals)

    @staticmethod
    def get_real_imag(
        mode: dict, params: Float[Array, " n_dim"]
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """
        Get the real and imaginary parts for a mode given parameters.

        Args:
            mode (dict): Mode dictionary containing 'real' and 'imag' EIM data.
            params (Float[Array, " n_dim"]): Source parameters.

        Returns:
            tuple: Real and imaginary parts as arrays.
        """
        params = params[None]
        real = NRHybSur3dq8Model.get_eim(mode["real"], params)
        imag = NRHybSur3dq8Model.get_eim(mode["imag"], params)
        return real, imag

    @staticmethod
    def get_multi_real_imag(
        modes: list[dict], params: Float[Array, " n_dim"]
    ) -> tuple[list[Float[Array, " n_sample"]], list[Float[Array, " n_sample"]]]:
        """
        Get real and imaginary parts for multiple modes.

        Args:
            modes (list[dict]): List of mode dictionaries.
            params (Float[Array, " n_dim"]): Source parameters.

        Returns:
            tuple: Lists of real and imaginary arrays for each mode.
        """
        return jax.tree_util.tree_map(
            lambda mode: NRHybSur3dq8Model.get_real_imag(mode, params),
            modes,
            is_leaf=lambda x: isinstance(x, dict),
        )

    def get_mode(
        self,
        real: Float[Array, " n_sample"],
        imag: Float[Array, " n_sample"],
        time: Float[Array, " n_time"],
    ) -> Float[Array, " n_sample"]:
        """
        Interpolate real and imaginary mode data to the given time grid.

        Args:
            real (Float[Array, " n_sample"]): Real part of mode.
            imag (Float[Array, " n_sample"]): Imaginary part of mode.
            time (Float[Array, " n_time"]): Time grid.

        Returns:
            Float[Array, " n_sample"]: Complex mode data at requested times.
        """
        return CubicSpline(self.data.sur_time, real, self.spline_factorization)(
            time
        ) + 1j * CubicSpline(self.data.sur_time, imag, self.spline_factorization)(time)

    def get_22_mode(
        self,
        time: Float[Array, " n_samples"],
        params: Float[Array, " n_dim"],
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """
        Compute the (2,2) mode and its phase for the waveform.

        Returns (h22, phase_interp) where phase_interp is needed by
        get_waveform_geometric to rotate coorbital modes to the inertial frame.
        """
        q = params[0]
        mapped = _map_params(params)[None]  # EIM was trained on [log(q), chiHat, chi_a]
        amp = self.get_eim(self.data.modes[self.mode_22_index]["amp"], mapped)
        phase = -self.get_eim(self.data.modes[self.mode_22_index]["phase"], mapped)
        phase = phase + get_T3_phase(q, self.data.sur_time)
        amp_interp = CubicSpline(self.data.sur_time, amp, self.spline_factorization)(
            time
        )
        phase_interp = CubicSpline(
            self.data.sur_time, phase, self.spline_factorization
        )(time)
        return amp_interp * jnp.exp(1j * phase_interp), phase_interp

    def get_waveform_geometric(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_dim"],
        theta: FloatLike = 0.0,
        phi: FloatLike = 0.0,
        omega_lower: FloatLike = 0.0,
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """
        Compute the geometric waveform (plus and cross polarizations) for given parameters.

        Current implementation separates the 22 mode from the rest of the modes,
        due to data structure and combination method. This means CubicSpline is called in a loop,
        which is not ideal (double the run time). The data structure could be merged for efficiency.

        Args:
            time (Float[Array, " n_sample"]): Time grid.
            params (Float[Array, " n_dim"]): Source parameters.
            theta (Float, optional): Polar angle. Defaults to 0.0.
            phi (Float, optional): Azimuthal angle. Defaults to 0.0.

        Returns:
            tuple: Plus and cross polarizations of the waveform.
        """
        mapped_params = _map_params(
            params
        )  # [log(q), chiHat, chi_a] for EIM evaluation
        coeff = jnp.stack(
            jnp.array(self.get_multi_real_imag(self.mode_no22, mapped_params))
        )
        modes = eqx.filter_vmap(self.get_mode, in_axes=(0, 0, None))(
            coeff[:, 0], coeff[:, 1], time
        )

        waveform = jnp.zeros_like(time, dtype=jnp.complex128)

        # get_22_mode returns (h22, phase_22).  phase_22 doubles as the orbital phase
        # (×2) used to rotate coorbital modes into the inertial frame:
        #   h_lm_inertial = h_coorb_lm * exp(-i·m·φ_orb)
        # JAXNRSur convention: h22 = A·exp(+i·phase_22), gwsurrogate: A·exp(-i·φ_22),
        # so phase_22 = -φ_22_raw and φ_orb = -phase_22/2, giving
        #   h_lm_inertial = h_coorb_lm * exp(+i·m·phase_22/2).
        # gwsurrogate aligns φ_22 at the LAST time point (φ_22 -= φ_22[refIdx]).
        # refIdx is found by _search_omega(omega22, 0) which returns the last
        # element because omega22[-1]=0 is appended and is the closest to 0.
        # To match, subtract phase_22[-1] from the continuous phase before use.
        h22, phase_22 = self.get_22_mode(time, params)
        phase_22_ref = phase_22[-1]
        phase_22 = phase_22 - phase_22_ref
        h22 = h22 * jnp.exp(-1j * phase_22_ref)
        waveform += h22 * SpinWeightedSphericalHarmonics(-2, 2, 2)(theta, -phi)
        waveform += jnp.conj(h22) * SpinWeightedSphericalHarmonics(-2, 2, -2)(
            theta, -phi
        )

        for i, harmonics in enumerate(self.harmonics):
            m = self.m_mode[i]  # JAX scalar; int() would fail under JIT
            # rotate coorbital → inertial frame before projection
            h_lm = modes[i] * jnp.exp(1j * m * phase_22 / 2.0)
            waveform += h_lm * harmonics(theta, -phi)
            waveform += (
                self.negative_mode_prefactor[i]
                * jnp.conj(h_lm)
                * self.negative_harmonics[i](theta, -phi)
            )

        # Find t_lower from the 22-mode continuous GW phase (d(phase)/dt / 2 = orb freq)
        # Re-compute phase on the full sur_time grid (needed for frequency mask).
        mapped_p = _map_params(params)[None]
        phase_sur = -self.get_eim(
            self.data.modes[self.mode_22_index]["phase"], mapped_p
        )
        phase_sur = phase_sur + get_T3_phase(params[0], self.data.sur_time)
        orb_freq_grid = jnp.asarray(jnp.gradient(phase_sur, self.data.sur_time)) / 2.0
        in_band = orb_freq_grid >= jnp.asarray(omega_lower)
        t_lower_m = self.data.sur_time[jnp.argmax(in_band)]
        t_start = jnp.where(omega_lower > 0, t_lower_m, self.data.sur_time[0])

        mask = (time >= t_start) * (time <= self.data.sur_time[-1])
        hp = jnp.where(mask, waveform.real, 0.0)
        hc = jnp.where(mask, -waveform.imag, 0.0)
        return hp, hc
