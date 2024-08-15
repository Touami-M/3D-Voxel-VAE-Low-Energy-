from dataclasses import dataclass
from enum import Enum

import numpy as np

from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit



# Number of calorimeter layers (z-axis segmentation).
N_CELLS_Z = 35
# Segmentation in the r,phi direction.
N_CELLS_R = 22
N_CELLS_PHI = 16
# Cell size in the r and z directions
SIZE_R = 4.296
SIZE_Z = 4.847

VALID_DIR = "./validation"

FULL_SIM_HISTOGRAM_COLOR = "blue"
ML_SIM_HISTOGRAM_COLOR = "red"
FULL_SIM_GAUSSIAN_COLOR = "green"
ML_SIM_GAUSSIAN_COLOR = "orange"
HISTOGRAM_TYPE = "step"


@dataclass
class Observable:
    """ An abstract class defining interface of all observables.

    Do not use this class directly.

    Attributes:
          _input: A numpy array with shape = (NE, R, PHI, Z), where NE stays for number of events.
    """
    _input: np.ndarray


class ProfileType(Enum):
    """ Enum class of various profile types.

    """
    LONGITUDINAL = 0
    LATERAL = 1


@dataclass
class Profile(Observable):
    """ An abstract class describing behaviour of LongitudinalProfile and LateralProfile.

    Do not use this class directly. Use LongitudinalProfile or LateralProfile instead.

    """

    def calc_profile(self) -> np.ndarray:
        pass

    def calc_first_moment(self) -> np.ndarray:
        pass

    def calc_second_moment(self) -> np.ndarray:
        pass


@dataclass
class LongitudinalProfile(Profile):
    """ A class defining observables related to LongitudinalProfile.

    Attributes:
        _energies_per_event: A numpy array with shape = (NE, Z) where NE stays for a number of events. An
            element [i, j] is a sum of energies detected in all cells located in a jth layer for an ith event.
        _total_energy_per_event: A numpy array with shape = (NE, ). An element [i] is a sum of energies detected in all
            cells for an ith event.
        _w: A numpy array = [0, 1, ..., Z - 1] which represents weights used in computation of first and second moment.

    """

    def __post_init__(self):
        self._energies_per_event = np.sum(self._input, axis=(1, 2))
        self._total_energy_per_event = np.sum(self._energies_per_event, axis=1)
        self._w = np.arange(N_CELLS_Z)

    def calc_profile(self) -> np.ndarray:
        """ Calculates a longitudinal profile.

        A longitudinal profile for a given layer l (l = 0, ..., Z - 1) is defined as:
        sum_{i = 0}^{NE - 1} energy_per_event[i, l].

        Returns:
            A numpy array of longitudinal profiles for each layer with a shape = (Z, ).

        """
        return np.sum(self._energies_per_event, axis=0)

    def calc_first_moment(self) -> np.ndarray:
        """ Calculates a first moment of profile.

        A first moment of a longitudinal profile for a given event e (e = 0, ..., NE - 1) is defined as:
        FM[e] = alpha * (sum_{i = 0}^{Z - 1} energies_per_event[e, i] * w[i]) / total_energy_per_event[e], where
        w = [0, 1, 2, ..., Z - 1],
        alpha = SIZE_Z defined in core/constants.py.

        Returns:
            A numpy array of first moments of longitudinal profiles for each event with a shape = (NE, ).

        """
        return SIZE_Z * np.dot(self._energies_per_event, self._w) / self._total_energy_per_event

    def calc_second_moment(self) -> np.ndarray:
        """ Calculates a second moment of a longitudinal profile.

        A second moment of a longitudinal profile for a given event e (e = 0, ..., NE - 1) is defined as:
        SM[e] = (sum_{i = 0}^{Z - 1} (w[i] - alpha - FM[e])^2 * energies_per_event[e, i]) total_energy_per_event[e],
        where
        w = [0, 1, 2, ..., Z - 1],
        alpha = SIZE_Z defined in ochre/constants.py

        Returns:
            A numpy array of second moments of longitudinal profiles for each event with a shape = (NE, ).
        """
        first_moment = self.calc_first_moment()
        first_moment = np.expand_dims(first_moment, axis=1)
        w = np.expand_dims(self._w, axis=0)
        # w has now a shape = [1, Z] and first moment has a shape = [NE, 1]. There is a broadcasting in the line
        # below how that one create an array with a shape = [NE, Z]
        return np.sum(np.multiply(np.power(w * SIZE_Z - first_moment, 2), self._energies_per_event),
                      axis=1) / self._total_energy_per_event


@dataclass
class LateralProfile(Profile):
    """ A class defining observables related to LateralProfile.

    Attributes:
        _energies_per_event: A numpy array with shape = (NE, R) where NE stays for a number of events. An
            element [i, j] is a sum of energies detected in all cells located in a jth layer for an ith event.
        _total_energy_per_event: A numpy array with shape = (NE, ). An element [i] is a sum of energies detected in all
            cells for an ith event.
        _w: A numpy array = [0, 1, ..., R - 1] which represents weights used in computation of first and second moment.

    """

    def __post_init__(self):
        self._energies_per_event = np.sum(self._input, axis=(2, 3))
        self._total_energy_per_event = np.sum(self._energies_per_event, axis=1)
        self._w = np.arange(N_CELLS_R)

    def calc_profile(self) -> np.ndarray:
        """ Calculates a lateral profile.

        A lateral profile for a given layer l (l = 0, ..., R - 1) is defined as:
        sum_{i = 0}^{NE - 1} energy_per_event[i, l].

        Returns:
            A numpy array of longitudinal profiles for each layer with a shape = (R, ).

        """
        return np.sum(self._energies_per_event, axis=0)

    def calc_first_moment(self) -> np.ndarray:
        """ Calculates a first moment of profile.

        A first moment of a lateral profile for a given event e (e = 0, ..., NE - 1) is defined as:
        FM[e] = alpha * (sum_{i = 0}^{R - 1} energies_per_event[e, i] * w[i]) / total_energy_per_event[e], where
        w = [0, 1, 2, ..., R - 1],
        alpha = SIZE_R defined in core/constants.py.

        Returns:
            A numpy array of first moments of lateral profiles for each event with a shape = (NE, ).

        """
        return SIZE_R * np.dot(self._energies_per_event, self._w) / self._total_energy_per_event

    def calc_second_moment(self) -> np.ndarray:
        """ Calculates a second moment of a lateral profile.

        A second moment of a lateral profile for a given event e (e = 0, ..., NE - 1) is defined as:
        SM[e] = (sum_{i = 0}^{R - 1} (w[i] - alpha - FM[e])^2 * energies_per_event[e, i]) total_energy_per_event[e],
        where
        w = [0, 1, 2, ..., R - 1],
        alpha = SIZE_R defined in ochre/constants.py

        Returns:
            A numpy array of second moments of lateral profiles for each event with a shape = (NE, ).
        """
        first_moment = self.calc_first_moment()
        first_moment = np.expand_dims(first_moment, axis=1)
        w = np.expand_dims(self._w, axis=0)
        # w has now a shape = [1, R] and first moment has a shape = [NE, 1]. There is a broadcasting in the line
        # below how that one create an array with a shape = [NE, R]
        return np.sum(np.multiply(np.power(w * SIZE_R - first_moment, 2), self._energies_per_event),
                      axis=1) / self._total_energy_per_event


@dataclass
class Energy(Observable):
    """ A class defining observables total energy per event and cell energy.

    """

    def calc_total_energy(self):
        """ Calculates total energy detected in an event.

        Total energy for a given event e (e = 0, ..., NE - 1) is defined as a sum of energies detected in all cells
        for this event.

        Returns:
            A numpy array of total energy values with shape = (NE, ).
        """
        return np.sum(self._input, axis=(1, 2, 3))

    def calc_cell_energy(self):
        """ Calculates cell energy.

        Cell energy for a given event (e = 0, ..., NE - 1) is defined by an array with shape (R * PHI * Z) storing
        values of energy in particular cells.

        Returns:
            A numpy array of cell energy values with shape = (NE * R * PHI * Z, ).

        """
        return np.copy(self._input).reshape(-1)

    def calc_energy_per_layer(self):
        """ Calculates total energy detected in a particular layer.

        Energy per layer for a given event (e = 0, ..., NE - 1) is defined by an array with shape (Z, ) storing
        values of total energy detected in a particular layer

        Returns:
            A numpy array of cell energy values with shape = (NE, Z).

        """
        return np.sum(self._input, axis=(1, 2))





plt.rcParams.update({"font.size": 22})


@dataclass
class Plotter:
    """ An abstract class defining interface of all plotters.

    Do not use this class directly. Use ProfilePlotter or EnergyPlotter instead.

    Attributes:
        _particle_energy: An integer which is energy of the primary particle in GeV units.
        _particle_angle: An integer which is an angle of the primary particle in degrees.
        _geometry: A string which is a name of the calorimeter geometry (e.g. SiW, SciPb).

    """
    _particle_energy: int
    _particle_angle: int
    _geometry: str

    def plot_and_save(self):
        pass


def _gaussian(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
    """ Computes a value of a Gaussian.

    Args:
        x: An argument of a function.
        a: A scaling parameter.
        mu: A mean.
        sigma: A variance.

    Returns:
        A value of a function for given arguments.

    """
    return a * np.exp(-((x - mu)**2 / (2 * sigma**2)))


def _best_fit(data: np.ndarray,
              bins: np.ndarray,
              hist: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """ Finds estimated shape of a Gaussian using Use non-linear least squares.

    Args:
        data: A numpy array with values of observables from multiple events.
        bins: A numpy array specifying histogram bins.
        hist: If histogram is calculated. Then data is the frequencies.

    Returns:
        A tuple of two lists. Xs and Ys of predicted curve.

    """
    # Calculate histogram.
    if not hist:
        hist, _ = np.histogram(data, bins)
    else:
        hist = data

    # Choose only those bins which are nonzero. Nonzero() return a tuple of arrays. In this case it has a length = 1,
    # hence we are interested in its first element.
    indices = hist.nonzero()[0]

    # Based on previously chosen nonzero bin, calculate position of xs and ys_bar (true values) which will be used in
    # fitting procedure. Len(bins) == len(hist + 1), so we choose middles of bins as xs.
    bins_middles = (bins[:-1] + bins[1:]) / 2
    xs = bins_middles[indices]
    ys_bar = hist[indices]

    # Set initial parameters for curve fitter.
    a0 = np.max(ys_bar)
    mu0 = np.mean(xs)
    sigma0 = np.var(xs)

    # Fit a Gaussian to the prepared data.
    (a, mu, sigma), _ = curve_fit(f=_gaussian,
                                  xdata=xs,
                                  ydata=ys_bar,
                                  p0=[a0, mu0, sigma0],
                                  method="trf",
                                  maxfev=1000)

    # Calculate values of an approximation in given points and return values.
    ys = _gaussian(xs, a, mu, sigma)
    return xs, ys


@dataclass
class ProfilePlotter(Plotter):
    """ Plotter responsible for preparing plots of profiles and their first and second moments.

    Attributes:
        _full_simulation: A numpy array representing a profile of data generated by Geant4.
        _ml_simulation: A numpy array representing a profile of data generated by ML model.
        _plot_gaussian: A boolean. Decides whether first and second moment should be plotted as a histogram or
            a fitted gaussian.
        _profile_type: An enum. A profile can be either lateral or longitudinal.

    """
    _full_simulation: Profile
    _ml_simulation: Profile
    _plot_gaussian: bool = False

    def __post_init__(self):
        # Check if profiles are either both longitudinal or lateral.
        full_simulation_type = type(self._full_simulation)
        ml_generation_type = type(self._ml_simulation)
        assert full_simulation_type == ml_generation_type, "Both profiles within a ProfilePlotter must be the same " \
                                                           "type."

        # Set an attribute with profile type.
        if full_simulation_type == LongitudinalProfile:
            self._profile_type = ProfileType.LONGITUDINAL
        else:
            self._profile_type = ProfileType.LATERAL

    def _plot_and_save_customizable_histogram(
            self,
            full_simulation: np.ndarray,
            ml_simulation: np.ndarray,
            bins: np.ndarray,
            xlabel: str,
            observable_name: str,
            plot_profile: bool = False,
            y_log_scale: bool = False) -> None:
        """ Prepares and saves a histogram for a given pair of observables.

        Args:
            full_simulation: A numpy array of observables coming from full simulation.
            ml_simulation: A numpy array of observables coming from ML simulation.
            bins: A numpy array specifying histogram bins.
            xlabel: A string. Name of x-axis on the plot.
            observable_name: A string. Name of plotted observable.
            plot_profile: A boolean. If set to True, full_simulation and ml_simulation are histogram weights while x is
                defined by the number of layers. This means that in order to plot histogram (and gaussian), one first
                need to create a data repeating each layer or R index appropriate number of times. Should be set to True
                only while plotting profiles not first or second moments.
            y_log_scale: A boolean. Used log scale on y-axis is set to True.

        Returns:
            None.

        """
        fig, axes = plt.subplots(2,
                                 1,
                                 figsize=(15, 10),
                                 clear=True,
                                 sharex="all")

        # Plot histograms.
        if plot_profile:
            # We already have the bins (layers) and freqencies (energies),
            # therefore directly plotting a step plot + lines instead of a hist plot.
            axes[0].step(bins[:-1],
                         full_simulation,
                         label="FullSim",
                         color=FULL_SIM_HISTOGRAM_COLOR)
            axes[0].step(bins[:-1],
                         ml_simulation,
                         label="MLSim",
                         color=ML_SIM_HISTOGRAM_COLOR)
            axes[0].vlines(x=bins[0],
                           ymin=0,
                           ymax=full_simulation[0],
                           color=FULL_SIM_HISTOGRAM_COLOR)
            axes[0].vlines(x=bins[-2],
                           ymin=0,
                           ymax=full_simulation[-1],
                           color=FULL_SIM_HISTOGRAM_COLOR)
            axes[0].vlines(x=bins[0],
                           ymin=0,
                           ymax=ml_simulation[0],
                           color=ML_SIM_HISTOGRAM_COLOR)
            axes[0].vlines(x=bins[-2],
                           ymin=0,
                           ymax=ml_simulation[-1],
                           color=ML_SIM_HISTOGRAM_COLOR)
            axes[0].set_ylim(0, None)

            # For using it later for the ratios.
            energy_full_sim, energy_ml_sim = full_simulation, ml_simulation
        else:
            energy_full_sim, _, _ = axes[0].hist(
                x=full_simulation,
                bins=bins,
                label="FullSim",
                histtype=HISTOGRAM_TYPE,
                color=FULL_SIM_HISTOGRAM_COLOR)
            energy_ml_sim, _, _ = axes[0].hist(x=ml_simulation,
                                               bins=bins,
                                               label="MLSim",
                                               histtype=HISTOGRAM_TYPE,
                                               color=ML_SIM_HISTOGRAM_COLOR)

        # Plot Gaussians if needed.
        if self._plot_gaussian:
            if plot_profile:
                (xs_full_sim, ys_full_sim) = _best_fit(full_simulation,
                                                       bins,
                                                       hist=True)
                (xs_ml_sim, ys_ml_sim) = _best_fit(ml_simulation,
                                                   bins,
                                                   hist=True)
            else:
                (xs_full_sim, ys_full_sim) = _best_fit(full_simulation, bins)
                (xs_ml_sim, ys_ml_sim) = _best_fit(ml_simulation, bins)
            axes[0].plot(xs_full_sim,
                         ys_full_sim,
                         color=FULL_SIM_GAUSSIAN_COLOR,
                         label="FullSim")
            axes[0].plot(xs_ml_sim,
                         ys_ml_sim,
                         color=ML_SIM_GAUSSIAN_COLOR,
                         label="MLSim")

        if y_log_scale:
            axes[0].set_yscale("log")
        axes[0].legend(loc="best")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel("Energy [Mev]")
        axes[0].set_title(
            f" $e^-$, {self._particle_energy} [GeV], {self._particle_angle}$^{{\circ}}$, {self._geometry}"
        )

        # Calculate ratios.
        ratio = np.divide(energy_ml_sim,
                          energy_full_sim,
                          out=np.ones_like(energy_ml_sim),
                          where=(energy_full_sim != 0))
        # Since len(bins) == 1 + data, we calculate middles of bins as xs.
        bins_middles = (bins[:-1] + bins[1:]) / 2
        axes[1].plot(bins_middles, ratio, "-o")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel("MLSim/FullSim")
        axes[1].axhline(y=1, color="black")
        plt.savefig(
            f"{VALID_DIR}/{observable_name}_Geo_{self._geometry}_E_{self._particle_energy}_"
            + f"Angle_{self._particle_angle}.png")
        plt.clf()

    def _plot_profile(self) -> None:
        """ Plots profile of an observable.

        Returns:
            None.

        """
        full_simulation_profile = self._full_simulation.calc_profile()
        ml_simulation_profile = self._ml_simulation.calc_profile()
        if self._profile_type == ProfileType.LONGITUDINAL:
            # matplotlib will include the right-limit for the last bar,
            # hence extending by 1.
            bins = np.linspace(0, N_CELLS_Z, N_CELLS_Z + 1)
            observable_name = "LongProf"
            xlabel = "Layer index"
        else:
            bins = np.linspace(0, N_CELLS_R, N_CELLS_R + 1)
            observable_name = "LatProf"
            xlabel = "R index"
        self._plot_and_save_customizable_histogram(full_simulation_profile,
                                                   ml_simulation_profile,
                                                   bins,
                                                   xlabel,
                                                   observable_name,
                                                   plot_profile=True)

    def _plot_first_moment(self) -> None:
        """ Plots and saves a first moment of an observable's profile.

        Returns:
            None.

        """
        full_simulation_first_moment = self._full_simulation.calc_first_moment(
        )
        ml_simulation_first_moment = self._ml_simulation.calc_first_moment()
        if self._profile_type == ProfileType.LONGITUDINAL:
            xlabel = "$<\lambda> [mm]$"
            observable_name = "LongFirstMoment"
            bins = np.linspace(0, 0.4 * N_CELLS_Z * SIZE_Z, 128)
        else:
            xlabel = "$<r> [mm]$"
            observable_name = "LatFirstMoment"
            bins = np.linspace(0, 0.75 * N_CELLS_R * SIZE_R, 128)

        self._plot_and_save_customizable_histogram(
            full_simulation_first_moment, ml_simulation_first_moment, bins,
            xlabel, observable_name)

    def _plot_second_moment(self) -> None:
        """ Plots and saves a second moment of an observable's profile.

        Returns:
            None.

        """
        full_simulation_second_moment = self._full_simulation.calc_second_moment(
        )
        ml_simulation_second_moment = self._ml_simulation.calc_second_moment()
        if self._profile_type == ProfileType.LONGITUDINAL:
            xlabel = "$<\lambda^{2}> [mm^{2}]$"
            observable_name = "LongSecondMoment"
            bins = np.linspace(0, pow(N_CELLS_Z * SIZE_Z, 2) / 35., 128)
        else:
            xlabel = "$<r^{2}> [mm^{2}]$"
            observable_name = "LatSecondMoment"
            bins = np.linspace(0, pow(N_CELLS_R * SIZE_R, 2) / 8., 128)

        self._plot_and_save_customizable_histogram(
            full_simulation_second_moment, ml_simulation_second_moment, bins,
            xlabel, observable_name)

    def plot_and_save(self) -> None:
        """ Main plotting function.

        Calls private methods and prints the information about progress.

        Returns:
            None.

        """
        if self._profile_type == ProfileType.LONGITUDINAL:
            profile_type_name = "longitudinal"
        else:
            profile_type_name = "lateral"
        print(f"Plotting the {profile_type_name} profile...")
        self._plot_profile()
        print(f"Plotting the first moment of {profile_type_name} profile...")
        self._plot_first_moment()
        print(f"Plotting the second moment of {profile_type_name} profile...")
        self._plot_second_moment()


@dataclass
class EnergyPlotter(Plotter):
    """ Plotter responsible for preparing plots of profiles and their first and second moments.

    Attributes:
        _full_simulation: A numpy array representing a profile of data generated by Geant4.
        _ml_simulation: A numpy array representing a profile of data generated by ML model.

    """
    _full_simulation: Energy
    _ml_simulation: Energy

    def _plot_total_energy(self, y_log_scale=True) -> None:
        """ Plots and saves a histogram with total energy detected in an event.

        Args:
            y_log_scale: A boolean. Used log scale on y-axis is set to True.

        Returns:
            None.

        """
        full_simulation_total_energy = self._full_simulation.calc_total_energy(
        )
        ml_simulation_total_energy = self._ml_simulation.calc_total_energy()

        plt.figure(figsize=(12, 8))
        bins = np.linspace(
            np.min(full_simulation_total_energy) -
            np.min(full_simulation_total_energy) * 0.05,
            np.max(full_simulation_total_energy) +
            np.max(full_simulation_total_energy) * 0.05, 50)
        plt.hist(x=full_simulation_total_energy,
                 histtype=HISTOGRAM_TYPE,
                 label="FullSim",
                 bins=bins,
                 color=FULL_SIM_HISTOGRAM_COLOR)
        plt.hist(x=ml_simulation_total_energy,
                 histtype=HISTOGRAM_TYPE,
                 label="MLSim",
                 bins=bins,
                 color=ML_SIM_HISTOGRAM_COLOR)
        plt.legend(loc="upper left")
        if y_log_scale:
            plt.yscale("log")
        plt.xlabel("Energy [MeV]")
        plt.ylabel("# events")
        plt.title(
            f" $e^-$, {self._particle_energy} [GeV], {self._particle_angle}$^{{\circ}}$, {self._geometry} "
        )
        plt.savefig(
            f"{VALID_DIR}/E_tot_Geo_{self._geometry}_E_{self._particle_energy}_Angle_{self._particle_angle}.png"
        )
        plt.clf()

    def _plot_cell_energy(self) -> None:
        """ Plots and saves a histogram with number of detector's cells across whole
        calorimeter with particular energy detected.

        Returns:
            None.

        """
        full_simulation_cell_energy = self._full_simulation.calc_cell_energy()
        ml_simulation_cell_energy = self._ml_simulation.calc_cell_energy()

        log_full_simulation_cell_energy = np.log10(
            full_simulation_cell_energy,
            out=np.zeros_like(full_simulation_cell_energy),
            where=(full_simulation_cell_energy != 0))
        log_ml_simulation_cell_energy = np.log10(
            ml_simulation_cell_energy,
            out=np.zeros_like(ml_simulation_cell_energy),
            where=(ml_simulation_cell_energy != 0))
        plt.figure(figsize=(12, 8))
        bins = np.linspace(-4, 1, 1000)
        plt.hist(x=log_full_simulation_cell_energy,
                 bins=bins,
                 histtype=HISTOGRAM_TYPE,
                 label="FullSim",
                 color=FULL_SIM_HISTOGRAM_COLOR)
        plt.hist(x=log_ml_simulation_cell_energy,
                 bins=bins,
                 histtype=HISTOGRAM_TYPE,
                 label="MLSim",
                 color=ML_SIM_HISTOGRAM_COLOR)
        plt.xlabel("log10(E/MeV)")
        plt.ylim(bottom=1)
        plt.yscale("log")
        plt.ylim(bottom=1)
        plt.ylabel("# entries")
        plt.title(
            f" $e^-$, {self._particle_energy} [GeV], {self._particle_angle}$^{{\circ}}$, {self._geometry} "
        )
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.savefig(
            f"{VALID_DIR}/E_cell_Geo_{self._geometry}_E_{self._particle_energy}_Angle_{self._particle_angle}.png"
        )
        plt.clf()

    def _plot_energy_per_layer(self):
        """ Plots and saves N_CELLS_Z histograms with total energy detected in particular layers.

        Returns:
            None.

        """
        full_simulation_energy_per_layer = self._full_simulation.calc_energy_per_layer(
        )
        ml_simulation_energy_per_layer = self._ml_simulation.calc_energy_per_layer(
        )

        number_of_plots_in_row = 9
        number_of_plots_in_column = 5

        bins = np.linspace(np.min(full_simulation_energy_per_layer - 10),
                           np.max(full_simulation_energy_per_layer + 10), 25)

        fig, ax = plt.subplots(number_of_plots_in_column,
                               number_of_plots_in_row,
                               figsize=(20, 15),
                               sharex="all",
                               sharey="all",
                               constrained_layout=True)

        for layer_nb in range(N_CELLS_Z):
            i = layer_nb // number_of_plots_in_row
            j = layer_nb % number_of_plots_in_row

            ax[i][j].hist(full_simulation_energy_per_layer[:, layer_nb],
                          histtype=HISTOGRAM_TYPE,
                          label="FullSim",
                          bins=bins,
                          color=FULL_SIM_HISTOGRAM_COLOR)
            ax[i][j].hist(ml_simulation_energy_per_layer[:, layer_nb],
                          histtype=HISTOGRAM_TYPE,
                          label="MLSim",
                          bins=bins,
                          color=ML_SIM_HISTOGRAM_COLOR)
            ax[i][j].set_title(f"Layer {layer_nb}", fontsize=13)
            ax[i][j].set_yscale("log")
            ax[i][j].tick_params(axis='both', which='major', labelsize=10)

        fig.supxlabel("Energy [MeV]", fontsize=14)
        fig.supylabel("# entries", fontsize=14)
        fig.suptitle(
            f" $e^-$, {self._particle_energy} [GeV], {self._particle_angle}$^{{\circ}}$, {self._geometry} "
        )

        # Take legend from one plot and make it a global legend.
        handles, labels = ax[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.15, 0.5))

        plt.savefig(
            f"{VALID_DIR}/E_layer_Geo_{self._geometry}_E_{self._particle_energy}_Angle_{self._particle_angle}.png",
            bbox_inches="tight")
        plt.clf()

    def plot_and_save(self):
        """ Main plotting function.

        Calls private methods and prints the information about progress.

        Returns:
            None.

        """
        print("Plotting total energy...")
        self._plot_total_energy()
        print("Plotting cell energy...")
        self._plot_cell_energy()
        print("Plotting energy per layer...")
        self._plot_energy_per_layer()







# main function
def main():
    # Parse commandline arguments
    particle_energy = "1GeV"
    particle_angle = "90°"
    geometry = "PbWO4"
    # 1. Full simulation data loading
    # Load energy of showers from a single geometry, energy and angle
    e_layer_g4 = showers
    # 2. Fast simulation data loading, scaling to original energy range & reshaping
    # Reshape the events into 3D
    e_layer_vae = regenerated_samples

    print("Data has been loaded.")

    # 3. Create observables from raw data.
    full_sim_long = LongitudinalProfile(_input=e_layer_g4)
    full_sim_lat = LateralProfile(_input=e_layer_g4)
    full_sim_energy = Energy(_input=e_layer_g4)
    ml_sim_long = LongitudinalProfile(_input=e_layer_vae)
    ml_sim_lat = LateralProfile(_input=e_layer_vae)
    ml_sim_energy = Energy(_input=e_layer_vae)

    print("Created observables.")

    # 4. Plot observables
    longitudinal_profile_plotter = ProfilePlotter(particle_energy, particle_angle, geometry, full_sim_long, ml_sim_long,
                                                  _plot_gaussian=False)
    lateral_profile_plotter = ProfilePlotter(particle_energy, particle_angle,
                                             geometry, full_sim_lat, ml_sim_lat, _plot_gaussian=False)
    energy_plotter = EnergyPlotter(particle_energy, particle_angle, geometry, full_sim_energy, ml_sim_energy)

    longitudinal_profile_plotter.plot_and_save()
    lateral_profile_plotter.plot_and_save()
    energy_plotter.plot_and_save()
    print("Done.")


if __name__ == "__main__":
    exit(main())
