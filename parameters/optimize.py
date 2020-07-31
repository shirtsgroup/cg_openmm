import numpy as np
import matplotlib.pyplot as pyplot
import mdtraj as md
import simtk.unit as unit
from statistics import mean
from cg_openmm.simulation.tools import *

# from genetic_selection import GeneticSelectionCV


def optimize_model_parameter(optimization_target):
    """
        """

    return


def get_fwhm_symmetry(C_v, T_list):
    """
        """

    symmetry_list = []
    fwhm_list = []

    for i in range(1, len(C_v) - 1):
        if C_v[i] >= C_v[i - 1] and C_v[i] >= C_v[i + 1]:
            max_value = C_v[i]
            max_value_T = T_list[i]
            half_max = 0.5 * max_value

            for j in range(i, 0, -1):
                if C_v[j] < half_max:
                    break
            # interpolate to get lower HM
            lower_C = C_v[j]
            lower_T = T_list[j]
            upper_C = C_v[j + 1]
            upper_T = T_list[j + 1]

            delta_T = upper_T - lower_T
            delta_C = upper_C - lower_C
            lower_delta_C = half_max - lower_C

            delta_hm = lower_delta_C / delta_C

            lower_hm_T = delta_hm * delta_T

            for j in range(i, len(C_v)):
                if C_v[j] < half_max:
                    break
            # interpolate to get lower HM
            lower_C = C_v[j]
            lower_T = T_list[j - 1]
            upper_C = C_v[j - 1]
            upper_T = T_list[j]

            delta_T = upper_T - lower_T
            delta_C = upper_C - lower_C
            upper_delta_C = C_v[j - 1] - half_max

            delta_hm = upper_delta_C / delta_C

            upper_hm_T = delta_hm * delta_T

            fwhm = upper_hm_T - lower_hm_T
            fwhm_list.append(fwhm)

            lower_delta = max_value_T - lower_hm_T
            upper_delta = upper_hm_T - max_value_T

            symmetry = abs(lower_delta - upper_delta)

            symmetry_list.append(symmetry)

    print("Symmetry = " + str(symmetry_list))
    print("FWHM = " + str(fwhm_list))

    symmetry = mean(symmetry_list)
    fwhm = mean(fwhm_list)

    return (symmetry, fwhm)


def get_num_maxima(C_v):
    """
        """

    maxima = 0

    for i in range(1, len(C_v)):
        if C_v[i] >= C_v[i - 1] and C_v[i] >= C_v[i + 1]:
            maxima = maxima + 1

    return maxima


def calculate_C_v_fitness(C_v, T_list):
    """
        """
    num_maxima = get_num_maxima(C_v)
    symmetry = get_fwhm_symmetry(C_v, T_list)

    num_maxima_weight = 0.33
    symmetry_wieght = 0.33
    fwhm_weight = 0.33

    maxima_fitness = math.exp(-(num_maxima - 1) * num_maxima_weight)
    symmetry_fitness = math.exp(-(abs(symmetry - 1)) * symmetry_weight)
    fwhm_fitness = math.exp(-(fwhm) * fwhm_weight)

    fitness = maxima_fitness * symmetry_fitness * fwhm_fitness

    return fitness


def optimize_parameter(
    cgmodel, optimization_parameter, optimization_range_min, optimization_range_max, steps=None
):
    """
        """
    if steps == None:
        steps = 100
    step_size = (optimization_range_max - optimization_range_min) / 100
    parameter_values = [step * step_size for step in range(1, steps)]
    potential_energies = []
    for parameter in parameter_values:
        cgmodel.optimization_parameter = parameter
        positions, potential_energy, time_step = minimize_structure(
            cgmodel.topology,
            cgmodel.system,
            cgmodel.positions,
            temperature=300.0 * unit.kelvin,
            simulation_time_step=None,
            total_simulation_time=1.0 * unit.picosecond,
            output_pdb="minimum.pdb",
            output_data="minimization.dat",
            print_frequency=1,
        )
        potential_energies.append(potential_energy)

    best_value = min(potential_energies)

    return (best_value, potential_energies, parameter_values)
