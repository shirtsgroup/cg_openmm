import numpy as np
import matplotlib.pyplot as pyplot
import mdtraj as md
import simtk.unit as unit
from statistics import mean
from cg_openmm.simulation.tools import *
from cg_openmm.parameters.evaluate_energy import *
from cg_openmm.thermo.calc import bootstrap_heat_capacity
from scipy.optimize import minimize, minimize_scalar

def optimize_force_field_parameters_Cv_FWHM(cgmodel, file_list, temperature_list, param_bounds_dict,
    frame_begin=0, frame_end=-1, frame_stride=1, output_data='output.nc',
    verbose=False, n_cpu=1, min_eff_samples=50,
    n_trial_boot=50, num_intermediate_states=0, plotfile="optimize_FWHM_iterations.pdf",
    min_method='TNC'):
    """
    Generalized function for optimizating a set of cgmodel force field parameters to minimize
    the full-width half-maximum (FWHM) of heat capacity as a function of temperature. Energy
    re-evaluation and MBAR reweighting is used in place of a full simulation whenever possible.
    
    :param cgmodel: CGModel() class object used to generate the original simulation energies
    :type cgmodel: class

    :param file_list: List of replica trajectory files to evaluate the energies of
    :type file_list: list or str

    :param temperature_list: List of temperatures associated with file_list
    :type temperature_list: List( float * simtk.unit.temperature )

    :param param_bounds_dict: dictionary containing force field parameter names and bounds
    :type param_bounds_dict: dict{'param_name': (bound_lo * simtk.unit, bound_hi * simtk.unit)}

    :param frame_begin: analyze starting from this frame, discarding all prior as equilibration period (default=0)
    :type frame_begin: int

    :param frame_end: analyze up to this frame only, discarding the rest (default=-1).
    :type frame_end: int

    :param frame_stride: advance by this many frames between each evaluation (default=1)
    :type frame_stride: int

    :param verbose: option to print out detailed per-particle parameter changes (default=False)
    :type verbose: Boolean
    
    :param n_cpu: number of cpus for running parallel energy evaluations (default=1)
    :type n_cpu: int
    
    :param min_eff_samples: minimum number of effective samples determined from the mbar object, below which a full simulation is needed
    :type min_eff_samples: int
    
    :param min_method: SciPy minimize method to use
    :type min_method: str
    
    :param num_intermediate_states: The number of states to insert between existing states in 'temperature_list' (default=0)
    :type num_intermediate_states: int
    
    :param plotfile: path to filename to output plot
    :type plotfile: str
    
    :returns:
        - opt_param_dict - a dictionary containing the optimized force field parameters
        - opt_results - scipy minimize results summary
        - opt_FWHM - value of heat capacity full-width half-maximum for the optimal parameters
    """

    # Parse the force field parameter change dict:
    x0 = []
    param_names = []
    bounds = []
    units = []
    
    for key,value in param_bounds_dict.items():
        # value should be [(bound_lo, bound_hi)]
        # key should be a valid force field parameter name
        param_names.append(key)
        # Every parameter except periodicity should have units
        units.append(value[0].unit)
        bounds.append((value[0].value_in_unit(units[-1]),value[1].value_in_unit(units[-1])))
        # Use mean value as starting guess:
        x0.append((value[1].value_in_unit(units[-1])+value[0].value_in_unit(units[-1]))/2)

    def get_reeval_FWHM(param_values, cgmodel, file_list, temperature_list, output_data,
        param_names, units, frame_begin, frame_stride, frame_end,
        n_cpu, n_trial_boot, num_intermediate_states):
        """
        Objective function to be minimized
        """

        # Construct dictionary of parameter update instructions:
        param_dict = {}
        print(f'Current parameter value: {param_values}')
        for i in range(len(param_names)):
            param_dict[param_names[i]] = param_values * units[i]

            # For multivariate we would use this:
            # param_dict[param_names[i]] = param_values[i] * units[i]
        
        # Re-evaluate energy with current force field parameters:
        # For bootstrapping, evaluate all frames between [frame_begin:frame_end], and only
        # apply the stride to the heat capacity part
        U_eval, simulation = eval_energy(
            cgmodel,
            file_list,
            temperature_list,
            param_dict,
            frame_begin=frame_begin,
            frame_stride=1,
            frame_end=frame_end,
            n_cpu=n_cpu,
            verbose=verbose,
            )

        # Evaluate heat capacity and full-width half-maximum from bootstrapping:
        (new_temperature_list,
        C_v_values, C_v_uncertainty,
        Tm_value, Tm_uncertainty,
        Cv_height_value, Cv_height_uncertainty,
        FWHM_value, FWHM_uncertainty,
        N_eff_values) = bootstrap_heat_capacity(
            U_kln=U_eval,
            output_data=output_data,
            frame_begin=frame_begin,
            frame_end=frame_end,
            sample_spacing=frame_stride,
            num_intermediate_states=num_intermediate_states,
            n_trial_boot=n_trial_boot,
            plot_file=f'heat_capacity_boot_{param_names[0]}_{param_values}.pdf',
        )

        print(f'Current FWHM: {FWHM_value} +/- {FWHM_uncertainty[0]}')
        return FWHM_value.value_in_unit(unit.kelvin)

    # Run optimization:

    if len(param_names) == 1:
        # Do scalar optimization:
        opt_results = minimize_scalar(get_reeval_FWHM, x0,
            args=(cgmodel, file_list, temperature_list, output_data, param_names, units,
            frame_begin, frame_stride, frame_end, n_cpu, n_trial_boot, num_intermediate_states),
            method='bounded',
            bounds=[bounds[0][0],bounds[0][1]],
            options={'maxiter': 5},
        )

    else:
        # Do multivariate optimization:
        opt_results = minimize(get_reeval_FWHM, x0,
            args=(cgmodel, file_list, temperature_list, output_data, param_names, units,
            frame_begin, frame_stride, frame_end, n_cpu, n_trial_boot, num_intermediate_states),
            method=min_method,
            bounds=bounds,
            options={'maxfun': 5},
        )    
        
    # TODO: plot the heat capacity curves at each iteration, and make a plot of all FWHM_values    

    # Construct dictionary of optimal parameters:
    opt_param_dict = {}    
    
    k = 0
    for key,value in param_bounds_dict.items():
        opt_param_dict[key] = opt_results.x[k] * units[k]
        k += 1
    
    return opt_param_dict, opt_results


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
