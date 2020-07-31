import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import *
from foldamers.thermo.calc import calculate_heat_capacity
from cg_openmm.simulation.rep_exch import *

# Job settings
top_directory = "output"
if not os.path.exists(top_directory):
    os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 5  # Number of steps to skip when printing output
total_simulation_time = 100.0 * unit.picosecond  # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
number_replicas = 200
min_temp = 10.0 * unit.kelvin
max_temp = 200.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
if total_steps > 10000:
    exchange_attempts = round(total_steps / 1000)
else:
    exchange_attempts = 10

bond_length = 7.5 * unit.angstrom
sigma_increments = 10
sigma_range = range(round(bond_length._value * 1.5), round(bond_length._value * 2.5))
delta = round((sigma_range[-1] - sigma_range[0]) / (sigma_increments - 1), 2)
sigma_list = [
    unit.Quantity(sigma_range[0] + (i * delta), bond_length.unit) for i in range(sigma_increments)
]

df_ij_list = []
ddf_ij_list = []
Delta_u_list = []
dDelta_u_list = []
Delta_s_list = []
dDelta_s_list = []
C_v_list = []
dC_v_list = []

for sigma in sigma_list:
    output_data = str(str(top_directory) + "/rep_ex_" + str(sigma) + ".nc")
    sigmas = {"bb_bb_sigma": sigma, "bb_sc_sigma": sigma, "sc_sc_sigma": sigma}
    cgmodel = CGModel(sigmas=sigmas)
    if not os.path.exists(output_data):
        print("Performing simulations and free energy analysis for a coarse grained model")
        print("with sigma values of " + str(sigma))
        replica_energies, replica_positions, replica_states = run_replica_exchange(
            cgmodel.topology,
            cgmodel.system,
            cgmodel.positions,
            temperature_list=temperature_list,
            simulation_time_step=simulation_time_step,
            total_simulation_time=total_simulation_time,
            print_frequency=print_frequency,
            output_data=output_data,
        )
        steps_per_stage = round(total_steps / exchange_attempts)
        plot_replica_exchange_energies(
            replica_energies,
            temperature_list,
            simulation_time_step,
            steps_per_stage=steps_per_stage,
        )
        plot_replica_exchange_summary(
            replica_states, temperature_list, simulation_time_step, steps_per_stage=steps_per_stage
        )
    else:
        replica_energies, replica_positions, replica_states = read_replica_exchange_data(
            system=cgmodel.system,
            topology=cgmodel.topology,
            temperature_list=temperature_list,
            output_data=output_data,
            print_frequency=print_frequency,
        )

    num_intermediate_states = 1
    mbar, E_kn, E_expect, dE_expect, new_temp_list = get_mbar_expectation(
        replica_energies, temperature_list, num_intermediate_states
    )

    mbar, E_kn, DeltaE_expect, dDeltaE_expect, new_temp_list = get_mbar_expectation(
        E_kn, temperature_list, num_intermediate_states, mbar=mbar, output="differences"
    )

    mbar, E_kn, E2_expect, dE2_expect, new_temp_list = get_mbar_expectation(
        E_kn ** 2, temperature_list, num_intermediate_states, mbar=mbar
    )

    df_ij, ddf_ij = get_free_energy_differences(mbar)
    df_ij_list.append(df_ij)
    ddf_ij_list.append(ddf_ij)

    C_v, dC_v = calculate_heat_capacity(
        E_expect,
        E2_expect,
        dE_expect,
        DeltaE_expect,
        dDeltaE_expect,
        df_ij,
        ddf_ij,
        new_temp_list,
        len(temperature_list),
        num_intermediate_states,
    )
    C_v_list.append(C_v)
    dC_v_list.append(dC_v)

    Delta_s, dDelta_s = get_entropy_differences(mbar)
    Delta_s_list.append(Delta_s)
    dDelta_s_list.append(dDelta_s)
    Delta_u, dDelta_u = get_enthalpy_differences(mbar)
    Delta_u_list.append(Delta_u)
    dDelta_u_list.append(dDelta_u)

file_name = str(str(top_directory) + "/free_energies.png")
figure = pyplot.figure(1)
original_temperature_list = np.array([temperature._value for temperature in temperature_list])
try:
    temperatures = np.array([temperature._value for temperature in new_temp_list])
except:
    temperatures = np.array([temperature for temperature in new_temp_list])
legend_labels = [
    str("$\sigma / r_{bond}$= " + str(round(sigma._value / bond_length._value, 2)))
    for sigma in sigma_list
]

for df_ij, ddf_ij in zip(df_ij_list, ddf_ij_list):
    df_ij = np.array([df_ij[i][0] for i in range(len(df_ij))])
    ddf_ij = np.array([ddf_ij[i][0] for i in range(len(ddf_ij))])
    pyplot.errorbar(temperatures, df_ij, yerr=ddf_ij, figure=figure)

pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("Dimensionless free energy difference")
pyplot.title("Dimensionless free energy differences for variable $\sigma / r_{bond}$")
pyplot.legend(legend_labels)
pyplot.xlim(10.0, 25.0)
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

file_name = str(str(top_directory) + "/entropies.png")
figure = pyplot.figure(2)
original_temperature_list = np.array([temperature._value for temperature in temperature_list])
try:
    temperatures = np.array([temperature._value for temperature in new_temp_list])
except:
    temperatures = np.array([temperature for temperature in new_temp_list])
legend_labels = [
    str("$\sigma / r_{bond}$= " + str(round(sigma._value / bond_length._value, 2)))
    for sigma in sigma_list
]

for Delta_s, dDelta_s in zip(Delta_s_list, dDelta_s_list):
    Delta_s = np.array([Delta_s[i][0] for i in range(len(Delta_s))])
    dDelta_s = np.array([dDelta_s[i][0] for i in range(len(dDelta_s))])
    pyplot.errorbar(temperatures, Delta_s, yerr=dDelta_s, figure=figure)

pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("Dimensionless Entropy Differences (S)")
pyplot.title("Dimensionless entropy differences for variable $\sigma / r_{bond}$")
pyplot.legend(legend_labels)
pyplot.xlim(10.0, 25.0)
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

file_name = str(str(top_directory) + "/enthalpies.png")
figure = pyplot.figure(3)
original_temperature_list = np.array([temperature._value for temperature in temperature_list])
try:
    temperatures = np.array([temperature._value for temperature in new_temp_list])
except:
    temperatures = np.array([temperature for temperature in new_temp_list])
legend_labels = [
    str("$\sigma / r_{bond}$= " + str(round(sigma._value / bond_length._value, 2)))
    for sigma in sigma_list
]

for Delta_u, dDelta_u in zip(Delta_u_list, dDelta_u_list):
    Delta_u = np.array([Delta_u[i][0] for i in range(len(Delta_u))])
    dDelta_u = np.array([dDelta_u[i][0] for i in range(len(dDelta_u))])
    pyplot.errorbar(temperatures, Delta_u, yerr=dDelta_u, figure=figure)

pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("Dimensionless enthalpy differences (U)")
pyplot.title("Dimensionless enthalpy differences for variable $\sigma / r_{bond}$")
pyplot.legend(legend_labels)
pyplot.xlim(10.0, 25.0)
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

file_name = str(str(top_directory) + "/heat_capacity.png")
figure = pyplot.figure(4)
original_temperature_list = np.array([temperature._value for temperature in temperature_list])
try:
    temperatures = np.array([temperature._value for temperature in new_temp_list])
except:
    temperatures = np.array([temperature for temperature in new_temp_list])
legend_labels = [
    str("$\sigma / r_{bond}$= " + str(round(i / bond_length._value, 2))) for i in sigma_range
]

for C_v, dC_v in zip(C_v_list, dC_v_list):
    C_v = np.array([C_v[i][0] for i in range(len(C_v))])
    dC_v = np.array([dC_v[i][0] for i in range(len(dC_v))])
    pyplot.errorbar(temperatures, C_v, yerr=dC_v, figure=figure)

pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("C$_v$ ( kcal/mol * Kelvin )")
pyplot.title("Heat capacity for variable $\sigma / r_{bond}$")
pyplot.legend(legend_labels)
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

exit()
