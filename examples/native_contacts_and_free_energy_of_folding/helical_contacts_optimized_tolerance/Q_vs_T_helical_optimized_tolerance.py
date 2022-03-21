import os
import pickle
import time

from cg_openmm.parameters.free_energy import *
from cg_openmm.parameters.secondary_structure import *
from openmm import unit

# Using a helical backbone native contact scheme, determine the tolerance factor which
# maximizes Q_folded-Q_unfolded, and then calculate native contact fraction and free
# energy of folding as functions of T.

# Specify location of output .nc files
data_directory = '../../run_replica_exchange/output'

# Load in cgmodel:
cgmodel = pickle.load(open("../../run_replica_exchange/stored_cgmodel.pkl","rb"))

# Load in trajectory stats:
analysis_stats = pickle.load(open("../../run_replica_exchange/analysis_stats_discard_20ns.pkl","rb"))   

# Specify native structure file:
native_structure_file = "../native_medoid_min.dcd"

# Create list of replica trajectory files:
dcd_file_list_rep = []
number_replicas = 12

for rep in range(number_replicas):
    dcd_file_list_rep.append(f"{data_directory}/replica_{rep+1}.dcd")

# Determine helical native contacts and optimize tolerance factor:
(opt_seq_spacing, native_contact_tol, opt_results, Q_expect_results,
sigmoid_param_opt, sigmoid_param_cov)  = optimize_Q_tol_helix(
    cgmodel,
    native_structure_file,
    dcd_file_list_rep,
    output_data=f"{data_directory}/output.nc",
    num_intermediate_states=3,
    frame_begin=analysis_stats["production_start"],
    frame_stride=analysis_stats["energy_decorrelation"],
    plotfile="Q_vs_T_helical_opt_tol.pdf",
    verbose=True,
    backbone_type_name='bb',
    brute_step=0.05,
    )
    
# Get the native contact list and distances:
native_contact_list, native_contact_distances, opt_seq_spacing = get_helix_contacts(
    cgmodel,
    native_structure_file,
    backbone_type_name='bb',
)

print(f'Optimal native contacts are i to i+{opt_seq_spacing}')
print(f'Using fixed tolerance factor of {native_contact_tol}')

# Bootstrap native contacts to get Q_folded and uncertainties
temp_list, Q_values, Q_uncertainty, sigmoid_results_boot = bootstrap_native_contacts_expectation(
    cgmodel,
    dcd_file_list_rep,
    native_contact_list,
    native_contact_distances,
    output_data=f"{data_directory}/output.nc",
    frame_begin=analysis_stats["production_start"],
    sample_spacing=analysis_stats["energy_decorrelation"],
    native_contact_tol=native_contact_tol,
    num_intermediate_states=3,
    n_trial_boot=200,
    conf_percent='sigma',
    plotfile='native_contacts_boot200_opt_tol.pdf',
    )

Q_folded = sigmoid_results_boot['Q_folded_value']
print(f'Q_folded: {Q_folded}')

#------------------------------------#
# Free energy of folding calculation #
#------------------------------------#

# Using optimized parameters, compute the free energy of folding
# Determine native contact fraction of current trajectories:
Q, Q_avg, Q_stderr, decorrelation_time = fraction_native_contacts(
    cgmodel,
    dcd_file_list_rep,
    native_contact_list,
    native_contact_distances,
    frame_begin=analysis_stats["production_start"],
    native_contact_tol=native_contact_tol,
)

plot_native_contact_timeseries(
    Q,
    frame_begin=analysis_stats["production_start"],
    time_interval=1*unit.picosecond,
    plot_per_page=3,
    plotfile="Q_vs_time_opt_tol.pdf",
    figure_title="Native contact fraction",
)

# Compute free energy/entropy/enthalpy of folding curves
    
G_unit = unit.kilojoule / unit.mole 
S_unit = G_unit / unit.kelvin
H_unit = G_unit
    
# From bootstrapping:
(full_T_list_boot, deltaG_values_boot, deltaG_uncertainty_boot, \
    deltaS_values_boot, deltaS_uncertainty_boot, \
    deltaH_values_boot, deltaH_uncertainty_boot) = bootstrap_free_energy_folding(
    Q,
    Q_folded,
    frame_begin=analysis_stats["production_start"],
    sample_spacing=analysis_stats["energy_decorrelation"],
    output_data=f"{data_directory}/output.nc",
    num_intermediate_states=3,
    n_trial_boot=200,
    conf_percent='sigma',
    plotfile_dir=os.getcwd(),
)

deltaG_values_boot = deltaG_values_boot['state0_state1'].value_in_unit(G_unit)
deltaG_uncertainty_boot = deltaG_uncertainty_boot['state0_state1'][1].value_in_unit(G_unit)

deltaS_values_boot = deltaS_values_boot['state0_state1'].value_in_unit(S_unit)
deltaS_uncertainty_boot = deltaS_uncertainty_boot['state0_state1'][1].value_in_unit(S_unit)

deltaH_values_boot = deltaH_values_boot['state0_state1'].value_in_unit(H_unit)
deltaH_uncertainty_boot = deltaH_uncertainty_boot['state0_state1'][1].value_in_unit(H_unit)

print(f"T (K), deltaG (kJ/mol), deltaG_uncertainty (kJ/mol)")
for i in range(len(full_T_list_boot)):
    print(f"{full_T_list_boot[i].value_in_unit(unit.kelvin):>6.4f}, \
        {deltaG_values_boot[i]:>6.8f}, \
        {deltaG_uncertainty_boot[i]:>6.8f}")
        
print(f"\nT (K), deltaS (kJ/mol/K), deltaS_uncertainty (kJ/mol/K)")
for i in range(len(full_T_list_boot)):
    print(f"{full_T_list_boot[i].value_in_unit(unit.kelvin):>6.4f}, \
        {deltaS_values_boot[i]:>6.8f}, \
        {deltaS_uncertainty_boot[i]:>6.8f}")
        
print(f"\nT (K), deltaH (kJ/mol), deltaH_uncertainty_boot (kJ/mol)")
for i in range(len(full_T_list_boot)):
    print(f"{full_T_list_boot[i].value_in_unit(unit.kelvin):>6.4f}, \
        {deltaH_values_boot[i]:>6.8f}, \
        {deltaH_uncertainty_boot[i]:>6.8f}")