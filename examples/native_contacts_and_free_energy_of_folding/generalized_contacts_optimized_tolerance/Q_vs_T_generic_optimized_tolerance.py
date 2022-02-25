import os
import pickle
import time

from cg_openmm.parameters.free_energy import *
from cg_openmm.parameters.secondary_structure import *
from openmm import unit

# Using a generalized backbone native contact scheme, determine the native contact pairwise
# distance cutoff and tolerance factor maximizes Q_folded-Q_unfolded using a 2D optimization
# scheme. Then, calculate native contact fraction and free energy of folding as functions of T.

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

# Optimize native contact cutoff and tolerance factor:
(native_contact_cutoff, native_contact_tol, opt_results, Q_expect_results, \
sigmoid_param_opt, sigmoid_param_cov, contact_type_dict) = optimize_Q_cut(
    cgmodel,
    native_structure_file,
    dcd_file_list_rep,
    num_intermediate_states=3,
    output_data=f"{data_directory}/output.nc",
    frame_begin=analysis_stats["production_start"],
    frame_stride=analysis_stats["energy_decorrelation"],
    verbose=True,
    plotfile='native_contacts_2d_opt.pdf',
    minimizer_options={'seed':17, 'maxiter':20, 'atol':0.1},
    )

# Get native contact list:
native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
    cgmodel,
    native_structure_file,
    native_contact_cutoff,
)    
    
print(f'Native contact summary: {contact_type_dict}')
print(f'Using native contact cutoff of {native_contact_cutoff}')
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
    plotfile='native_contacts_boot200_2d_opt.pdf',
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
    plotfile="Q_vs_time_2d_opt.pdf",
    figure_title="Native contact fraction",
)

# Compute free energy/entropy/enthalpy of folding curves
    
G_unit = unit.kilojoule / unit.mole 
S_unit = G_unit / unit.kelvin
H_unit = G_unit
    
# From bootstrapping:
(full_T_list_boot, deltaG_values_boot, deltaG_uncertainty_boot, \
    deltaS_values_boot, deltaS_uncertainty_boot, \
    deltaU_values_boot, deltaU_uncertainty_boot) = bootstrap_free_energy_folding(
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

deltaU_values_boot = deltaU_values_boot['state0_state1'].value_in_unit(H_unit)
deltaU_uncertainty_boot = deltaU_uncertainty_boot['state0_state1'][1].value_in_unit(H_unit)

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
        
print(f"\nT (K), deltaU (kJ/mol), deltaU_uncertainty_boot (kJ/mol)")
for i in range(len(full_T_list_boot)):
    print(f"{full_T_list_boot[i].value_in_unit(unit.kelvin):>6.4f}, \
        {deltaU_values_boot[i]:>6.8f}, \
        {deltaU_uncertainty_boot[i]:>6.8f}")