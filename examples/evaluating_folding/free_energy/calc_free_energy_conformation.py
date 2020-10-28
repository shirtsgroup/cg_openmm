# This example script demonstrates the calculation of free energy differences
# between 2 conformational states from a replica exchange simulation, which are
# classified by their RMSD using a Kmeans clustering algorithm.

import os
import numpy as np
import matplotlib.pyplot as pyplot
import mdtraj as md
from simtk import unit
from cg_openmm.parameters.secondary_structure import *
from cg_openmm.parameters.free_energy import *
from cg_openmm.parameters.reweight import *
from cg_openmm.utilities.util import fit_sigmoid
from analyze_foldamers.ensembles.cluster import *
import pickle

output_directory = "output"
output_data = os.path.join(output_directory, "output.nc")
number_replicas = 36
min_temp = 50.0 * unit.kelvin
max_temp = 400.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

# Load in cgmodel
cgmodel = pickle.load(open( "stored_cgmodel.pkl", "rb" ))

# Create list of trajectory files for clustering analysis
number_replicas = 36
dcd_file_list_rep = []
for i in range(number_replicas):
    dcd_file_list_rep.append("output/replica_%s.dcd" %(i+1))

# Set clustering parameters
frame_start=100
frame_stride=100
frame_end=-1
    
# Run DBSCAN clustering
medoid_positions, cluster_size, cluster_rmsd, n_noise, silhouette_avg = get_cluster_medoid_positions_DBSCAN(
    file_list=dcd_file_list_rep,
    cgmodel=cgmodel,
    frame_start=frame_start,
    frame_stride=frame_stride,
    frame_end=-1,
    min_samples=10,
    eps=0.1,
    filter_ratio=0.10,
    output_dir=output_directory
)
    
print(f"Intra-cluster rmsd: {cluster_rmsd}")

k_min = np.argmin(cluster_rmsd)

# Minimize energy of native structure
native_positions, PE_start, PE_end, simulation = minimize_structure(
    cgmodel,
    medoid_positions[k_min],
    output_file=f"medoid_{k_min}_min.dcd",
)

native_structure_file=f"medoid_{k_min}_min.dcd"

# Set cutoff parameters:
# Cutoff for native structure pairwise distances:
native_contact_cutoff = 3.5* unit.angstrom

# Cutoff for current trajectory distances, as a multiple of native_contact_cutoff
native_contact_cutoff_ratio = 1.0

# Cutoff for native contact fraction folded vs. unfolded states:
Q_folded = 0.4

 # Determine native contacts:
native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
    cgmodel,
    native_structure_file,
    native_contact_cutoff
)

# Determine native contact fraction of current trajectories:
rep_traj = md.load(dcd_file_list_rep[0],top=md.Topology.from_openmm(cgmodel.topology))
nframes = rep_traj.n_frames

Q, Q_avg, Q_stderr, decorrelation_time = fraction_native_contacts(
    cgmodel,
    dcd_file_list_rep,
    native_contact_list,
    native_contact_distances,
    frame_begin=frame_start,
    native_contact_cutoff_ratio=native_contact_cutoff_ratio
)

num_intermediate_states = 1

results = expectations_fraction_contacts(
    Q,
    temperature_list,
    frame_begin=frame_start,
    sample_spacing=1,
    output_data=output_data,
    num_intermediate_states=num_intermediate_states,
)

plot_native_contact_fraction(
    results["T"],
    results["Q"],
    results["dQ"],
    plotfile=f"Q_expect_vs_T.pdf",
)

print("T (K), native contact fraction")
for t in range(len(results["T"])):
    print(f"{results['T'][t].value_in_unit(unit.kelvin)}   {results['Q'][t]}")    

# plot Q_avg vs. frame
plot_native_contact_timeseries(
    Q,
    frame_begin=frame_start,
    time_interval=10*unit.picosecond,
    plot_per_page=3,
    plotfile="Q_vs_time.pdf",
    figure_title="Native contact fraction",
)

# fit to hyperbolic switching function
param_opt, param_cov = fit_sigmoid(results["T"],results["Q"])

array_folded_states = np.zeros((len(Q[:,0]),number_replicas))

for rep in range(number_replicas):
    # Classify into folded/unfolded states:
    for frame in range(len(Q[:,rep])):
        if Q[frame,rep] >= Q_folded:
            # Folded
            array_folded_states[frame,rep] = 0
        else:
            # Unfolded
            array_folded_states[frame,rep] = 1

num_intermediate_states = 0  

full_T_list, deltaF_values, deltaF_uncertainty = expectations_free_energy(
    array_folded_states,
    temperature_list,
    frame_begin=frame_start,
    sample_spacing=1,
    output_data=output_data,
    num_intermediate_states=0,
)
    
plot_free_energy_results(
    full_T_list,
    deltaF_values,
    deltaF_uncertainty,
    plotfile="free_energy.pdf"
)        
    
F_unit = unit.kilojoule / unit.mole 
S_unit = F_unit / unit.kelvin
U_unit = F_unit
    
# Compute entropy and enthalpy
deltaS_values, deltaU_values = get_entropy_enthalpy(deltaF_values, full_T_list)    
    
deltaF_values = deltaF_values['state0_state1'].value_in_unit(F_unit)
deltaF_uncertainty = deltaF_uncertainty['state0_state1'].value_in_unit(F_unit)

deltaS_values = deltaS_values['state0_state1'].value_in_unit(S_unit)
deltaU_values = deltaU_values['state0_state1'].value_in_unit(U_unit)

# Get uncertainties from bootstrapping:
(full_T_list_boot, deltaF_values_boot, deltaF_uncertainty_boot, \
    deltaS_values_boot, deltaS_uncertainty_boot, \
    deltaU_values_boot, deltaU_uncertainty_boot) = bootstrap_free_energy_folding(
    array_folded_states,
    temperature_list,
    frame_begin=frame_start,
    sample_spacing=1,
    output_data=output_data,
    num_intermediate_states=0,
    n_sample_boot=200,
    n_trial_boot=10,
)

deltaF_values_boot = deltaF_values_boot['state0_state1'].value_in_unit(F_unit)
deltaF_uncertainty_boot = deltaF_uncertainty_boot['state0_state1'].value_in_unit(F_unit)

deltaS_values_boot = deltaS_values_boot['state0_state1'].value_in_unit(S_unit)
deltaS_uncertainty_boot = deltaS_uncertainty_boot['state0_state1'].value_in_unit(S_unit)

deltaU_values_boot = deltaU_values_boot['state0_state1'].value_in_unit(U_unit)
deltaU_uncertainty_boot = deltaU_uncertainty_boot['state0_state1'].value_in_unit(U_unit)

print(f"T (K), deltaF (kJ/mol), deltaF_boot, deltaF_uncertainty, deltaF_uncertainty_boot")
for i in range(len(full_T_list)):
    print(f"{full_T_list_boot[i].value_in_unit(unit.kelvin):>6.4f}, \
        {deltaF_values[i]:>6.8f}, \
        {deltaF_values_boot[i]:>6.8f}, \
        {deltaF_uncertainty[i]:>6.8f}, \
        {deltaF_uncertainty_boot[i]:>6.8f}")
        
print(f"\nT (K), deltaS (kJ/mol/K), deltaS_boot, deltaS_uncertainty_boot")
for i in range(len(full_T_list)):
    print(f"{full_T_list_boot[i].value_in_unit(unit.kelvin):>6.4f}, \
        {deltaS_values[i]:>6.8f}, \
        {deltaS_values_boot[i]:>6.8f}, \
        {deltaS_uncertainty_boot[i]:>6.8f}")
        
print(f"\nT (K), deltaU (kJ/mol), deltaU_boot, deltaU_uncertainty_boot")
for i in range(len(full_T_list)):
    print(f"{full_T_list_boot[i].value_in_unit(unit.kelvin):>6.4f}, \
        {deltaU_values[i]:>6.8f}, \
        {deltaU_values_boot[i]:>6.8f}, \
        {deltaU_uncertainty_boot[i]:>6.8f}")
      
