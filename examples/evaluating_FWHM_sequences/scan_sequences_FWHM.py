#!~/anaconda3/bin/python

import os
import pickle

from cg_openmm.parameters.evaluate_energy import *
from cg_openmm.parameters.reweight import get_temperature_list
from cg_openmm.thermo.calc import *
from simtk import unit

# This example demonstrates how to use the energy evaluation and MBAR reweighting
# framework to calculate the full-width half-maximum of a non-simulated monomer sequence,
# given a reference cgmodel and corresponding replica trajectories and energies.

# Set location of reference simulation output files:
output_directory = '../run_replica_exchange/output/'
output_data = os.path.join(output_directory, "output.nc")

# Load in reference trajectory stats:
analysis_stats = pickle.load(open("../run_replica_exchange/analysis_stats_discard_20ns.pkl","rb"))

# Load in reference CGModel
cgmodel = pickle.load(open("../run_replica_exchange/stored_cgmodel.pkl","rb"))

# Get temperature list (logarithmic spacing) corresponding to trajectory file list:
n_replicas = 12
min_temp = 200.0 * unit.kelvin
max_temp = 600.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, n_replicas)

# Create list of reference trajectory files to use for evaluating energies
rep_traj_file_list = []

for i in range(n_replicas):
    rep_traj_file_list.append(f'{output_directory}/replica_{i+1}.dcd')
    
# Specify A and B monomer types:
# ***Note: these must have the same topology as the monomer(s) defined in the reference cgmodel

mass = 100 * unit.amu

# Particle definitions:
bb1 = {
    "particle_type_name": "bb1",
    "sigma": 2.25 * unit.angstrom,
    "epsilon": 1.5 * unit.kilojoules_per_mole,
    "mass": mass
}
bb2 = {
    "particle_type_name": "bb2",
    "sigma": 2.25 * unit.angstrom,
    "epsilon": 1.5 * unit.kilojoules_per_mole,
    "mass": mass
}
sc1 = {
    "particle_type_name": "sc1",
    "sigma": 3.5 * unit.angstrom,
    "epsilon": 5.0 * unit.kilojoules_per_mole,
    "mass": mass
}
sc2 = {
    "particle_type_name": "sc2",
    "sigma": 3.5 * unit.angstrom,
    "epsilon": 6.0 * unit.kilojoules_per_mole,
    "mass": mass
}


# Monomer definitions:
A = {
    "monomer_name": "A",
    "particle_sequence": [bb1, sc1],
    "bond_list": [[0, 1]],
    "start": 0,
    "end": 0,
}

B = {
    "monomer_name": "B",
    "particle_sequence": [bb2, sc2],
    "bond_list": [[0, 1]],
    "start": 0,
    "end": 0,
}

# Set monomer list and sequences:

monomer_list = [A,B]
seq_list = []

# AB alternating copolymer:
seq_1 = []
for i in range(int(len(cgmodel.sequence)/2)):
    seq_1.append(A)
    seq_1.append(B)
    
# AB multiblock copolymer:
seq_2 = []
for i in range(int(len(cgmodel.sequence)/8)):
    seq_2.append(A)
    seq_2.append(A)
    seq_2.append(A)
    seq_2.append(A)
    
    seq_2.append(B)
    seq_2.append(B)
    seq_2.append(B)
    seq_2.append(B)
    
seq_list.append(seq_1)
seq_list.append(seq_2)    

# Set frames to analyze:
frame_begin = analysis_stats["production_start"]
sample_spacing = analysis_stats["energy_decorrelation"]
sparsify_stride = 2 # Evaluate energies of every other frame
frame_end = -1

# Set heat capacity calculation parameters:
n_trial_boot = 200          # Number of bootstraping trials
num_intermediate_states = 3 # Number of intemediate temperature states for MBAR calculation

(T_list, seq_Cv, seq_Cv_uncertainty,
seq_Tm, seq_Tm_uncertainty,
seq_Cv_height, seq_Cv_height_uncertainty,
seq_FWHM, seq_FWHM_uncertainty,
seq_N_eff) = eval_energy_sequences(
    cgmodel, rep_traj_file_list, temperature_list, monomer_list,
    output_data=output_data, sequence=seq_list, 
    frame_begin=frame_begin, frame_end=frame_end, sparsify_stride=sparsify_stride,
    sample_spacing=sample_spacing, num_intermediate_states=num_intermediate_states,
    n_trial_boot=n_trial_boot,
    verbose=True, n_cpu=12)
