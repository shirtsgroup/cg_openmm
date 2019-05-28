#!/usr/bin/python

import csv, statistics
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
# foldamers utilities
from multiprocessing import Pool
from multiprocessing import cpu_count
import matplotlib.pyplot as pyplot
import foldamers
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.utilities.util import *
from foldamers.src.utilities.iotools import *
from cg_openmm.src.cg_mm_tools.cg_openmm import *

# Use parallel processing
processors = cpu_count() - 4
print("Found "+str(cpu_count)+" processors on the system.")
print("Using "+str(processors)+" processors for this calculation.")

# OpenMM simulation settings
total_simulations = 1
box_size = 10.00 * unit.nanometer # box width
cutoff = box_size / 2.0 * 0.99
simulation_time_step = 0.01 * unit.femtosecond # Units = picoseconds
temperature = 300.0 * unit.kelvin
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 0.5 * unit.picosecond # Units = picoseconds
total_steps = 500
#total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Coarse grained model settings

####
#
# User note: If multiple 'backbone_lengths', 'sidechain_lengths',
#
#            'sidechain_positions', or 'sidechain_branches' are
#
#            provided as input, and optimize_model_parameters = False,
#
#            then the code will apply the same model parameters as 
#
#            input for all possible combinations of these input. 
#
#            If optimize_model_parameters = True, the code will optimize
#
#            parameters for all possible combinations of these input.
#
####

mass = unit.Quantity(1.0,unit.amu)
sigma = unit.Quantity(2.4,unit.angstrom)
bond_length = unit.Quantity(1.0,unit.angstrom)
epsilon = unit.Quantity(0.5,unit.kilocalorie_per_mole)
bond_force_constant = 9.9e5
charge = unit.Quantity(0.0,unit.elementary_charge)
torsion_force_constant = 200
equil_dihedral_angle = 180
bond_angle_force_constant = 200

increments = 10

backbone_lengths = [1] # Number of backbone beads in unique monomer types
# List of backbone_lengths for which to construct unique monomer topology definitions
# List( [ integers ( Number of backbone beads for each monomer type ) ] )
sidechain_lengths = [1] # Number of sidechain beads in unique monomer types
# List of sidechain_lengths for which to construct unique monomer topology definitions
# List( [ integers ( Number of sidechain beads for each monomer type ) ] )
constrain_bonds = False # Constrain bonds for these particle types?
# List( [ Logical ( Constrain bonds btwn backbone beads? ), ( Constrain bonds btwn backbone and sidechain beads? ), ( Constrain bonds btwn sidechain beads? ) ] )
sidechain_positions = [0] # Index of the backbone bead(s) to which sidechains are bonded
sidechain_branches = [1] # Index of the sidechain bead off of which to branch (bond) another sidechain


polymer_length = 8 # Number of monomers in the polymer
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass} # List of bead masses 
sigmas = {'bb_bb_sigma': sigma,'bb_sc_sigma': sigma,'sc_sc_sigma': sigma} # Lennard-Jones interaction distances.  List of unique interaction types
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length} # bond length
bond_force_constants = {'bb_bb_bond_k': bond_force_constant,'bb_sc_bond_k': bond_force_constant, 'sc_sc_bond_k': bond_force_constant} # Units = kJ/mol/A^2 List of bond force constants for unique bond types
#constrain_bonds = True
epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': epsilon} # Lennard-Jones interaction strength.  List of unique interaction types
charges = {'backbone_bead_charges': charge,'sidechain_bead_charges': charge} # Charge of beads.
torsion_force_constants = {'bb_bb_bb_bb_torsion_k': torsion_force_constant,'bb_bb_bb_sc_torsion_k': torsion_force_constant,'bb_bb_sc_sc_torsion_k': torsion_force_constant, 'bb_sc_sc_sc_torsion_k': torsion_force_constant, 'sc_bb_bb_sc_torsion_k': torsion_force_constant, 'bb_sc_sc_bb_torsion_k': torsion_force_constant, 'sc_sc_sc_sc_torsion_k': torsion_force_constant} # List of torsion force constants (k) for each of the torsions in our coarse grained model
bond_angle_force_constants = {'bb_bb_bb_angle_k': bond_angle_force_constant,'bb_bb_sc_angle_k': bond_angle_force_constant,'bb_sc_sc_angle_k': bond_angle_force_constant,'sc_sc_sc_angle_k': bond_angle_force_constant} # List of bond angle force constants (k) for each of the bond angles in our coarse grained model
optimize_model_parameters = True # If true, we will attempt to (self-consistently) optimize all parameters
# in our coarse grained model, and determine a valid simulation time step for a model with these parameters.
# This option can also be used to determine/evaluate parameter ratios using the ratios = True option.
optimize_parameter_ratios = True # If true, we will attempt to (self-consistently) optimize the ratios for
# related parameters in our coarse grained model
max_force = 1e6 # The maximum force ( in units of kJ/mol/A^2 ) that is accepted when determining the suitability
# of a simulation time step
homopolymer = True


if len(sigmas) != 0: include_nonbonded_forces = True
if len(bond_force_constants) != 0: include_bond_forces = True
include_bond_forces = False
if len(bond_angle_force_constants) != 0: include_bond_angle_forces = True
if len(torsion_force_constants) != 0: include_torsion_forces = True
include_bond_angle_forces = False
if max_force != None: check_energy_conservation = True
parameter_combo_list = []
sigma_list = [unit.Quantity(sigma,unit.angstrom) for sigma in [ 1.5 + 0.5 * index for index in range(0,increments)]]
epsilon_list = [unit.Quantity(epsilon,unit.kilocalorie_per_mole) for epsilon in [ 0.2 + 0.2 * index for index in range(0,increments)]]
variance_list = []
for sigma in sigma_list:
 for epsilon in epsilon_list:
  for simulation_index in range(total_simulations):
   print("Performing simulation with sigma="+str(sigma)+" and epsilon="+str(epsilon))
   sigmas = {'bb_bb_sigma': sigma,'bb_sc_sigma': sigma,'sc_sc_sigma': sigma} # Lennard-Jones interaction distances.  List of unique interaction types
   epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': epsilon} # Lennard-Jones interaction strength.  List of unique interaction types
   input_coordinates = str("init_"+str(simulation_index)+".pdb") # Read coordinates and build a coarse-grained model
   pdb_mm_obj = PDBFile(input_coordinates)
   positions = pdb_mm_obj.getPositions()
   cgmodel = CGModel(positions=positions,polymer_length=polymer_length,backbone_lengths=backbone_lengths, sidechain_lengths=sidechain_lengths, sidechain_positions = sidechain_positions, masses = masses, sigmas = sigmas, epsilons = epsilons, bond_lengths = bond_lengths, bond_force_constants = bond_force_constants, torsion_force_constants=torsion_force_constants, equil_dihedral_angle=equil_dihedral_angle,bond_angle_force_constants=bond_angle_force_constants, charges = charges,constrain_bonds=constrain_bonds,include_bond_forces=include_bond_forces,include_nonbonded_forces=include_nonbonded_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,check_energy_conservation=False)

   output_pdb = str("simulation_"+str(simulation_index)+".pdb")
   output_data = str("simulation_"+str(simulation_index)+".dat")
# Build an OpenMM simulation object
   simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature=temperature,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,output_pdb=output_pdb,output_data=output_data,print_frequency=print_frequency)

#print(total_steps)
   simulation.step(total_steps)
# print("Finished simulation.")
   all_energies = []
   with open(output_data) as csvfile:
    readCSV = csv.reader(csvfile,delimiter=',')
    next(readCSV)
    for row in readCSV:
     all_energies.append(float(row[3]))
   variance = statistics.variance(all_energies)
   variance_list.append(variance)
   parameter_combo_list.append({'sigma': sigma, 'epsilon': epsilon, 'variance': variance})

lowest_variance = None
best_combo = None
for parameter_combo in parameter_combo_list:
 if lowest_variance == None: 
  best_combo = parameter_combo
  lowest_variance = parameter_combo['variance']
 else:
  if parameter_combo['variance'] < lowest_variance:
   lowest_variance = parameter_combo['variance']
   best_combo = parameter_combo

sigma=np.unique([float(sig._value) for sig in sigma_list])
epsilon=np.unique([float(eps._value) for eps in epsilon_list])
variance=np.array(variance_list)
#for sigma_index in range(len(sigma_list)):
# for epsilon_index in range(len(epsilon_list)):
#  for parameter_combo in parameter_combo_list:
#   if parameter_combo['sigma'] == sigma_list[sigma_index] and parameter_combo['epsilon'] == epsilon_list[epsilon_index]:
#    variance[sigma_index][epsilon_index] = parameter_combo['variance']
X,Y = np.meshgrid(sigma,epsilon)
Z=variance.reshape(len(epsilon),len(sigma))
file = open('sig.dat','w')
file.write(str(X))
file.close()
file = open('eps.dat','w')
file.write(str(Y))
file.close()
file = open('var.dat','w')
file.write(str(Z))
file.close()
figure = pyplot.figure(0)
pyplot.xlabel("Sigma (Angstroms)")
pyplot.ylabel("Epsilon (kcal/mol)")
pyplot.title("Variance in energy of a (1,1) 8-mer")
pyplot.pcolormesh(X,Y,Z)
pyplot.savefig(str("energy_variance_vs_sig_eps.png"))
pyplot.close()

# Print the results for the best parameter set
print("The best combination of parameters is:")
print(best_combo)
exit()
