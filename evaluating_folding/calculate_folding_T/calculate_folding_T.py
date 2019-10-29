import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
import mdtraj as md
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import *
from foldamers.ensembles.ens_build import *
from foldamers.parameters.secondary_structure import *
from cg_openmm.simulation.tools import *

# Define Boltzmann's constant
kB = unit.Quantity(0.0019872041,unit.kilocalorie_per_mole)

total_simulation_time = 100.0 * unit.picosecond
simulation_time_step = 5.0 * unit.femtosecond
print_frequency = 5
number_temperatures = 50
min_temp = 1.0 * unit.kelvin
max_temp = 500.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_temperatures)
output = 'output'
if not os.path.exists(output):
  os.mkdir(output)

# Model settings
polymer_length=12
backbone_lengths=[1]
sidechain_lengths=[1]
sidechain_positions=[0]
include_bond_forces=False
include_bond_angle_forces=False
include_nonbonded_forces=True
include_torsion_forces=True
constrain_bonds = True

# Bond definitions
bond_length = 7.5 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
bond_force_constant = 0 * unit.kilocalorie_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {'bb_bb_bond_k': bond_force_constant, 'bb_sc_bond_k': bond_force_constant, 'sc_sc_bond_k': bond_force_constant}

# Particle definitions
mass = 100.0 * unit.amu
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass}
# Set the value of epsilon equal to kT at the mean temperature
epsilon = 0.005 * unit.kilocalorie_per_mole
epsilons = {'bb_eps': epsilon,'sc_eps': epsilon}
r_min =  3.0 * bond_length # Lennard-Jones potential r_min
sigma = r_min/(2.0**(1/6))
sigmas = {'bb_sigma': sigma,'sc_sigma': sigma}

# Torsion angle definitions
torsion_force_constant = 0.0001 * unit.kilocalorie_per_mole / unit.radian / unit.radian
torsion_force_constants = {'bb_bb_bb_bb_torsion_k': torsion_force_constant}#,'sc_bb_bb_sc_torsion_k': torsion_force_constant}
bb_bb_bb_bb_equil_torsion_angle = 78.0 * (3.14/180.0) # OpenMM defaults to units of radians for angle definitions
#sc_bb_bb_sc_equil_torsion_angle = 110.0 * (3.14/180.0)
equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': bb_bb_bb_bb_equil_torsion_angle}#,'sc_bb_bb_sc_torsion_0': sc_bb_bb_sc_equil_torsion_angle}
torsion_periodicities = {'bb_bb_bb_bb_period': 1}#,'sc_bb_bb_sc_period': 2}

# Initiate cgmodel using positions from local file
positions = PDBFile("init.pdb").getPositions()

native_structure = positions

cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,torsion_force_constants=torsion_force_constants,equil_torsion_angles=equil_torsion_angles,torsion_periodicities=torsion_periodicities,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds,positions=positions)

# Set parameters for definition/evaluation of native contacts
native_structure_contact_distance_cutoff = 1.05 * cgmodel.get_sigma(0) # This distance cutoff determines which nonbonded interactions are considered 'native' contacts
native_contact_cutoff_ratio = 1.1 # The distance ratio (in comparison with the distance of a contact in the native structure) below which a nonbonded interaction is considered 'native'

min_Q_list = []
for temperature in temperature_list:
 print(str("Running a simulation with temperature = "+str(temperature._value)))
 output_directory = str(str(output)+'/'+str(round(temperature._value,1)))
 if not os.path.exists(output_directory):
  os.mkdir(output_directory)

 run_simulation(cgmodel,output_directory,total_simulation_time,simulation_time_step,temperature,print_frequency)

 data = read_simulation_data(str(str(output_directory)+"/simulation.dat"),simulation_time_step)
 trajectory = md.load(str(str(output_directory)+"/simulation.pdb")).xyz

 Q_list = []
 for pose in trajectory[100:]:
  pose = unit.Quantity(pose,unit.nanometer)
  Q = fraction_native_contacts(cgmodel,pose,native_structure,native_structure_contact_distance_cutoff=native_structure_contact_distance_cutoff,native_contact_cutoff_ratio=native_contact_cutoff_ratio)
  Q_list.append(Q)

 min_Q_list.append(min(Q_list))

x = np.array([float(temperature._value) for temperature in temperature_list])
y = np.array([float(Q) for Q in min_Q_list])
figure = pyplot.figure(1)
pyplot.xlabel("Temperature (Kelvin)")
pyplot.ylabel("Minimum Q")
pyplot.plot(x,y)
pyplot.savefig("Min_Q_v_T.png")
exit()
