import os
from simtk.openmm.app.forcefield import ForceField
import simtk.openmm.app.element as elem
from simtk.openmm.app.element import Element
from simtk.openmm.app import PDBFile
from simtk import unit
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.utilities.iotools import write_pdbfile_without_topology
from foldamers.src.utilities.util import random_positions
from cg_openmm.src.build.cg_build import write_xml_file, build_system, build_topology
from cg_openmm.src.simulation.tools import run_simulation, build_mm_simulation, minimize_structure
from cg_openmm.src.utilities.iotools import read_mm_energies
from cg_openmm.src.simulation.rep_exch import replica_exchange
temperature = 300.0 * unit.kelvin
print_frequency = 1 # Number of steps to skip when printing output
total_simulation_time = 0.1 * unit.picosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond

cg_model = CGModel(backbone_lengths=[3],sidechain_lengths=[3],include_torsion_forces=False)
#cg_model.positions = random_positions(cg_model)
#print(cg_model.positions)
write_pdbfile_without_topology(cg_model,"test.pdb")
#exit()
#pdb = PDBFile("test.pdb")
#xml_file = "test.xml"
#top = pdb.topology
#particle_list = write_xml_file(cg_model,xml_file)
#ff = ForceField(xml_file)
#element_list = elem.Element._elements_by_symbol
#for residue in top.residues():
  #unique_bonds = []
  #print([bond for bond in residue.bonds()])
  #for bond in residue.bonds():
  #  if bond not in unique_bonds:
  #    unique_bonds.append(bond)
#  for atom in residue.atoms():
#    if atom.element == None:
#      try:
#        atom.element = element_list[atom.name]
#      except:
#        print("No element found.")
#templates,residues = ff.generateTemplatesForUnmatchedResidues(top)
#cg_model.topology = top
#cg_model.system = ff.createSystem(top,residueTemplates=templates)
#print([cg_model.system.getForce(i) for i in range(cg_model.system.getNumForces())])
#exit()
#cg_model.system.setDefaultPeriodicBoxVectors([10.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0])
#run_simulation(cg_model,os.getcwd(),total_simulation_time,simulation_time_step,temperature,print_frequency)
#simulation = build_mm_simulation(cg_model.topology,cg_model.system,cg_model.positions,simulation_time_step=simulation_time_step,print_frequency=1)
#positions = simulation.context.getState(getPositions=True).getPositions()
#potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
#forces = simulation.context.getState(getForces=True).getForces()
#print(forces)
#parameters = simulation.context.getState(getParameters=True).getParameters()
#parameter_derivatives = simulation.context.getState(getParameterDerivatives=True).getEnergyParameterDerivatives()
#print(parameter_derivatives)
#exit()
#velocities = simulation.context.getState(getVelocities=True).getVelocities()
#print(positions)
#print(potential_energy)

#cg_model.system = build_system(cg_model)
#cg_model.topology = build_topology(cg_model)
simulation = build_mm_simulation(cg_model.topology,cg_model.system,cg_model.positions,simulation_time_step=simulation_time_step,print_frequency=1)
positions,energy = minimize_structure(cg_model.topology,cg_model.system,cg_model.positions,temperature=300.0 * unit.kelvin,simulation_time_step=5.0 * unit.femtosecond,total_simulation_time=0.1 * unit.picosecond,output_pdb='minimum.pdb',output_data='minimization.dat',print_frequency=1)
cg_model.positions = positions
write_pdbfile_without_topology(cg_model,'minimum.pdb')
replica_energies,reduced_replica_energies,replica_positions,temperature_list = replica_exchange(cg_model.topology,cg_model.system,cg_model.positions,simulation_time_step=5.0*unit.femtosecond,test_time_step=False)
print(replica_energies)
#print([parameter for parameter in parameters])
#print([derivative for derivative in parameter_derivatives])
#print(velocities)
#simulation.step(1)
#simulation_data = read_mm_energies(str('simulation.dat'))
#print(simulation_data)
#print("Success")
exit()
