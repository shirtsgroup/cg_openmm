## No default python environment

import os, sys, timeit
from io import StringIO
import numpy as np
import math, random
# OpenMM utilities
import simtk.openmm.app.element as elem
from simtk.openmm.app.pdbfile import PDBFile
from simtk import openmm as mm
from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm.vec3 import Vec3
# foldamers utilities
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.utilities.iotools import *
from cg_openmm.src.cg_mm_tools.cg_openmm import *

box_size = 10.00 * unit.nanometer # box width
cutoff = box_size / 2.0 * 0.99
simulation_time_step = 0.002 * unit.picosecond # Units = picoseconds
simulation_steps = 1000 # Number of steps used in individual Langevin dynamics simulations
print_frequency = 10 # Number of steps to skip when printing output
total_simulation_time = simulation_time_step * simulation_steps # Units = picoseconds

# Model settings
backbone_length = 1 # Number of backbone beads
sidechain_length = 1 # Number of sidechain beads
sidechain_positions = [0] # Index of backbone bead on which the side chains are placed
polymer_length = 12 # Number of monomers in the polymer
mass = 12.0 * unit.amu # Mass of beads
sigma = 8.4 * unit.angstrom # Lennard-Jones interaction distance
bond_length = 1.0 * unit.angstrom # bond length
bond_force_constant = 9.9e5
constrain_bonds = False
epsilon = 0.5 * unit.kilocalorie_per_mole # Lennard-Jones interaction strength
charge = 0.0 * unit.elementary_charge # Charge of beads

sigma = sigma.in_units_of(unit.nanometer)._value
charge = charge._value
epsilon = epsilon.in_units_of(unit.kilojoule_per_mole)._value
bond_length = bond_length.in_units_of(unit.nanometer)._value

def add_new_elements():
        elem.Element(117,'cg-backbone','CG1',mass)
        elem.Element(118,'cg-sidechain','CG2',mass)
        return

def build_system():
        # Create system
        system = mm.System()
        nonbonded_force = mm.NonbondedForce()
        bead_index = 0
        for monomer in range(polymer_length):
          for backbone_bead in range(backbone_length):
            system.addParticle(mass)
            nonbonded_force.addParticle(charge,sigma,epsilon)
            if monomer != 0:
             bead_index = bead_index + 1

             if backbone_bead == 0:
              force = mm.HarmonicBondForce()
              force.addBond(bead_index-sidechain_length-1, bead_index, bond_length,bond_force_constant)
              system.addForce(force)
              nonbonded_force.addException(bead_index-sidechain_length-1,bead_index,charge,bond_length,epsilon=0.0)

              if constrain_bonds:
               system.addConstraint(bead_index-sidechain_length-1, bead_index, bond_length)

             if backbone_bead != 0:
              force = mm.HarmonicBondForce()
              force.addBond(bead_index-1, bead_index, bond_length,bond_force_constant)
              system.addForce(force)
              nonbonded_force.addException(bead_index-1, bead_index,charge,bond_length,epsilon=0.0)

              if constrain_bonds:
               system.addConstraint(bead_index-1, bead_index, bond_length)

            if backbone_bead in sidechain_positions:
              for sidechain in range(sidechain_length):
                system.addParticle(mass)
                nonbonded_force.addParticle(charge,sigma,epsilon)
                bead_index = bead_index + 1

                force = mm.HarmonicBondForce()
                force.addBond(bead_index-1, bead_index, bond_length,bond_force_constant)
                system.addForce(force)
                nonbonded_force.addException(bead_index-1, bead_index,charge,bond_length,epsilon=0.0)

                if constrain_bonds:
                  system.addConstraint(bead_index,bead_index-1,bond_length)
  
        system.addForce(nonbonded_force)

        return(system)

def write_xml_file(filename):
 xml_object = open(filename,"w")
 xml_object.write("<ForceField>\n")
 xml_object.write(" <AtomTypes>\n")
 xml_object.write('  <Type name="0" class="C" element="CG1" mass="'+str(mass._value)+'"/>\n')
 xml_object.write('  <Type name="1" class="C" element="CG2" mass="'+str(mass._value)+'"/>\n')
 xml_object.write(" </AtomTypes>\n")
 xml_object.write(" <Residues>\n")
 xml_object.write('  <Residue name="M">\n')
 xml_object.write('   <Atom name="CG1" type="0"/>\n')
 xml_object.write('   <Atom name="CG2" type="1"/>\n')
 xml_object.write('   <Bond from="0" to="1"/>\n')
 xml_object.write('   <ExternalBond from="0"/>\n')
 xml_object.write('   <ExternalBond from="0"/>\n')
 xml_object.write("  </Residue>\n")
 xml_object.write('  <Residue name="MT">\n')
 xml_object.write('   <Atom name="CG1" type="0"/>\n')
 xml_object.write('   <Atom name="CG2" type="1"/>\n')
 xml_object.write('   <Bond from="0" to="1"/>\n')
 xml_object.write('   <ExternalBond from="0"/>\n')
 xml_object.write("  </Residue>\n")
 xml_object.write(" </Residues>\n")
 xml_object.write(" <HarmonicBondForce>\n")
 xml_object.write('  <Bond class1="CG" class2="CG" length="'+str(bond_length)+'" k="'+str(bond_force_constant)+'"/>\n')
 xml_object.write(" </HarmonicBondForce>\n")
 xml_object.write(' <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">\n')
 xml_object.write('  <Atom type="0" charge="'+str(charge)+'" sigma="'+str(sigma)+'" epsilon="'+str(epsilon)+'"/>\n')
 xml_object.write('  <Atom type="1" charge="'+str(charge)+'" sigma="'+str(sigma)+'" epsilon="'+str(epsilon)+'"/>\n')
 xml_object.write(" </NonbondedForce>\n")
 xml_object.write("</ForceField>\n")
 xml_object.close()

 return

add_new_elements()
cgmodel = CGModel()
#system = build_system()
pdb_file = "test.pdb"
write_pdbfile(cgmodel,pdb_file)
pdb_mm_obj = PDBFile(pdb_file)
topology = pdb_mm_obj.getTopology()
xml_file = "test.xml"
write_xml_file(xml_file)
forcefield = ForceField(xml_file)
system = forcefield.createSystem(topology,nonbondedMethod=PME,constraints=AllBonds,residueTemplates={'M','MT'})
test_pdb = "test_topology.pdb"
file = open(test_pdb,'w')
PDBFile.writeFile(topology,cgmodel.positions,file)
file.close()
simulation = build_mm_simulation(topology,system,cgmodel.positions)
simulation.step(10000)
positions = simulation.context.getState(getPositions=True).getPositions()
potential_energies = round(simulation.context.getState(getEnergy=True).getPotentialEnergy()._value,2)
#nonbonded_energies = "{:.2E}".format(calculate_nonbonded_energy(model_settings,particle_properties,positions)._value)
print(potential_energies)
#print(nonbonded_energies)
exit()
