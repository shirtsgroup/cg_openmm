# This 

import numpy as np
from simtk import openmm as mm
from simtk import unit
from include.get_random_coordinates import *
from simtk.openmm.app.pdbfile import PDBFile
from simtk.openmm.app import *
import simtk.openmm.app.element as elem
from simtk.openmm.app.forcefield import _createResidueTemplate,_matchResidue

def get_box_vectors(box_size):
 units = box_size.unit
 a = unit.Quantity(np.zeros([3]), units)
 a[0] = box_size
 b = unit.Quantity(np.zeros([3]), units)
 b[1] = box_size
 c = unit.Quantity(np.zeros([3]), units)
 c[2] = box_size
 return([a,b,c])

def assign_default_box_vectors(system,box_size):
 a,b,c = get_box_vectors(box_size)
 system.setDefaultPeriodicBoxVectors(a, b, c)
 return(system)

def get_mm_force(model_settings,particle_properties):
 box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings
 num_particles = (backbone_length + sidechain_length) * polymer_length
 mass,q,sigma,epsilon = particle_properties[:]
 force = mm.NonbondedForce()
 force.setCutoffDistance(1*unit.nanometer)
 bead_index = 0
 for particle in range(num_particles):
  force.addParticle(q, sigma, epsilon)
 for monomer in range(polymer_length):
  for backbone_bead in range(backbone_length):
   if bead_index != 0:
    bead_index = bead_index + 1
    force.addException(particle1=bead_index,particle2=bead_index-sidechain_length-1,sigma=sigma*0.1,epsilon=0.0,chargeProd=0.0)
   if backbone_bead in sidechain_positions:
    for sidechain in range(sidechain_length):
     bead_index = bead_index + 1
     force.addException(particle1=bead_index,particle2=bead_index-1,sigma=sigma*0.1,epsilon=0.0,chargeProd=0.0)
 return(force)

def build_cg_topology(model_settings,particle_properties):
        # Create topology.
        box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings
        topology = Topology()
        backbone_elem = elem.Element(117,'cgbackbone','X',10.0)
        sidechain_elem = elem.Element(118,'cgsidechain','Q',10.0)
#        element(117,'cgbackbone','X',10.0)
#        element(118,'cgsidechain','Q',10.0)
#        backbone_elem = element.get_by_symbol('X')
#        sidechain_elem = element.get_by_symbol('Q')
        chain = topology.addChain()
        for monomer in range(polymer_length):
            residue = topology.addResidue('CG', chain)
            topology.addAtom('X', backbone_elem, residue)
            topology.addAtom('Q', sidechain_elem, residue)
        return(topology)

def distance(positions_1,positions_2):
 direction_comp = [0.0 for i in range(0,3)]
 for direction in range(len(direction_comp)):
  direction_comp[direction] = positions_1[direction].__sub__(positions_2[direction])
 direction_comb = [0.0 for i in range(0,3)]
 for direction in range(len(direction_comb)):
  direction_comb[direction] = direction_comp[direction].__pow__(2.0)
 sqrt_arg = direction_comb[0].__add__(direction_comb[1]).__add__(direction_comb[2])
 value = math.sqrt(sqrt_arg._value)
 units = sqrt_arg.unit.sqrt()
 distance = unit.Quantity(value=value,unit=units)
 return(distance)

def lj_v(positions_1,positions_2,sigma,epsilon):
 dist = distance(positions_1,positions_2)
 attr = dist.__div__(sigma).__pow__(6.0)
 rep = dist.__div__(sigma).__pow__(12.0)
 v = 4.0 * epsilon.__mul__(rep.__sub__(attr))
 return(v)

def calculate_nonbonded_energy(model_settings,particle_properties,positions):
 box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings
 mass,q,sigma,epsilon = particle_properties
 energy = unit.Quantity(0.0,unit.kilojoules_per_mole)
 bead_index = 0
 for monomer in range(polymer_length):
  for backbone_bead in range(backbone_length):
   if bead_index != 0:
    bead_index = bead_index + 1
    energy = energy.__add__(lj_v(positions[bead_index],positions[bead_index-sidechain_length-1],sigma,epsilon))
   if backbone_bead in sidechain_positions:
    for sidechain in range(sidechain_length):
     bead_index = bead_index + 1
     energy = energy.__add__(lj_v(positions[bead_index],positions[bead_index-sidechain_length-1],sigma,epsilon))
 return(energy)

def build_cg_system(model_settings,particle_properties):
 box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings
 num_particles = (backbone_length + sidechain_length) * polymer_length
 mass,q,sigma,epsilon = particle_properties[:]
 system = mm.System()
 bead_index = 0
 for monomer in range(polymer_length):
  for backbone_bead in range(backbone_length):
   system.addParticle(mass)
   if bead_index != 0:
    bead_index = bead_index + 1
    system.addConstraint(bead_index,bead_index-sidechain_length-1,sigma*0.1)
   if backbone_bead in sidechain_positions:
    for sidechain in range(sidechain_length):
     system.addParticle(mass)
     bead_index = bead_index + 1
     system.addConstraint(bead_index,bead_index-1,sigma*0.1)
 return(system)

def build_cg_model(model_settings,particle_properties,positions):
 box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings
 num_particles = (backbone_length + sidechain_length) * polymer_length
 mass,q,sigma,epsilon = particle_properties[:]
# Record the positions
 pdb_file = 'temp.pdb'
# write_positions_to_pdbfile(positions,pdb_file,model_settings)
# Build a topology
 topology = build_cg_topology(model_settings,particle_properties)
#
 nonbonded = get_mm_force(model_settings,particle_properties)
 system = build_cg_system(model_settings,particle_properties)
 system.addForce(nonbonded) 
# system = assign_default_box_vectors(system,box_size)
 return(system,topology)
