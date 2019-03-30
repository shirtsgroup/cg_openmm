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
 box_size = box_size * unit.angstrom
 a = unit.Quantity(np.zeros([3]), unit.angstroms)
 a[0] = box_size
 b = unit.Quantity(np.zeros([3]), unit.angstroms)
 b[1] = box_size
 c = unit.Quantity(np.zeros([3]), unit.angstroms)
 c[2] = box_size
 return([a,b,c])

def assign_default_box_vectors(system,box_size):
 box_size = box_size * unit.angstrom
 a = unit.Quantity(np.zeros([3]), unit.angstroms)
 a[0] = box_size
 b = unit.Quantity(np.zeros([3]), unit.angstroms)
 b[1] = box_size
 c = unit.Quantity(np.zeros([3]), unit.angstroms)
 c[2] = box_size
 system.setDefaultPeriodicBoxVectors(a, b, c)
 return(system)

def write_xml_file(filename,model_settings,particle_properties):
 box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings
 mass,q,sigma,epsilon = particle_properties
 xml_object = open(filename,"w")
 xml_object.write("<ForceField>\n")
 xml_object.write(" <AtomTypes>\n")
 xml_object.write('  <Type name="1" class="CG" element="X" mass="'+str(mass)+'"/>\n')
 xml_object.write('  <Type name="2" class="CG" element="Q" mass="'+str(mass)+'"/>\n')
 xml_object.write(" </AtomTypes>\n")
 xml_object.write(" <Residues>\n")
 xml_object.write('  <Residue name="CG1">\n')
 xml_object.write('   <Atom name="X" type="1"/>\n')
 xml_object.write('   <Atom name="Q" type="1"/>\n') 
 xml_object.write('   <Bond from="0" to="1"/>\n')
 xml_object.write('   <ExternalBond from="0"/>\n')
 xml_object.write("  </Residue>\n")
 xml_object.write('  <Residue name="CG2">\n')
 xml_object.write('   <Atom name="X" type="1"/>\n')
 xml_object.write('   <Atom name="Q" type="1"/>\n')
 xml_object.write('   <Bond from="0" to="1"/>\n')
 xml_object.write('   <ExternalBond from="0"/>\n')
 xml_object.write('   <ExternalBond from="0"/>\n')
 xml_object.write("  </Residue>\n")
 xml_object.write(" </Residues>\n")
 xml_object.write(" <HarmonicBondForce>\n")
 int_distance = str(sigma)
 xml_object.write('  <Bond class1="CG" class2="CG" length="'+str(int_distance)+'" k="'+str(9e20)+'"/>\n')
 xml_object.write(" </HarmonicBondForce>\n")
 xml_object.write(' <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">\n')
 charge = str(q).replace(' e','')
 int_strength = str(epsilon).replace(' kcal/mol','')
 xml_object.write('  <Atom type="0" charge="'+str(charge)+'" sigma="'+str(int_distance)+'" epsilon="'+str(int_strength)+'"/>\n')
 xml_object.write(" </NonbondedForce>\n")
 xml_object.write("</ForceField>\n")
 xml_object.close()
 return

def add_bonds_to_topology(topology,model_settings):
 box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings
 element = "C"
 chain = Chain(1,topology,id="A")
# topology.addChain(Chain)
 bead_index = 1
 atom_list = []
 residue_name = "CG"
# Add bonds
 for monomer_index in range(0,polymer_length):
  residue = Residue(residue_name,monomer_index,chain,id=str("R"+str(monomer_index)),insertionCode='')
  for backbone_bead in range(0,backbone_length):
   atom_name = "X"
   atom = Atom(atom_name,element,bead_index-1,residue,id=atom_name)
   bead_index = bead_index + 1
   atom_list.append(atom)
   if backbone_bead == 0 and bead_index != 2:
    topology.addBond(atom_list[bead_index-sidechain_length-3],atom_list[bead_index-2])
   if backbone_bead != 0 and bead_index == 2:
    topology.addBond(atom_list[bead_index-2],atom_list[bead_index-1])
   if backbone_bead in sidechain_positions:
    for sidechain in range(0,sidechain_length):
     atom_name = "Q"
     atom = Atom(atom_name,element,bead_index-1,residue,id=atom_name)
     atom_list.append(atom)
     bead_index = bead_index + 1
     topology.addBond(atom_list[bead_index-3],atom_list[bead_index-2])
 return(topology) 

def build_cg_forcefield(model_settings,particle_properties):
        # Build the ForceField programmatically.
        box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings
        mass,q,sigma,epsilon = particle_properties
        ff = ForceField()
        ff.registerAtomType({'name':'X', 'class':'1', 'mass':mass*unit.daltons, 'element':elem.cgbackbone})
        ff.registerAtomType({'name':'Q', 'class':'2', 'mass':mass*unit.daltons, 'element':elem.cgsidechain})
        nonbonded = forcefield.NonbondedGenerator(ff, 0.8333, 0.5)
        charge = str(q).replace(' e','')
        int_strength = str(epsilon).replace(' kcal/mol','')
        nonbonded.registerAtom({'type':'X', 'charge':charge, 'sigma':0.1*sigma*unit.nanometers, 'epsilon':int_strength*unit.kilojoules_per_mole})
        nonbonded.registerAtom({'type':'Q', 'charge':charge, 'sigma':0.1*sigma*unit.nanometers, 'epsilon':int_strength*unit.kilojoules_per_mole})
        ff.registerGenerator(nonbonded)
        return(ff)

def build_cg_topology(model_settings):
        # Create topology.
        box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings
        topology = Topology()
        backbone_elem = elem.get_by_symbol('X')
        sidechain_elem = elem.get_by_symbol('Q')
        chain = topology.addChain()
        for monomer in range(polymer_length):
            residue = topology.addResidue('CG', chain)
            topology.addAtom('X', backbone_elem, residue)
            topology.addAtom('Q', sidechain_elem, residue)
        return(topology)

def get_nonbonded_forcefield(model_settings,particle_properties):
 box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings
 mass,q,sigma,epsilon = particle_properties[:]
 num_particles = (backbone_length + sidechain_length) * polymer_length
 ff = ForceField()
 nonbonded = forcefield.NonbondedGenerator(ff, 0.8333, 0.5)
 int_strength = str(epsilon).replace(' kcal/mol','')
 charge = str(q).replace(' e','')
 nonbonded.registerAtom({'type':'X', 'charge':charge, 'sigma':sigma*unit.angstroms, 'epsilon':int_strength*unit.kilocalories_per_mole})
 nonbonded.registerAtom({'type':'Q', 'charge':charge, 'sigma':sigma*unit.angstroms, 'epsilon':int_strength*unit.kilocalories_per_mole})
 for monomer in range(polymer_length):
  for backbone_bead in range(backbone_length):
   if bead_index != 0:
    bead_index = bead_index + 1
    nonbonded.addException(bead_index,bead_index-sidechain_length-1,chargeProd=0.0,sigma=sigma*0.1,epsilon=0.0)
   if backbone_bead in sidechain_positions:
    for sidechain in range(sidechain_length):
     bead_index = bead_index + 1
     nonbonded.addException(bead_index,bead_index-1,chargeProd=0.0,sigma=sigma*0.1,epsilon=0.0)
 print(nonbonded.getNumExceptions())
 ff.registerGenerator(nonbonded)
 return(ff)

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
 print(force.getNumExceptions())
 return(force)

def build_cg_model(model_settings,particle_properties,positions):
 box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings
 num_particles = (backbone_length + sidechain_length) * polymer_length
 mass,q,sigma,epsilon = particle_properties[:]
# xml_file = 'temp.xml'
# write_xml_file(xml_file,model_settings,particle_properties)
 pdb_file = 'temp.pdb'
 write_positions_to_pdbfile(positions,pdb_file,model_settings)

 topology = build_cg_topology(model_settings)
# forcefield = get_nonbonded_forcefield(model_settings,particle_properties)

 force = get_mm_force(model_settings,particle_properties)
 system = build_cg_system(model_settings,particle_properties)
 system.addForce(force) 
# system = assign_default_box_vectors(system,box_size)
 return(system,topology)
