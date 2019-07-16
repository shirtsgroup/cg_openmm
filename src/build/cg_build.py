import numpy as np
from simtk import openmm as mm
from simtk.openmm.openmm import LangevinIntegrator
from simtk import unit
from simtk.openmm.app.pdbreporter import PDBReporter
from simtk.openmm.app.statedatareporter import StateDataReporter
from simtk.openmm.app.simulation import Simulation
from simtk.openmm.app.topology import Topology
import simtk.openmm.app.element as elem
from foldamers.src.cg_model import cgmodel
from cg_openmm.src.simulation.tools import get_simulation_time_step

def add_new_elements(cgmodel,list_of_masses):
        """
        Adds new coarse grained particle types to OpenMM

        Parameters
        ----------

        cgmodel: CGModel() class object

        list_of_masses: List of masses for the particles we want to add to OpenMM

        """
        element_index = 117
        mass_index = 0
        cg_particle_index = 1
        particle_list = []
        for monomer_type in cgmodel.monomer_types:
         for backbone_bead in range(monomer_type['backbone_length']):
          particle_name = str("bb-"+str(cg_particle_index))
          particle_symbol = str("B"+str(cg_particle_index))
          if particle_symbol not in elem.Element._elements_by_symbol:
           elem.Element(element_index,particle_name,particle_symbol,list_of_masses[mass_index])
           particle_list.append(particle_symbol)
           element_index = element_index + 1
           cg_particle_index = cg_particle_index + 1
           mass_index = mass_index + 1
          if type(monomer_type['sidechain_positions']) == int:
           sidechain_positions = [monomer_type['sidechain_positions']]
          else:
           sidechain_positions = monomer_type['sidechain_positions']
          if backbone_bead in sidechain_positions:
           for sidechain in range(monomer_type['sidechain_length']):
            if particle_symbol not in elem.Element._elements_by_symbol:
             particle_name = str("sc-"+str(cg_particle_index))
             particle_symbol = str("S"+str(cg_particle_index))
             elem.Element(element_index,particle_name,particle_symbol,list_of_masses[mass_index])
             particle_list.append(particle_symbol)
             element_index = element_index + 1
             cg_particle_index = cg_particle_index + 1
             mass_index = mass_index + 1
        return(particle_list)

def build_mm_force(sigma,epsilon,charge,num_beads,cutoff=1*unit.nanometer):
        """

        Build an OpenMM 'Force' for the non-bonded interactions in our model.

        Parameters
        ----------

        sigma: Non-bonded bead Lennard-Jones interaction distances,
        ( float * simtk.unit.distance )

        epsilon: Non-bonded bead Lennard-Jones interaction strength,
        ( float * simtk.unit.energy )

        charge: Charge for all beads
        ( float * simtk.unit.charge ) 

        cutoff: Cutoff distance for nonbonded interactions
        ( float * simtk.unit.distance )

        num_beads: Total number of beads in our coarse grained model
        ( integer )

        """

        force = mm.NonbondedForce()
        
        force.setCutoffDistance(cutoff)

        for particle in range(num_particles):
          force.addParticle( charge, sigma, epsilon )
        return(force)

def verify_topology(cgmodel):
        """

        Verify the OpenMM topology for our coarse grained model

        Parameters
        ----------
        """
        if cgmodel.num_beads != cgmodel.topology.getNumAtoms():
          print("ERROR: The number of particles in the coarse grained model\n")
          print("does not match the number of particles in the OpenMM topology.\n")
          print("There are "+str(cgmodel.num_beads)+" particles in the coarse grained model\n")
          print("and "+str(cgmodel.topology.getNumAtoms())+" particles in the OpenMM topology.")
          exit()

        if cgmodel.polymer_length != cgmodel.topology.getNumResidues():
          print("ERROR: The number of monomers in the coarse grained model\n")
          print("does not match the number of residues in the OpenMM topology.\n")
          print("There are "+str(cgmodel.polymer_length)+" monomers in the coarse grained model\n")
          print("and "+str(cgmodel.topology.getNumResidues())+" monomers in the OpenMM topology.")
          exit()

        return


def build_topology(cgmodel):
        """

        Construct an OpenMM topology for our coarse grained model

        Parameters
        ----------

        polymer_length: Number of monomers in our coarse grained model
        ( integer )

        backbone_length: Number of backbone beads on individual monomers
        in our coarse grained model, ( integer )

        sidechain_length: Number of sidechain beads on individual monomers
        in our coarse grained model, ( integer )

        """

        topology = Topology()

        chain = topology.addChain()
        residue_index = 1
        cg_particle_index = 1
        for monomer_type in cgmodel.sequence:
         residue = topology.addResidue(str(residue_index), chain)
         for backbone_bead in range(monomer_type['backbone_length']):
          particle_name = str("bb-"+str(cg_particle_index))
          particle_symbol = str("B"+str(cg_particle_index))
          particle = topology.addAtom(particle_symbol, particle_name, residue)
          if backbone_bead == 0 and residue_index != 1:
           topology.addBond(particle,last_backbone_particle)
          last_backbone_particle = particle
          cg_particle_index = cg_particle_index + 1
          if backbone_bead in [monomer_type['sidechain_positions']]:
           for sidechain_bead in range(monomer_type['sidechain_length']):
             particle_name = str("sc-"+str(cg_particle_index))
             particle_symbol = str("S"+str(cg_particle_index))
             particle = topology.addAtom(particle_symbol, particle_name, residue)
             if sidechain_bead == 0:
              topology.addBond(particle,last_backbone_particle)
             if sidechain_bead != 0:
              topology.addBond(particle,last_sidechain_particle)
             last_sidechain_particle = particle
             cg_particle_index = cg_particle_index + 1
         residue_index = residue_index + 1
        cgmodel.topology = topology
        verify_topology(cgmodel)
        return(topology)


def get_num_forces(cgmodel):
        """
        Given a coarse grained model class object, this function dtermines how many forces we are including when evaluating its energy.

        Parameters
        ----------
        """
        total_forces = 0
        if cgmodel.include_bond_forces: total_forces = total_forces + 1
        if cgmodel.include_nonbonded_forces: total_forces = total_forces + 1
        if cgmodel.include_bond_angle_forces: total_forces = total_forces + 1
        if cgmodel.include_torsion_forces: total_forces = total_forces + 1
        return(total_forces)

def verify_system(cgmodel):
        """
        Given a coarse grained model class object, this function confirms that its OpenMM system object is configured correctly.

        Parameters
        ----------
        """

        if get_num_forces(cgmodel) != cgmodel.system.getNumForces():
          print("ERROR: the number of forces included in the coarse grained model\n")
          print("does not match the number of forces in the OpenMM system object.\n")
          print(" There are "+str(get_num_forces(cgmodel))+" forces in the coarse grained model\n")
          print("and "+str(cgmodel.system.getNumForces())+" forces in the OpenMM System().")
          exit()

        if cgmodel.num_beads != cgmodel.system.getNumParticles():
          print("ERROR: The number of particles in the coarse grained model\n")
          print("does not match the number of particles in the OpenMM system.\n")
          print("There are "+str(cgmodel.num_beads)+" particles in the coarse grained model\n")
          print("and "+str(cgmodel.ssytem.getNumParticles())+" particles in the OpenMM system.")
          exit()

        if cgmodel.constrain_bonds:
          if len(cgmodel.bond_list) != cgmodel.system.getNumConstraints():
            print("ERROR: Bond constraints were requested, but the\n")
            print("number of constraints in the coarse grained model\n")
            print("does not match the number of constraintes in the OpenMM system.\n")
            print("There are "+str(cgmodel.bond_list)+" bond constraints in the coarse grained model\n")
            print("and "+str(cgmodel.ssytem.getNumConstraints())+" constraints in the OpenMM system.")
            exit()


        return



def build_system(cgmodel):
        """
        Builds an OpenMM System() class object, given a CGModel() class object as input.

        Parameters
        ----------

        cgmodel: CGModel() class object

        Returns
        -------

        system: OpenMM System() class object

        """
#        sigma = cgmodel.sigma.in_units_of(unit.nanometer)._value
#        charge = cgmodel.charge._value
#        epsilon = cgmodel.epsilon.in_units_of(unit.kilojoule_per_mole)._value
#        bond_length = cgmodel.bond_length.in_units_of(unit.nanometer)._value

        # Create system
        system = mm.System()
        bead_index = 0
        for monomer_type in cgmodel.sequence:
          for backbone_bead in range(monomer_type['backbone_length']):
            mass = cgmodel.get_particle_mass(bead_index)
            system.addParticle(mass)
            bead_index = bead_index + 1
            if backbone_bead in [monomer_type['sidechain_positions']]:
              for sidechain_bead in range(monomer_type['sidechain_length']):
                mass = cgmodel.get_particle_mass(bead_index)
                system.addParticle(mass)
                bead_index = bead_index + 1

        if cgmodel.include_bond_forces:
         # Create bond (harmonic) potentials
         bond_force = mm.HarmonicBondForce()
         bond_list = []
         for bond_indices in cgmodel.get_bond_list():
              bond_list.append([bond_indices[0]-1,bond_indices[1]-1])
              bond_force_constant = cgmodel.get_bond_force_constant(bond_indices[0]-1,bond_indices[1]-1)
              bond_length = cgmodel.get_bond_length(bond_indices[0]-1,bond_indices[1]-1)
              if cgmodel.constrain_bonds:
               system.addConstraint(bond_indices[0]-1,bond_indices[1]-1,bond_length)
              bond_length = bond_length.in_units_of(unit.nanometer)._value
              bond_force.addBond(bond_indices[0]-1,bond_indices[1]-1,bond_length,bond_force_constant)
         if len(bond_list) != bond_force.getNumBonds():
           print("ERROR: The number of bonds in the coarse grained model is different\n")
           print("from the number of bonds in its OpenMM System object\n")
           print("There are "+str(len(bond_list))+" bonds in the coarse grained model\n")
           print("and "+str(bond_force.getNumBonds())+" bonds in the OpenMM system object.")
           exit()
         system.addForce(bond_force)


        if cgmodel.include_nonbonded_forces:
         # Create nonbonded forces
         nonbonded_force = mm.NonbondedForce()
         bead_index = 0
         for monomer_type in cgmodel.sequence:
          for backbone_bead in range(monomer_type['backbone_length']):
            mass = cgmodel.get_particle_mass(bead_index)
            charge = cgmodel.get_particle_charge(bead_index)
            sigma = cgmodel.get_sigma(bead_index)
            epsilon = cgmodel.get_epsilon(bead_index)
            bead_index = bead_index + 1
            sigma = sigma.in_units_of(unit.nanometer)._value
            charge = charge._value
            epsilon = epsilon.in_units_of(unit.kilojoule_per_mole)._value
            nonbonded_force.addParticle(charge,sigma,epsilon)
            if backbone_bead in [monomer_type['sidechain_positions']]:
              for sidechain_bead in range(monomer_type['sidechain_length']):
                mass = cgmodel.get_particle_mass(bead_index)
                charge = cgmodel.get_particle_charge(bead_index)
                sigma = cgmodel.get_sigma(bead_index)
                epsilon = cgmodel.get_epsilon(bead_index)
                bead_index = bead_index + 1
                sigma = sigma.in_units_of(unit.nanometer)._value
                charge = charge._value
                epsilon = epsilon.in_units_of(unit.kilojoule_per_mole)._value
                nonbonded_force.addParticle(charge,sigma,epsilon)
         nonbonded_force_test = nonbonded_force.__deepcopy__(memo={})
         nonbonded_exclusion_list = []
         # Make nonbonded exclusions from bonds
         for particle_1 in range(cgmodel.num_beads):
           for particle_2 in range(particle_1+1,cgmodel.num_beads):
             for bond in bond_list:
               if particle_1 in bond and particle_2 in bond:
                 if [particle_1,particle_2] not in nonbonded_exclusion_list and [particle_2,particle_1] not in nonbonded_exclusion_list:
                   charge_product = cgmodel.get_particle_charge(particle_1)*cgmodel.get_particle_charge(particle_2)
                   nonbonded_force.addException(particle_1,particle_2,charge_product,sigma,0.0)
                   nonbonded_exclusion_list.append([particle_1,particle_2])
         #print("There are "+str(len(nonbonded_exclusion_list))+" nonbonded exclusions after iterating over bonds.")
         if len(nonbonded_exclusion_list) != len(bond_list):
           print("ERROR: There are "+str(len(nonbonded_exclusion_list))+" nonbonded particle exclusions built from bonds, however,")
           print("there are "+str(len(bond_list))+" bonds in the coarse grained model.")
           exit()
         # Make nonbonded exclusions from angles
         for particle_1 in range(cgmodel.num_beads):
           for particle_2 in range(particle_1+1,cgmodel.num_beads):
            if [particle_1,particle_2] not in nonbonded_exclusion_list and [particle_2,particle_1] not in nonbonded_exclusion_list:
             angle_list = [[angle[0]-1,angle[1]-1,angle[2]-1] for angle in cgmodel.bond_angle_list]
             for angle in angle_list:
               if particle_1 in angle and particle_2 in angle:
                 if [particle_1,particle_2] not in nonbonded_exclusion_list and [particle_2,particle_1] not in nonbonded_exclusion_list:
                   charge_product = cgmodel.get_particle_charge(particle_1)*cgmodel.get_particle_charge(particle_2)
                   nonbonded_force.addException(particle_1,particle_2,charge_product,sigma,0.0)
                   nonbonded_exclusion_list.append([particle_1,particle_2])
         #print("There are "+str(len(nonbonded_exclusion_list))+" nonbonded exclusions after iterating over bond angles.")

         for particle_1 in range(cgmodel.num_beads):
           for particle_2 in range(particle_1+1,cgmodel.num_beads):
            if [particle_1,particle_2] not in nonbonded_exclusion_list and [particle_2,particle_1] not in nonbonded_exclusion_list:
             torsion_list = [[torsion[0]-1,torsion[1]-1,torsion[2]-1,torsion[3]-1] for torsion in cgmodel.torsion_list]
             for torsion in torsion_list:
               if particle_1 in torsion and particle_2 in torsion:
                 if (particle_1 == torsion[0] and particle_2 == torsion[3]) or (particle_1 == torsion[3] and particle_2 == torsion[0]):
                  if [particle_1,particle_2] not in nonbonded_exclusion_list and [particle_2,particle_1] not in nonbonded_exclusion_list:
                   charge_product = cgmodel.get_particle_charge(particle_1)*cgmodel.get_particle_charge(particle_2)
                   if cgmodel.include_torsion_forces:
                     nonbonded_force.addException(particle_1,particle_2,charge_product,sigma,epsilon)
                   else:
                     nonbonded_force.addException(particle_1,particle_2,charge_product,sigma,0.0)
                   nonbonded_exclusion_list.append([particle_1,particle_2])

         #print("There are "+str(len(nonbonded_exclusion_list))+" nonbonded exclusions after iterating over torsions.")

         if cgmodel.num_beads != nonbonded_force.getNumParticles():
           print("ERROR: The number of particles in the coarse grained model is different")
           print("from the number of particles with nonbonded force definitions in the OpenMM NonbondedForce.\n")
           print("There are "+str(cgmodel.num_beads)+" particles in the coarse grained model")
           print("and "+str(nonbonded_force.getNumParticles())+" particles in the OpenMM NonbondedForce.")
           exit()

#         nonbonded_force_test.createExceptionsFromBonds(bond_list,1.0,1.0)
#         if len(nonbonded_exclusion_list) != nonbonded_force_test.getNumExceptions():
           #print("ERROR: The number of nonbonded exceptions in the coarse grained model is different\n")
           #print("from the number of exceptions generated with OpenMM NonbondedForce.createExceptionsFromBonds()\n")
           #print("There are "+str(len(nonbonded_exclusion_list))+" exceptions in the coarse grained model\n")
           #print("and "+str(nonbonded_force_test.getNumExceptions())+" exceptions in the OpenMM NonbondedForce.")
#           exit()
         system.addForce(nonbonded_force)


        if cgmodel.include_bond_angle_forces:
         # Create bond angle potentials
         angle_force = mm.HarmonicAngleForce()
         for angle in angle_list:
              bond_angle_force_constant = cgmodel.get_bond_angle_force_constant(angle[0],angle[1],angle[2])
              equil_bond_angle = cgmodel.get_equil_bond_angle(angle[0],angle[1],angle[2]) * np.pi/180.0
              angle_force.addAngle(angle[0],angle[1],angle[2],equil_bond_angle,bond_angle_force_constant)
         system.addForce(angle_force)

        if cgmodel.include_torsion_forces:

         # Create torsion potentials
         torsion_force = mm.PeriodicTorsionForce()
         for torsion in torsion_list:
              torsion_force_constant = cgmodel.get_torsion_force_constant(torsion)
              equil_torsion_angle = cgmodel.get_equil_torsion_angle(torsion) * np.pi/180.0
              periodicity = 1
              torsion_force.addTorsion(torsion[0],torsion[1],torsion[2],torsion[3],periodicity,equil_torsion_angle,torsion_force_constant)
         system.addForce(torsion_force)


        return(system)


