import os
import datetime
import numpy as np
from simtk import openmm as mm
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from simtk.openmm.app.topology import Topology
import simtk.openmm.app.element as elem
from cg_openmm.simulation.tools import build_mm_simulation
from cg_openmm.utilities.util import lj_v
from cg_openmm.utilities.iotools import write_pdbfile_without_topology

def add_new_elements(cgmodel):
        """
        Add coarse grained particle types to OpenMM.

        :param cgmodel: CGModel object (contains all attributes for a coarse grained model).
        :type cgmodel: class

        :returns: 
           - particle_list (list) - a list of the particles that were added to OpenMM's 'Element' List.

        :Example:

        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> cgmodel = CGModel()
        >>> particle_types = add_new_elements(cgmodel)

        .. warning:: If the particle names were user defined, and any of the names conflict with existing element names in OpenMM, OpenMM will issue an error exit.

        """
        element_index = 117
        cg_particle_index = 1
        particle_list = []

        for monomer_type in cgmodel.monomer_types:
         for backbone_bead in range(monomer_type['backbone_length']):
          particle_name = str("X"+str(cg_particle_index))
          particle_symbol = str("X"+str(cg_particle_index))
          if particle_symbol not in elem.Element._elements_by_symbol:
           mass = cgmodel.get_particle_mass(cg_particle_index-1)
           elem.Element(element_index,particle_name,particle_symbol,mass)
           particle_list.append(particle_symbol)
           element_index = element_index + 1
          cg_particle_index = cg_particle_index + 1
          if type(monomer_type['sidechain_positions']) == int:
           sidechain_positions = [monomer_type['sidechain_positions']]
          else:
           sidechain_positions = monomer_type['sidechain_positions']
          if backbone_bead in sidechain_positions:
           for sidechain in range(monomer_type['sidechain_length']):
             particle_name = str("A"+str(cg_particle_index))
             particle_symbol = str("A"+str(cg_particle_index))
             if particle_symbol not in elem.Element._elements_by_symbol:
               mass = cgmodel.get_particle_mass(cg_particle_index-1)
               elem.Element(element_index,particle_name,particle_symbol,mass)
               particle_list.append(particle_symbol)
               element_index = element_index + 1
             cg_particle_index = cg_particle_index + 1
        return(particle_list)

def write_xml_file(cgmodel,xml_file_name):
        """
        Write an XML-formatted forcefield file for a coarse grained model.
         
        :param cgmodel: CGModel() class object.
        :type cgmodel: class

        :param xml_file_name: Path to XML output file.
        :type xml_file_name: str

        :Example:

        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> cgmodel = CGModel()
        >>> xml_file_name = "openmm_cgmodel.xml"
        >>> write_xml_file(cgmodel,xml_file_name)

        """
        particle_list = add_new_elements(cgmodel)
        xml_object = open(xml_file_name,"w")
        xml_object.write("<ForceField>\n")
        xml_object.write(" <Info>\n")
        date = str(datetime.datetime.today()).split()[0]
        xml_object.write("  <DateGenerated> "+str(date)+" </DateGenerated>\n")
        xml_object.write("  <Source> https://github.com/shirtsgroup/cg_openmm </Source>\n")
        xml_object.write("  <Reference>\n")
        xml_object.write("  </Reference>\n")
        xml_object.write(" </Info>\n")
        xml_object.write(" <AtomTypes>\n")
        unique_particle_names = []
        unique_masses = []
        for particle_index in range(len(cgmodel.particle_list)):
          if cgmodel.particle_list[particle_index] not in unique_particle_names:
            unique_particle_names.append(cgmodel.particle_list[particle_index])
            unique_masses.append(cgmodel.get_particle_mass(particle_index))
        for particle_index in range(len(unique_particle_names)):          
          particle_type = cgmodel.get_particle_type(particle_index)
          xml_object.write('  <Type name="'+str(unique_particle_names[particle_index])+'" class="'+str(particle_type)+'" element="'+str(unique_particle_names[particle_index])+'" mass="'+str(unique_masses[particle_index]._value)+'"/>\n')
        xml_object.write(" </AtomTypes>\n")
        xml_object.write(" <Residues>\n")
        xml_object.write('  <Residue name="M">\n')
        for particle_index in range(len(unique_particle_names)):
          xml_object.write('   <Atom name="'+str(unique_particle_names[particle_index])+'" type="'+str(unique_particle_names[particle_index])+'"/>\n')
        for bond in cgmodel.bond_list:
          if all(bond[i] < len(unique_particle_names) for i in range(2)): 
            particle_1_name = cgmodel.get_particle_name(bond[0])
            particle_2_name = cgmodel.get_particle_name(bond[1])
            xml_object.write('   <Bond atomName1="'+str(particle_1_name)+'" atomName2="'+str(particle_2_name)+'"/>\n')
        xml_object.write('   <ExternalBond atomName="'+str(unique_particle_names[0])+'"/>\n')
        external_parent = unique_particle_names[len(unique_particle_names)-cgmodel.monomer_types[0]['sidechain_length']-1]
        xml_object.write('   <ExternalBond atomName="'+str(external_parent)+'"/>\n')
        xml_object.write("  </Residue>\n")
        xml_object.write('  <Residue name="MT">\n')
        for particle_index in range(len(unique_particle_names)):
          xml_object.write('   <Atom name="'+str(unique_particle_names[particle_index])+'" type="'+str(unique_particle_names[particle_index])+'"/>\n')
        for bond in cgmodel.bond_list:
          if all(bond[i] < len(unique_particle_names) for i in range(2)):
            particle_1_name = cgmodel.get_particle_name(bond[0])
            particle_2_name = cgmodel.get_particle_name(bond[1])
            xml_object.write('   <Bond atomName1="'+str(particle_1_name)+'" atomName2="'+str(particle_2_name)+'"/>\n')
        xml_object.write('   <ExternalBond atomName="'+str(external_parent)+'"/>\n')
        xml_object.write("  </Residue>\n")
        xml_object.write(" </Residues>\n")
        if cgmodel.include_bond_forces:
          xml_object.write(" <HarmonicBondForce>\n")
          unique_bond_list = []
          for bond in cgmodel.bond_list:
            if any(bond[i] < len(unique_particle_names) for i in range(2)):
              unique_bond_list.append(bond)
          for bond in unique_bond_list:
            particle_1_name = cgmodel.get_particle_name(bond[0])
            particle_2_name = cgmodel.get_particle_name(bond[1])
            xml_type_1 = particle_1_name #unique_particle_names.index(particle_1_name)
            xml_type_2 = particle_2_name #unique_particle_names.index(particle_2_name)
            bond_length = cgmodel.get_bond_length(bond[0],bond[1]).in_units_of(unit.nanometer)._value
            bond_force_constant = cgmodel.get_bond_force_constant(bond[0],bond[1])
            xml_object.write('  <Bond type1="'+str(xml_type_1)+'" type2="'+str(xml_type_2)+'" length="'+str(bond_length)+'" k="'+str(bond_force_constant)+'"/>\n')
          xml_object.write(" </HarmonicBondForce>\n")
        if cgmodel.include_bond_angle_forces:
          xml_object.write(" <HarmonicAngleForce>\n")
          unique_angle_list = []
          for angle in cgmodel.bond_angle_list:
            if any(angle[i] < len(unique_particle_names) for i in range(3)):
              unique_angle_list.append(angle)
          for angle in unique_angle_list:
            bond_angle_force_constant = cgmodel.get_bond_angle_force_constant(angle[0],angle[1],angle[2])
            equil_bond_angle = cgmodel.get_equil_bond_angle(angle[0],angle[1],angle[2])
            particle_1_name = cgmodel.get_particle_name(angle[0])
            particle_2_name = cgmodel.get_particle_name(angle[1])
            particle_3_name = cgmodel.get_particle_name(angle[2])
            xml_type_1 = particle_1_name #unique_particle_names.index(particle_1_name)
            xml_type_2 = particle_2_name #unique_particle_names.index(particle_2_name)
            xml_type_3 = particle_3_name #unique_particle_names.index(particle_3_name)
            xml_object.write('  <Angle angle="'+str(equil_bond_angle)+'" k="'+str(bond_angle_force_constant)+'" type1="'+str(xml_type_1)+'" type2="'+str(xml_type_2)+'" type3="'+str(xml_type_3)+'"/>\n')
          xml_object.write(" </HarmonicAngleForce>\n")
        if cgmodel.include_torsion_forces:
          xml_object.write(' <PeriodicTorsionForce ordering="amber">\n')
          unique_torsion_list = []
          #print(cgmodel.torsion_list)
          for torsion in cgmodel.torsion_list:
            if any(torsion[i] < len(unique_particle_names) for i in range(4)):
              unique_torsion_list.append(torsion)
          for torsion in unique_torsion_list:
            torsion_force_constant = cgmodel.get_torsion_force_constant([torsion[0],torsion[1],torsion[2],torsion[3]])
            equil_torsion_angle = cgmodel.get_equil_torsion_angle([torsion[0],torsion[1],torsion[2],torsion[3]])
            particle_1_name = cgmodel.get_particle_name(torsion[0])
            particle_2_name = cgmodel.get_particle_name(torsion[1])
            particle_3_name = cgmodel.get_particle_name(torsion[2])
            particle_4_name = cgmodel.get_particle_name(torsion[3])
            xml_type_1 = particle_1_name #unique_particle_names.index(particle_1_name)
            xml_type_2 = particle_2_name #unique_particle_names.index(particle_2_name)
            xml_type_3 = particle_3_name #unique_particle_names.index(particle_3_name)
            xml_type_4 = particle_4_name #unique_particle_names.index(particle_4_name)
            periodicity = cgmodel.get_torsion_periodicity(torsion)
            xml_object.write('  <Proper k1="'+str(torsion_force_constant)+'" periodicity1="'+str(periodicity)+'" phase1="'+str(equil_torsion_angle)+'" type1="'+str(xml_type_1)+'" type2="'+str(xml_type_2)+'" type3="'+str(xml_type_3)+'" type4="'+str(xml_type_4)+'"/>\n')
          xml_object.write(" </PeriodicTorsionForce>\n")
        if cgmodel.include_nonbonded_forces:
          xml_object.write(' <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">\n')
          for particle_index in range(len(unique_particle_names)):
            charge = cgmodel.get_particle_charge(particle_index)._value
            sigma = cgmodel.get_sigma(particle_index).in_units_of(unit.nanometer)._value
            epsilon = cgmodel.get_epsilon(particle_index)._value  
            particle_name = cgmodel.get_particle_name(particle_index)
            xml_object.write('  <Atom type="'+str(particle_name)+'" charge="'+str(charge)+'" sigma="'+str(sigma)+'" epsilon="'+str(epsilon)+'"/>\n')
          xml_object.write(" </NonbondedForce>\n")
        xml_object.write("</ForceField>\n")
        xml_object.close()
        return

def verify_topology(cgmodel):
        """
        Given a coarse grained model that contains a Topology() (cgmodel.topology), this function verifies the validity of the topology.

        :param cgmodel: CGModel() class object.
        :type cgmodel: class

        :Example:

        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> cgmodel = CGModel()
        >>> verify_topology(cgmodel)

        .. warning:: The function will force an error exit if the topology is invalid, and will proceed as normal if the topology is valid. 

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

def build_topology(cgmodel,use_pdbfile=False,pdbfile=None):
        """

        Construct an OpenMM `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_ class object for our coarse grained model,

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :param use_pdbfile: Determines whether or not to use a PDB file in order to generate the Topology().
        :type use_pdbfile: Logical

        :param pdbfile: Name of a PDB file to use when building the topology.
        :type pdbfile: str

        :returns:
            - topology (`Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_ ) - OpenMM Topology() object

        :Example:

        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> from foldamers.util.iotools import write_pdbfile_without_topology
        >>> input_pdb = "top.pdb"
        >>> cgmodel = CGModel()
        >>> write_pdbfile_without_topology(cgmodel,input_pdb)
        >>> topology = build_topology(cgmodel,use_pdbfile=True,pdbfile=input_pdb)
        >>> cgmodel.topology = topology

        .. warning:: When 'use_pdbfile'=True, this function will use the `PDBFile() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1pdbfile_1_1PDBFile.html>`_ class object from OpenMM to build the Topology().  In order for this approach to function correctly, the particle names in the PDB file must match the particle names in the coarse grained model.

        """
        if cgmodel.constrain_bonds == True:
         use_pdbfile = True
        if use_pdbfile == True:
         if pdbfile == None:
           write_pdbfile_without_topology(cgmodel,"topology_source.pdb")
           pdb = PDBFile("topology_source.pdb")
           topology = pdb.getTopology()
           os.remove("topology_source.pdb")
           return(topology)
         else:
           pdb = PDBFile(pdbfile)
           topology = pdb.getTopology()
           return(topology)

        topology = Topology()

        chain = topology.addChain()
        residue_index = 1
        cg_particle_index = 0
        for monomer_type in cgmodel.sequence:
         residue = topology.addResidue(str(residue_index), chain)
         for backbone_bead in range(monomer_type['backbone_length']):
          particle_symbol = cgmodel.get_particle_name(cg_particle_index)
          element = elem.Element.getBySymbol(particle_symbol)
          particle = topology.addAtom(particle_symbol, element, residue)
          if backbone_bead == 0 and residue_index != 1:
           if cgmodel.include_bond_forces or cgmodel.constrain_bonds:
            topology.addBond(particle,last_backbone_particle)
          last_backbone_particle = particle
          cg_particle_index = cg_particle_index + 1
          if backbone_bead in [monomer_type['sidechain_positions']]:
           for sidechain_bead in range(monomer_type['sidechain_length']):
             particle_symbol = cgmodel.get_particle_name(cg_particle_index)
             element = elem.Element.getBySymbol(particle_symbol)
             particle = topology.addAtom(particle_symbol, element, residue)
             if sidechain_bead == 0:
              if cgmodel.include_bond_forces or cgmodel.constrain_bonds:
               topology.addBond(particle,last_backbone_particle)
             if sidechain_bead != 0:
              if cgmodel.include_bond_forces or cgmodel.constrain_bonds:
               topology.addBond(particle,last_sidechain_particle)
             last_sidechain_particle = particle
             cg_particle_index = cg_particle_index + 1
         residue_index = residue_index + 1
        cgmodel.topology = topology
        verify_topology(cgmodel)
        return(topology)


def get_num_forces(cgmodel):
        """
        Given a CGModel() class object, this function determines how many forces we are including when evaluating the energy.

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :returns: 
            - total_forces (int) - Number of forces in the coarse grained model

        :Example:

        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> cgmodel = CGModel()
        >>> total_number_forces = get_num_forces(cgmodel)

        """
        total_forces = 0
        if cgmodel.include_bond_forces: total_forces = total_forces + 1
        if cgmodel.include_nonbonded_forces: total_forces = total_forces + 1
        if cgmodel.include_bond_angle_forces: total_forces = total_forces + 1
        if cgmodel.include_torsion_forces: total_forces = total_forces + 1
        return(total_forces)

def verify_system(cgmodel):
        """
        Given a CGModel() class object, this function confirms that its OpenMM `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_ object is configured correctly.

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :Example:

        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> cgmodel = CGModel()
        >>> verify_system(cgmodel)

        .. warning:: The function will force an error exit if the system is invalid, and will proceed as normal if the system is valid.

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
            print("and "+str(cgmodel.system.getNumConstraints())+" constraints in the OpenMM system.")
            exit()

        return

def test_force(cgmodel,force,force_type=None):
        """

        Given an OpenMM `Force() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1Force.html>`_, this function determines if there are any problems with its configuration.

        :param cgmodel: CGModel() class object.
        :type cgmodel: class

        :param force: An OpenMM Force() object.
        :type force: `Force() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1Force.html>`_, this function determines if there are any problems with its configuration.

        :param force_type: Designates the kind of 'force' provided. (Valid options include: "Nonbonded")
        :type force_type: str

        :returns: 
            - 'success' (Logical) - a variable indicating if the force test passed.

        :Example:

        >>> from simtk.openmm.openmm import NonbondedForce
        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> cgmodel = CGModel()
        >>> force = NonbondedForce()
        >>> force_type = "Nonbonded"
        >>> test_result = test_force(cgmodel,force,force_type="Nonbonded")

        """
        success=True
        if force_type == "Nonbonded":
           if cgmodel.num_beads != force.getNumParticles():
             print("ERROR: The number of particles in the coarse grained model is different")
             print("from the number of particles with nonbonded force definitions in the OpenMM NonbondedForce.\n")
             print("There are "+str(cgmodel.num_beads)+" particles in the coarse grained model")
             print("and "+str(force.getNumParticles())+" particles in the OpenMM NonbondedForce.")
             success=False

           total_nonbonded_energy = 0.0 * unit.kilojoule_per_mole
           #print(cgmodel.nonbonded_interaction_list)
           for nonbonded_interaction in cgmodel.nonbonded_interaction_list:
             particle_1_positions = cgmodel.positions[nonbonded_interaction[0]]
             particle_2_positions = cgmodel.positions[nonbonded_interaction[1]]
             sigma = cgmodel.get_sigma(nonbonded_interaction[0])
             epsilon = cgmodel.get_epsilon(nonbonded_interaction[0])
             int_energy = lj_v(particle_1_positions,particle_2_positions,sigma,epsilon)
             total_nonbonded_energy = total_nonbonded_energy.__add__(int_energy)
           
           cgmodel.include_bond_forces = False
           cgmodel.include_bond_angle_forces = False
           cgmodel.include_torsion_forces = False
           cgmodel.topology = build_topology(cgmodel)
           cgmodel.simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions,simulation_time_step=5.0*unit.femtosecond,print_frequency=1)
           potential_energy = cgmodel.simulation.context.getState(getEnergy=True).getPotentialEnergy()

           #if potential_energy.__sub__(total_nonbonded_energy).__gt__(0.1 * unit.kilojoule_per_mole):
             #print("Warning: The nonbonded potential energy computed by hand does not agree")
             #print("with the value computed by OpenMM.")
             #print("The value computed by OpenMM was: "+str(potential_energy))
             #print("The value computed by hand was: "+str(total_nonbonded_energy))
             #print("Check the units for your model parameters.  If the problem persists, there")
             #print("could be some other problem with the configuration of your coarse grained model.")
             #success = False
           #else:
             #print("The OpenMM nonbonded energy matches the energy computed by hand:")
             #print(str(potential_energy))

        return(success)

def add_rosetta_exception_parameters(cgmodel,nonbonded_force,particle_index_1,particle_index_2):
        """
        """
        exception_list = []
        for exception in range(nonbonded_force.getNumExceptions()):
          index_1,index_2,charge,sigma,epsilon = nonbonded_force.getExceptionParameters(exception)
          if [index_1,index_2] not in exception_list and [index_2,index_1] not in exception_list:
            exception_list.append([index_1,index_2])
        
        if [particle_index_1,particle_index_2] not in exception_list and [particle_index_2,particle_index_1] not in exception_list:
          charge_1 = cgmodel.get_particle_charge(particle_index_1)
          sigma_1 = cgmodel.get_sigma(particle_index_1).in_units_of(unit.nanometer)
          epsilon_1 = cgmodel.get_epsilon(particle_index_1).in_units_of(unit.kilojoule_per_mole)
          charge_2 = cgmodel.get_particle_charge(particle_index_2)
          sigma_2 = cgmodel.get_sigma(particle_index_2).in_units_of(unit.nanometer)
          epsilon_2 = cgmodel.get_epsilon(particle_index_2).in_units_of(unit.kilojoule_per_mole)
          nonbonded_force.addException(particle_index_1,particle_index_2,0.2*charge_1*charge_2,0.2*sigma_1*sigma_2,0.2*epsilon_1*epsilon_2)
        return(nonbonded_force)

def add_force(cgmodel,force_type=None,rosetta_scoring=False):
        """

        Given a 'cgmodel' and 'force_type' as input, this function adds
        the OpenMM force corresponding to 'force_type' to 'cgmodel.system'.

        :param cgmodel: CGModel() class object.
        :param type: class

        :param force_type: Designates the kind of 'force' provided. (Valid options include: "Bond", "Nonbonded", "Angle", and "Torsion")
        :type force_type: str

        :returns: 
             - cgmodel (class) - 'foldamers' CGModel() class object
             - force (class) - An OpenMM `Force() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1Force.html>`_ object.

        :Example:

        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> cgmodel = CGModel()
        >>> force_type = "Bond"
        >>> cgmodel,force = add_force(cgmodel,force_type=force_type)

        """
        if force_type == "Bond":

          bond_force = mm.HarmonicBondForce()
          bond_list = []

          for bond_indices in cgmodel.get_bond_list():
              bond_list.append([bond_indices[0],bond_indices[1]])
              if cgmodel.include_bond_forces:
                bond_force_constant = cgmodel.get_bond_force_constant(bond_indices[0],bond_indices[1])
                bond_length = cgmodel.get_bond_length(bond_indices[0],bond_indices[1]).in_units_of(unit.nanometer)._value
                bond_force.addBond(bond_indices[0],bond_indices[1],bond_length,bond_force_constant)
              if cgmodel.constrain_bonds:
                bond_length = cgmodel.get_bond_length(bond_indices[0],bond_indices[1]).in_units_of(unit.nanometer)._value
                if not cgmodel.include_bond_forces:
                  bond_force_constant = 0.0 * cgmodel.get_bond_force_constant(bond_indices[0],bond_indices[1])
                  bond_force.addBond(bond_indices[0],bond_indices[1],bond_length,bond_force_constant)
                cgmodel.system.addConstraint(bond_indices[0],bond_indices[1],bond_length)

          if len(bond_list) != bond_force.getNumBonds():
            print("ERROR: The number of bonds in the coarse grained model is different\n")
            print("from the number of bonds in its OpenMM System object\n")
            print("There are "+str(len(bond_list))+" bonds in the coarse grained model\n")
            print("and "+str(bond_force.getNumBonds())+" bonds in the OpenMM system object.")
            exit()

          cgmodel.system.addForce(bond_force)
          force = bond_force

        if force_type == "Nonbonded":

          nonbonded_force = mm.NonbondedForce()
          nonbonded_force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)

          for particle in range(cgmodel.num_beads):
            charge = cgmodel.get_particle_charge(particle)
            sigma = cgmodel.get_sigma(particle)
            epsilon = cgmodel.get_epsilon(particle)
            nonbonded_force.addParticle(charge,sigma,epsilon)

          if len(cgmodel.bond_list) >= 1:
            if not rosetta_scoring:
              nonbonded_force.createExceptionsFromBonds(cgmodel.bond_list,1.0,1.0)
            if rosetta_scoring:
              nonbonded_force.createExceptionsFromBonds(cgmodel.bond_list,0.0,0.0)
              for torsion in cgmodel.torsion_list:
                for bond in cgmodel.bond_list:
                  if bond[0] not in torsion:
                    if bond[1] == torsion[0]:
                      nonbonded_force = add_rosetta_exception_parameters(cgmodel,nonbonded_force,bond[0],torsion[3])
                    if bond[1] == torsion[3]:
                      nonbonded_force = add_rosetta_exception_parameters(cgmodel,nonbonded_force,bond[0],torsion[0])
                  if bond[1] not in torsion:
                    if bond[0] == torsion[0]:
                      nonbonded_force = add_rosetta_exception_parameters(cgmodel,nonbonded_force,bond[1],torsion[3])
                    if bond[0] == torsion[3]:
                      nonbonded_force = add_rosetta_exception_parameters(cgmodel,nonbonded_force,bond[1],torsion[0])
          cgmodel.system.addForce(nonbonded_force)
          force = nonbonded_force
          #for particle in range(cgmodel.num_beads):
            #print(force.getParticleParameters(particle))

        if force_type == "Angle":
          angle_force = mm.HarmonicAngleForce()  
          for angle in cgmodel.bond_angle_list:
            bond_angle_force_constant = cgmodel.get_bond_angle_force_constant(angle[0],angle[1],angle[2])
            equil_bond_angle = cgmodel.get_equil_bond_angle(angle[0],angle[1],angle[2])
            angle_force.addAngle(angle[0],angle[1],angle[2],equil_bond_angle,bond_angle_force_constant)
          cgmodel.system.addForce(angle_force)
          force = angle_force

        if force_type == "Torsion":
          torsion_force = mm.PeriodicTorsionForce()
          for torsion in cgmodel.torsion_list:
            torsion_force_constant = cgmodel.get_torsion_force_constant(torsion)
            equil_torsion_angle = cgmodel.get_equil_torsion_angle(torsion)
            periodicity = cgmodel.get_torsion_periodicity(torsion)
            torsion_force.addTorsion(torsion[0],torsion[1],torsion[2],torsion[3],periodicity,equil_torsion_angle,torsion_force_constant)
              #print(torsion_force.getNumTorsions())
          cgmodel.system.addForce(torsion_force)
          force = torsion_force

        return(cgmodel,force)

def test_forces(cgmodel):
        """
        Given a cgmodel that contains positions and an
        an OpenMM System() object, this function tests 
        the forces for cgmodel.system.

        More specifically, this function confirms that the
        model does not have any "NaN" or unphysically large forces.

        :param cgmodel: CGModel() class object.
        :param type: class

        :returns:
            - success (Logical) - Indicates if this cgmodel has unphysical forces.

        :Example:

        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> cgmodel = CGModel()
        >>> pass_forces_test = test_forces(cgmodel)

        """
        if cgmodel.topology == None:
          cgmodel.topology = build_topology(cgmodel)
        simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions,simulation_time_step=5.0*unit.femtosecond,print_frequency=1)
        forces = simulation.context.getState(getForces=True).getForces()
        success = True
        for force in forces:
          for component in force:
            if 'nan' in str(component):
              print("Detected 'nan' force value")
              print("for particle "+str(forces.index(force)))
              success = False
              return(success)
            if component.__gt__(9.9e9 * component.unit):
              print("Detected unusually large forces")
              print("for particle "+str(forces.index(force)))
              print("The force is: "+str("{:.2e}".format(component._value))+" "+str(component.unit))
              success = False
              return(success)
        return(success)

def build_system(cgmodel,rosetta_scoring=False,verify=True):
        """
        Builds an OpenMM `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_ object, given a CGModel() as input.

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :returns: 
            - system ( `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_ ) - OpenMM System() object

        :Example:

        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> cgmodel = CGModel()
        >>> system = build_system(cgmodel)
        >>> cgmodel.system = system

        """
        # Create system
        system = mm.System()
        for particle in range(cgmodel.num_beads):
            mass = cgmodel.get_particle_mass(particle)
            system.addParticle(mass)
        cgmodel.system = system

        #length_scale = cgmodel.bond_lengths['bb_bb_bond_length']
        #box_vectors = [[100.0*length_scale._value,0.0,0.0],[0.0,100.0*length_scale._value,0.0],[0.0,0.0,100.0*length_scale._value]]
        #system.setDefaultPeriodicBoxVectors(box_vectors[0],box_vectors[1],box_vectors[2])

        if cgmodel.include_bond_forces or cgmodel.constrain_bonds:
         # Create bond (harmonic) potentials
         cgmodel,bond_force = add_force(cgmodel,force_type="Bond")
         if cgmodel.positions != None:
          if not test_force(cgmodel,bond_force,force_type="Bond"):
           print("ERROR: The bond force definition is giving 'nan'")
           exit()

        if cgmodel.include_nonbonded_forces:
         # Create nonbonded forces
          cgmodel,nonbonded_force = add_force(cgmodel,force_type="Nonbonded",rosetta_scoring=rosetta_scoring)

          #if cgmodel.positions != None:
           #print("Testing the nonbonded forces")
           #if not test_force(cgmodel,nonbonded_force,force_type="Nonbonded"):
            #print("ERROR: there was a problem with the nonbonded force definitions.")
            #exit()

        if cgmodel.include_bond_angle_forces:
          # Create bond angle potentials
          cgmodel,bond_angle_force = add_force(cgmodel,force_type="Angle")
          if cgmodel.positions != None:
            if not test_force(cgmodel,bond_angle_force,force_type="Angle"):
              print("ERROR: There was a problem with the bond angle force definitions.")
              exit()

        if cgmodel.include_torsion_forces:
          # Create torsion potentials
          cgmodel,torsion_force = add_force(cgmodel,force_type="Torsion")
          if cgmodel.positions != None:
            if not test_force(cgmodel,torsion_force,force_type="Torsion"):
              print("ERROR: There was a problem with the torsion definitions.")
              exit()

        if verify:
          if cgmodel.positions != None:
            if not test_forces(cgmodel):
              print("ERROR: There was a problem with the forces.")
              exit()

        return(system)
