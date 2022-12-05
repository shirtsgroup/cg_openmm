import datetime
import os
import tempfile

import numpy as np
import openmm as mm
import openmm.app.element as elem
from cg_openmm.simulation.tools import build_mm_simulation
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from cg_openmm.utilities.util import lj_v, lj_go
from openmm import unit, LangevinIntegrator
from openmm.app.pdbfile import PDBFile
from openmm.app.topology import Topology
from openmm.app.simulation import Simulation


def add_new_elements(cgmodel):
    """
    Add coarse grained particle types to OpenMM.

    :param cgmodel: CGModel object (contains all attributes for a coarse grained model).
    :type cgmodel: class

    :returns:
       - new_particles (list) - a list of the particle names that were added to OpenMM's 'Element' List.

    :Example:

    >>> from foldamers.cg_model.cgmodel import CGModel
    >>> cgmodel = CGModel()
    >>> particle_types = add_new_elements(cgmodel)

    .. warning:: If the particle names were user defined, and any of the names conflict with existing element names in OpenMM, OpenMM will issue an error exit.

    """
    element_index = 117
    new_particles = []

    for particle in cgmodel.particle_list:
        particle_name = particle["name"]
        if particle_name.upper() not in elem.Element._elements_by_symbol:
            elem.Element(element_index, particle_name, particle_name, cgmodel.get_particle_mass(particle))
            element_index = element_index + 1
            new_particles.append(particle_name)

    return new_particles


def write_xml_file(cgmodel, xml_file_name):
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
    xml_object = open(xml_file_name, "w")
    xml_object.write("<ForceField>\n")
    xml_object.write(" <Info>\n")
    date = str(datetime.datetime.today()).split()[0]
    xml_object.write(f"  <DateGenerated> {date} </DateGenerated>\n")
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
        particle_type = cgmodel.get_particle_type_name(particle_index)
        xml_object.write(
            '  <Type name="'
            + str(unique_particle_names[particle_index])
            + '" class="'
            + str(particle_type)
            + '" element="'
            + str(unique_particle_names[particle_index])
            + '" mass="'
            + str(unique_masses[particle_index]._value)
            + '"/>\n'
        )
    xml_object.write(" </AtomTypes>\n")
    xml_object.write(" <Residues>\n")
    xml_object.write('  <Residue name="M">\n')
    for particle_index in range(len(unique_particle_names)):
        xml_object.write(
            '   <Atom name="'
            + str(unique_particle_names[particle_index])
            + '" type="'
            + str(unique_particle_names[particle_index])
            + '"/>\n'
        )
    for bond in cgmodel.bond_list:
        if all(bond[i] < len(unique_particle_names) for i in range(2)):
            particle_1_name = cgmodel.get_particle_name(bond[0])
            particle_2_name = cgmodel.get_particle_name(bond[1])
            xml_object.write(
                '   <Bond atomName1="'
                + str(particle_1_name)
                + '" atomName2="'
                + str(particle_2_name)
                + '"/>\n'
            )
    xml_object.write('   <ExternalBond atomName="' + str(unique_particle_names[0]) + '"/>\n')
    external_parent = unique_particle_names[
        len(unique_particle_names) - cgmodel.monomer_types[0]["sidechain_length"] - 1
    ]
    xml_object.write('   <ExternalBond atomName="' + str(external_parent) + '"/>\n')
    xml_object.write("  </Residue>\n")
    xml_object.write('  <Residue name="MT">\n')
    for particle_index in range(len(unique_particle_names)):
        xml_object.write(
            '   <Atom name="'
            + str(unique_particle_names[particle_index])
            + '" type="'
            + str(unique_particle_names[particle_index])
            + '"/>\n'
        )
    for bond in cgmodel.bond_list:
        if all(bond[i] < len(unique_particle_names) for i in range(2)):
            particle_1_name = cgmodel.get_particle_name(bond[0])
            particle_2_name = cgmodel.get_particle_name(bond[1])
            xml_object.write(
                '   <Bond atomName1="'
                + str(particle_1_name)
                + '" atomName2="'
                + str(particle_2_name)
                + '"/>\n'
            )
    xml_object.write('   <ExternalBond atomName="' + str(external_parent) + '"/>\n')
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
            # unique_particle_names.index(particle_1_name)
            xml_type_1 = particle_1_name
            # unique_particle_names.index(particle_2_name)
            xml_type_2 = particle_2_name
            bond_length = cgmodel.get_bond_length(bond).value_in_unit(unit.nanometer)
            bond_force_constant = cgmodel.get_bond_force_constant(bond)
            xml_object.write(
                '  <Bond type1="'
                + str(xml_type_1)
                + '" type2="'
                + str(xml_type_2)
                + '" length="'
                + str(bond_length)
                + '" k="'
                + str(bond_force_constant)
                + '"/>\n'
            )
        xml_object.write(" </HarmonicBondForce>\n")
    if cgmodel.include_bond_angle_forces:
        xml_object.write(" <HarmonicAngleForce>\n")
        unique_angle_list = []
        for angle in cgmodel.bond_angle_list:
            if any(angle[i] < len(unique_particle_names) for i in range(3)):
                unique_angle_list.append(angle)
        for angle in unique_angle_list:
            bond_angle_force_constant = cgmodel.get_bond_angle_force_constant(angle)
            equil_bond_angle = cgmodel.get_equil_bond_angle(angle)
            particle_1_name = cgmodel.get_particle_name(angle[0])
            particle_2_name = cgmodel.get_particle_name(angle[1])
            particle_3_name = cgmodel.get_particle_name(angle[2])
            # unique_particle_names.index(particle_1_name)
            xml_type_1 = particle_1_name
            # unique_particle_names.index(particle_2_name)
            xml_type_2 = particle_2_name
            # unique_particle_names.index(particle_3_name)
            xml_type_3 = particle_3_name
            xml_object.write(
                '  <Angle angle="'
                + str(equil_bond_angle)
                + '" k="'
                + str(bond_angle_force_constant)
                + '" type1="'
                + str(xml_type_1)
                + '" type2="'
                + str(xml_type_2)
                + '" type3="'
                + str(xml_type_3)
                + '"/>\n'
            )
        xml_object.write(" </HarmonicAngleForce>\n")
    if cgmodel.include_torsion_forces:
        xml_object.write(' <PeriodicTorsionForce ordering="amber">\n')
        unique_torsion_list = []
        # print(cgmodel.torsion_list)
        for torsion in cgmodel.torsion_list:
            if any(torsion[i] < len(unique_particle_names) for i in range(4)):
                unique_torsion_list.append(torsion)
        for torsion in unique_torsion_list:
            torsion_force_constant = cgmodel.get_torsion_force_constant(
                [torsion[0], torsion[1], torsion[2], torsion[3]]
            )
            torsion_phase_angle = cgmodel.get_torsion_phase_angle(
                [torsion[0], torsion[1], torsion[2], torsion[3]]
            )
            particle_1_name = cgmodel.get_particle_name(torsion[0])
            particle_2_name = cgmodel.get_particle_name(torsion[1])
            particle_3_name = cgmodel.get_particle_name(torsion[2])
            particle_4_name = cgmodel.get_particle_name(torsion[3])
            # unique_particle_names.index(particle_1_name)
            xml_type_1 = particle_1_name
            # unique_particle_names.index(particle_2_name)
            xml_type_2 = particle_2_name
            # unique_particle_names.index(particle_3_name)
            xml_type_3 = particle_3_name
            # unique_particle_names.index(particle_4_name)
            xml_type_4 = particle_4_name
            periodicity = cgmodel.get_torsion_periodicity(torsion)
            xml_object.write(
                '  <Proper k1="'
                + str(torsion_force_constant)
                + '" periodicity1="'
                + str(periodicity)
                + '" phase1="'
                + str(torsion_phase_angle)
                + '" type1="'
                + str(xml_type_1)
                + '" type2="'
                + str(xml_type_2)
                + '" type3="'
                + str(xml_type_3)
                + '" type4="'
                + str(xml_type_4)
                + '"/>\n'
            )
        xml_object.write(" </PeriodicTorsionForce>\n")
    if cgmodel.include_nonbonded_forces:
        xml_object.write(' <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">\n')
        for particle_index in range(len(unique_particle_names)):
            charge = cgmodel.get_particle_charge(particle_index)._value
            sigma = cgmodel.get_particle_sigma(particle_index).in_units_of(unit.nanometer)._value
            epsilon = cgmodel.get_particle_epsilon(particle_index)._value
            particle_name = cgmodel.get_particle_name(particle_index)
            xml_object.write(f'  <Atom type=\"{particle_name}\" charge=\"{charge}\" sigma=\"{charge} epsilon=\"{epsilon}\"/>\n')
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

    # Check number of beads:
    nbeads_cgmodel = cgmodel.num_beads
    nbeads_topology = cgmodel.topology.getNumAtoms()
    
    if nbeads_cgmodel != nbeads_topology:
        print("ERROR: The number of particles in the coarse grained model")
        print("does not match the number of particles in the OpenMM topology.")
        print(f"There are {nbeads_cgmodel} particles in the coarse grained model")
        print(f"and {nbeads_topology} particles in the OpenMM topology.")
        exit()

    # Check number of residues:
    nres_cgmodel = cgmodel.polymer_length
    nres_topology = cgmodel.topology.getNumResidues()
    
    if nres_cgmodel != nres_topology:
        print("ERROR: The number of monomers in the coarse grained model")
        print("does not match the number of residues in the OpenMM topology.")
        print(f"There are {nres_cgmodel} monomers in the coarse grained model")
        print(f"and {nres_topology} monomers in the OpenMM topology.")
        exit()

    return


def build_topology(cgmodel, use_pdbfile=False, pdbfile=None):
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
    if cgmodel.constrain_bonds:
        # ***Why do we force this option? 
        use_pdbfile = True

    if use_pdbfile:
        if pdbfile is None:
            tf = tempfile.NamedTemporaryFile()
            write_pdbfile_without_topology(cgmodel, tf.name)
            pdb = PDBFile(tf.name)
            topology = pdb.getTopology()
            tf.close()
            return topology
        else:
            pdb = PDBFile(pdbfile)
            topology = pdb.getTopology()
            return topology

    topology = Topology()

    chain = topology.addChain()
    residue_index = -1
    openmm_particle_list = list()
    for particle in cgmodel.particle_list:
        if particle["monomer"] > residue_index:
            residue_index = particle["monomer"]
            residue = topology.addResidue(str(residue_index), chain)
        particle_symbol = particle["name"]
        element = elem.Element.getBySymbol(particle_symbol)
        openmm_particle = topology.addAtom(particle_symbol, element, residue)
        openmm_particle_list.append(particle)

    for bond in cgmodel.bond_list:
        topology.addBond(openmm_particle_list[bond[0]],openmm_particle_list[bond[1]])

    cgmodel.topology = topology
    verify_topology(cgmodel)
    
    return topology


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
    
    if cgmodel.include_bond_forces:
        total_forces += 1
    if cgmodel.include_nonbonded_forces:
        total_forces += 1
    if cgmodel.include_bond_angle_forces:
        total_forces += 1
    if cgmodel.include_torsion_forces:
        total_forces += 1
    if cgmodel.hbonds:
        total_forces += 1
        
    return total_forces


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

    # Check number of forces:
    n_forces_cgmodel = get_num_forces(cgmodel)
    n_forces_system = cgmodel.system.getNumForces()
    
    if n_forces_cgmodel != n_forces_system:
        print(f"ERROR: Mismatch in number of forces included in the cgmodel ({n_forces_cgmodel})")
        print(f"and number of forces in the OpenMM system ({n_forces_system})")
        exit()
    
    # Check number of particles:
    n_particles_cgmodel = cgmodel.num_beads
    n_particles_system = cgmodel.system.getNumParticles()

    if n_particles_cgmodel != n_particles_system:
        print(f"ERROR: Mismatch in number of particles in the cgmodel ({n_particles_cgmodel})")
        print(f"and number of particles in the OpenMM system ({n_particles_system})")
        exit()

    if cgmodel.constrain_bonds:
    
        # Check number of bond constraints:
        n_bonds = len(cgmodel.bond_list)
        n_constraints = cgmodel.system.getNumConstraints()
        
        if n_bonds != n_constraints:
            print("ERROR: Bond constraints were requested, but the")
            print(f"number of constraints in the OpenMM system ({n_constraints})")
            print(f"does not match the number of bonds in the cgmodel ({n_bonds})")
            exit()

    return


def check_force(cgmodel, force, force_type=None):
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
    >>> test_result = check_force(cgmodel,force,force_type="Nonbonded")

    """
    success = True
    
    if force_type == "Nonbonded":

        # TODO: Check the precision of the numpy data being used in the manual calculation. 
        # TODO: add check of the nonbonded energy for Rosetta functional form
        if cgmodel.rosetta_functional_form:
            return True

        total_nonbonded_energy = 0.0 * unit.kilojoule_per_mole
        
        repulsive_exp = cgmodel.nonbond_repulsive_exp
        attractive_exp = cgmodel.nonbond_attractive_exp
        
        for pair in cgmodel.nonbonded_interaction_list:
            if (pair not in cgmodel.nonbonded_exclusion_list and \
                reversed(pair) not in cgmodel.nonbonded_exclusion_list):

                particle_1_positions = cgmodel.positions[pair[0]]
                particle_2_positions = cgmodel.positions[pair[1]]
                
                sigma1 = cgmodel.get_particle_sigma(pair[0])
                sigma2 = cgmodel.get_particle_sigma(pair[1])
                
                sigma_ij = (sigma1+sigma2)/2
                
                epsilon1 = cgmodel.get_particle_epsilon(pair[0])
                epsilon2 = cgmodel.get_particle_epsilon(pair[1])
                
                if cgmodel.binary_interaction_parameters:
                    # Check if binary interaction parameters apply to this pair:
                    type1 = cgmodel.get_particle_type_name(pair[0])
                    type2 = cgmodel.get_particle_type_name(pair[1])
                        
                    kappa_name = f"{type1}_{type2}_binary_interaction"
                    kappa_name_reverse = f"{type2}_{type1}_binary_interaction"
                        
                    if type1 == type2:
                        # Same type particle interactions are not modified
                        kappa_ij = 0
                    
                    else:
                        if kappa_name in cgmodel.binary_interaction_parameters:
                            # Apply the binary interaction parameter
                            kappa_ij = cgmodel.binary_interaction_parameters[kappa_name]
                            
                        elif kappa_name_reverse in cgmodel.binary_interaction_parameters:
                            # Apply the binary interaction parameter
                            kappa_ij = cgmodel.binary_interaction_parameters[kappa_name_reverse]
                        else:
                            # If this pair type has no defined binary_interaction, use 0,
                            # which is the default.
                            
                            kappa_ij = 0
                    
                else:
                    kappa_ij = 0
                    
                epsilon_ij = (1-kappa_ij)*np.sqrt(epsilon1*epsilon2)    
                    
                    
                if cgmodel.go_model:    
                    if cgmodel.go_repulsive_epsilon is None or kappa_ij == 0:
                        # Native or no uniform repulsive strength specified - using mixing rule:
                        epsilon_ij_repulsive = np.sqrt(epsilon1*epsilon2)
                    else:
                        # Use uniform value for all pair types:
                        epsilon_ij_repulsive = cgmodel.go_repulsive_epsilon
                        
                    epsilon_ij_attractive = (1-kappa_ij)*np.sqrt(epsilon1*epsilon2)
                    int_energy = lj_go(
                        particle_1_positions, particle_2_positions, sigma_ij, epsilon_ij_repulsive,
                        epsilon_ij_attractive, r_exp=repulsive_exp, a_exp=attractive_exp
                        )
                else:
                    int_energy = lj_v(
                        particle_1_positions, particle_2_positions, sigma_ij, epsilon_ij,
                        r_exp=repulsive_exp, a_exp=attractive_exp
                        )
                total_nonbonded_energy += int_energy
        
        
        # Here we can do an energy decomposition by using force groups:
        
        # Set up test simulation object:
        simulation_time_step_test = 5.0 * unit.femtosecond
        friction_test = 0.0 / unit.picosecond
        integrator_test = LangevinIntegrator(
            0.0 * unit.kelvin, friction_test, simulation_time_step_test.in_units_of(unit.picosecond)
        )
        
        simulation_test = Simulation(cgmodel.topology, cgmodel.system, integrator_test)
        simulation_test.context.setPositions(cgmodel.positions)    

        # Set force groups:
        force_names = {}

        for force_index, force in enumerate(simulation_test.system.getForces()):
            # These are the overall classes of forces, not the particle-specific forces
            force_names[force_index] = force.__class__.__name__
            force.setForceGroup(force_index) 

        openmm_nonbonded_energy  = 0.0 * unit.kilojoule_per_mole
        
        for force_index, force in enumerate(simulation_test.system.getForces()):      
            if force.__class__.__name__ in ["NonbondedForce", "CustomNonbondedForce"]:
                openmm_nonbonded_energy += simulation_test.context.getState(getEnergy=True,groups={force_index}).getPotentialEnergy() 
            
        # Numpy absolute value gets rid of units - add them back
        energy_diff = np.abs(
            total_nonbonded_energy.value_in_unit(unit.kilojoule_per_mole) - 
            openmm_nonbonded_energy.value_in_unit(unit.kilojoule_per_mole)
            ) * unit.kilojoule_per_mole

        if energy_diff > 1E-1 * unit.kilojoule_per_mole:
            print("Error: The nonbonded potential energy computed by hand does not agree")
            print("with the value computed by OpenMM.")
            print(f"The value computed by OpenMM was: {openmm_nonbonded_energy}")
            print(f"The value computed by hand was: {total_nonbonded_energy}")
            print("Check the units for your model parameters.  If the problem persists, there")
            print("could be some other problem with the configuration of your coarse grained model.")
            success = False
        else:
            # The OpenMM nonbonded energy matches the energy computed manually"
            success = True

    return success


def add_rosetta_exception_parameters(cgmodel, nonbonded_force, particle_index_1, particle_index_2):
    """
    """
    exception_list = []
    for exception in range(nonbonded_force.getNumExceptions()):
        index_1, index_2, charge, sigma, epsilon = nonbonded_force.getExceptionParameters(
            exception
        )
        if [index_1, index_2] not in exception_list and [index_2, index_1] not in exception_list:
            exception_list.append([index_1, index_2])

    if (([particle_index_1, particle_index_2] not in exception_list) and
        ([particle_index_2, particle_index_1] not in exception_list)):
        
        charge_1 = cgmodel.get_particle_charge(particle_index_1)
        sigma_1 = cgmodel.get_particle_sigma(particle_index_1).in_units_of(unit.nanometer)
        epsilon_1 = cgmodel.get_particle_epsilon(particle_index_1).in_units_of(unit.kilojoule_per_mole)
        
        charge_2 = cgmodel.get_particle_charge(particle_index_2)
        sigma_2 = cgmodel.get_particle_sigma(particle_index_2).in_units_of(unit.nanometer)
        epsilon_2 = cgmodel.get_particle_epsilon(particle_index_2).in_units_of(unit.kilojoule_per_mole)
        
        sigma = (sigma_1 + sigma_2) / 2.0
        epsilon = 0.2 * unit.sqrt(epsilon_1 * epsilon_2)
        nonbonded_force.addException(
            particle_index_1, particle_index_2, 0.2 * charge_1 * charge_2, sigma, epsilon
        )
    return nonbonded_force


def add_force(cgmodel, force_type=None, rosetta_functional_form=False):
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
            bond_list.append([bond_indices[0], bond_indices[1]])
            if cgmodel.include_bond_forces:
                bond_force_constant = cgmodel.get_bond_force_constant(bond_indices)
                bond_length = cgmodel.get_bond_length(bond_indices)
                bond_force.addBond(
                    bond_indices[0],
                    bond_indices[1],
                    bond_length.value_in_unit(unit.nanometer),
                    bond_force_constant.value_in_unit(
                        unit.kilojoule_per_mole / unit.nanometer ** 2
                    ),
                )
            if cgmodel.constrain_bonds:
                bond_length = cgmodel.get_bond_length(bond_indices)
                if not cgmodel.include_bond_forces:
                    bond_force.addBond(
                        bond_indices[0],
                        bond_indices[1],
                        bond_length.value_in_unit(unit.nanometer),
                        0.0,
                    )
                cgmodel.system.addConstraint(bond_indices[0], bond_indices[1], bond_length)

        if len(bond_list) != bond_force.getNumBonds():
            print("ERROR: The number of bonds in the coarse grained model is different\n")
            print("from the number of bonds in its OpenMM System object\n")
            print("There are " + str(len(bond_list)) + " bonds in the coarse grained model\n")
            print("and " + str(bond_force.getNumBonds()) + " bonds in the OpenMM system object.")
            exit()

        cgmodel.system.addForce(bond_force)
        force = bond_force

    if force_type == "HBond":
        
        #####################################
        # Scheme A: Angles are sc-bb---bb
        #####################################
        
        # Here we define:
        # a1 = acceptor backbone
        # a2 = acceptor sidechain
        # d1 = donor backbone
        # d2 = donor sidechain        
        
        hbond_force = mm.CustomHbondForce(
           f"epsilon_hb*step(-abs(angle(d2,d1,a1)-theta_d)+pi/2)*step(-abs(angle(d2,a1,a2)-theta_a)+pi/2)*(5*(sigma_hb/distance(a1,d1))^12-6*(sigma_hb/distance(a1,d1))^10)*cos(angle(d2,d1,a1)-theta_d)^2*cos(angle(d1,a1,a2)-theta_a)^2"
           )

        #####################################
        # Scheme B: Angles are bb-bb-bb 
        #####################################
        
        # The center atom (d2/a2) is the h-bonding donor/acceptor           
        
        # hbond_force = mm.CustomHbondForce(
            # f"epsilon_hb*step(-abs(theta_d_shift)+pi/2)*step(-abs(theta_a_shift)+pi/2)*(5*(sigma_hb/dist_da)^12-6*(sigma_hb/dist_da)^10)*cos(theta_d_shift)^2*cos(theta_a_shift)^2; dist_da=distance(a2,d2); theta_d_shift=angle(d1,d2,d3)-theta_d; theta_a_shift=angle(a1,a2,a3)-theta_a"
            # )        
            
        #####################################
        # Scheme C: Angles are bb-bb---bb 
        #####################################
        
        # The donor group contains b_(i)*-b_(i+1), and the acceptor group contains b_(i-1)-b*
        # where the * indicates the particles whose hbond distance is measured and i is the
        # residue index 
        
        # a1 = acceptor bb_(i)*
        # a2 = acceptor bb_(i-1)
        # d1 = donor bb_(i)*
        # d2 = donor bb_(i+1)               
        
        # hbond_force = mm.CustomHbondForce(
            # f"epsilon_hb*step(-abs(theta_d_shift)+pi/2)*step(-abs(theta_a_shift)+pi/2)*(5*(sigma_hb/dist_da)^12-6*(sigma_hb/dist_da)^10)*cos(theta_d_shift)^2*cos(theta_a_shift)^2; dist_da=distance(a1,d1); theta_d_shift=angle(d2,d1,a1)-theta_d; theta_a_shift=angle(d1,a1,a2)-theta_a"
            # )   
        
        # Note: the step functions are 0 if step(x) < 0, and 1 otherwise
        # We can set pi as a global parameter here
        # For now the epsilon_hb and sigma_hb and global parameters for homopolymer helices.
        
        hbond_force.addGlobalParameter('pi',np.pi)
        hbond_force.addGlobalParameter('epsilon_hb',cgmodel.hbonds['epsilon_hb'].in_units_of(unit.kilojoules_per_mole))
        hbond_force.addGlobalParameter('sigma_hb',cgmodel.hbonds['sigma_hb'].in_units_of(unit.nanometer))
        hbond_force.addGlobalParameter('theta_d',cgmodel.hbonds['theta_d'].in_units_of(unit.radian))
        hbond_force.addGlobalParameter('theta_a',cgmodel.hbonds['theta_a'].in_units_of(unit.radian))
        
        # Get lists of donor and acceptor residues:
        donor_list = cgmodel.hbonds['donors']
        acceptor_list = cgmodel.hbonds['acceptors']
        
        # For homopolymers, polymers get built in the order of the particle sequence
        # specified in the monomer definition. For now, require that the particle sequence
        # in the monomers be [d1,d2] and equivalently [a1,a2].
        # For a scheme in which backbone-backbone contacts get the HBond interaction, this means [bb,sc]

        mono0 = cgmodel.get_particle_monomer_type(0)
        n_particles_per_mono = len(mono0["particle_sequence"])
        
        # Map the residue ids to the donor/acceptor ids
        donor_index_map = {}    
        acceptor_index_map = {}
        
        #####################################
        # Scheme A: Angles are sc-bb---bb
        #####################################        
        
        for donor in donor_list:
            d1 = donor*n_particles_per_mono    # Particle index of donor1 bead
            d2 = donor*n_particles_per_mono+1  # Particle index of donor2 bead
            donor_id = hbond_force.addDonor(d1,d2,-1)     # Third particle not used, so set to -1
            donor_index_map[donor] = donor_id
        
        for acceptor in acceptor_list:
            a1 = acceptor*n_particles_per_mono    # Particle index of acceptor1 bead
            a2 = acceptor*n_particles_per_mono+1  # Particle index of acceptor2 bead
            acceptor_id = hbond_force.addAcceptor(a1,a2,-1)     # Third particle not used, so set to -1  
            acceptor_index_map[acceptor] = acceptor_id
            
        #####################################
        # Scheme B: Angles are bb-bb-bb 
        #####################################  
            
        # for donor in donor_list:
            # d1 = (donor-1)*n_particles_per_mono  # Particle index of donor1 bead
            # d2 = donor*n_particles_per_mono      # Particle index of donor2 bead
            # d3 = (donor+1)*n_particles_per_mono  # Particle index of donor3 bead
            # donor_id = hbond_force.addDonor(d1,d2,d3)
            # donor_index_map[donor] = donor_id
        
        # for acceptor in acceptor_list:
            # a1 = (acceptor-1)*n_particles_per_mono  # Particle index of acceptor1 bead
            # a2 = acceptor*n_particles_per_mono      # Particle index of acceptor2 bead
            # a3 = (acceptor+1)*n_particles_per_mono  # Particle index of acceptor3 bead
            # acceptor_id = hbond_force.addAcceptor(a1,a2,a3)
            # acceptor_index_map[acceptor] = acceptor_id
            
        #####################################
        # Scheme C: Angles are bb-bb---bb 
        #####################################  
            
        # for donor in donor_list:
            # d1 = donor*n_particles_per_mono           # Particle index of donor1 bead
            # d2 = (donor+1)*n_particles_per_mono       # Particle index of donor2 bead
            # donor_id = hbond_force.addDonor(d1,d2,-1) # Third particle not used, so set to -1
            # donor_index_map[donor] = donor_id
        
        # for acceptor in acceptor_list:
            # a1 = acceptor*n_particles_per_mono              # Particle index of acceptor1 bead
            # a2 = (acceptor-1)*n_particles_per_mono          # Particle index of acceptor2 bead
            # acceptor_id = hbond_force.addAcceptor(a1,a2,-1) # Third particle not used, so set to -1
            # acceptor_index_map[acceptor] = acceptor_id

        # The hbond potential will get applied to all combinations of donor/acceptor pairs, before applying exclusions.
        all_hbond_pairs = []
        for donor in donor_list:
            for acceptor in acceptor_list:
                all_hbond_pairs.append([donor,acceptor])
        
        # Now make a list of the hbond pairs to include, formatted as [donor,acceptor]:
        included_hbond_pairs = []             
        for i in range(len(donor_list)):
            included_hbond_pairs.append([donor_list[i],acceptor_list[i]])
                
        # Now apply exclusions:        
        excluded_hbond_pairs = []
        # Exclude all hbond pairs not in included_hbond_pairs:  
        # This does not work because each donor can have a maximimum of 4 exclusions
        # for pair in all_hbond_pairs:
            # if pair not in included_hbond_pairs:
                #excluded_hbond_pairs.append(pair)
                #hbond_force.addExclusion(donor_index_map[pair[0]],acceptor_index_map[pair[1]])      
              
        # We can use these 4 exclusions to exclude any self interaction and 1-2 neighbors which will cause instability:      
        # Note - this code is specific to 1-1 models.
        for pair in all_hbond_pairs:
            if np.abs(pair[0]-pair[1]) < 2:
                # If 1-2 or 1-3 neighbors (by residue), exclude:
                excluded_hbond_pairs.append(pair)
                hbond_force.addExclusion(donor_index_map[pair[0]],acceptor_index_map[pair[1]])      
              
        num_exclusions_openmm = hbond_force.getNumExclusions()
        num_donors_openmm = hbond_force.getNumDonors()
        num_acceptors_openmm = hbond_force.getNumAcceptors()
        print(f'num_exclusions_openmm: {num_exclusions_openmm}')
        print(f'num_donors_openmm: {num_donors_openmm}')
        print(f'num_acceptors_openmm: {num_acceptors_openmm}')
        print(f'hbond_exclusions: {excluded_hbond_pairs}')
        
        cgmodel.system.addForce(hbond_force)
        force = hbond_force                

    if force_type == "Nonbonded":

        if cgmodel.nonbond_repulsive_exp != 12 or cgmodel.nonbond_attractive_exp != 6:
            
            # Use a custom Mie potential instead of the standard LJ 12-6
            n = cgmodel.nonbond_repulsive_exp
            m = cgmodel.nonbond_attractive_exp
            
            # Check feasibility of Mie exponents:
            if m >= n:
                print(f'Error: invalid Mie potential exponents')
                exit()
            
            # Use custom nonbonded force with or without binary interaction parameter
            if cgmodel.binary_interaction_parameters:
                # Binary interaction paramters with n-m Mie potential:
                # If not an empty dictionary, use the parameters within
                
                print(f'Mie potentials with binary interaction parameters not yet implemented')
                exit()

            else:
                # Mie potential with standard mixing rules
                nonbonded_force = mm.CustomNonbondedForce(f"(n/(n-m))*(n/m)^(m/(n-m))*epsilon*((sigma/r)^n-(sigma/r)^m); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)")
            
                nonbonded_force.addPerParticleParameter("sigma")
                nonbonded_force.addPerParticleParameter("epsilon")           
                
                # Here we can potentially also add mixing rules for the exponents,
                # in which case they would be per-particle parameters defined in the particle dictionaries
                # For now, the exponents are global parameters:
                
                nonbonded_force.addGlobalParameter("n",n)
                nonbonded_force.addGlobalParameter("m",m)
                
                # TODO: add the rosetta_functional_form switching function
                nonbonded_force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
                
                for particle in range(cgmodel.num_beads):
                    # We don't need to define charge here, though we should add it in the future
                    # We also don't need to define kappa since it is a global parameter
                    # Each nonbonded force must have the same number of particles as the system.
                    # So in each nonbonded force, we are setting the non-interaction group interactions
                    # to zero. 
                    sigma = cgmodel.get_particle_sigma(particle)
                    epsilon = cgmodel.get_particle_epsilon(particle)
                    nonbonded_force.addParticle((sigma, epsilon))   

                # Add nonbonded exclusions:
                for pair in cgmodel.get_nonbonded_exclusion_list():
                    nonbonded_force.addExclusion(pair[0],pair[1])
                    
                cgmodel.system.addForce(nonbonded_force)
                force = nonbonded_force                
            
        else:     
            if not cgmodel.go_model:
                # Standard LJ 12-6 potential
                nonbonded_force = mm.NonbondedForce()

                if rosetta_functional_form:
                    # rosetta has a 4.5-6 A vdw cutoff.  Note the OpenMM cutoff may not be quite the same
                    # functional form as the Rosetta cutoff, but it should be somewhat close.
                    nonbonded_force.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
                    nonbonded_force.setCutoffDistance(0.6)  # rosetta cutoff distance in nm
                    nonbonded_force.setUseSwitchingFunction(True)
                    nonbonded_force.setSwitchingDistance(0.45)  # start of rosetta switching distance in nm
                else:
                    nonbonded_force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)

                for particle in range(cgmodel.num_beads):
                    charge = cgmodel.get_particle_charge(particle)
                    sigma = cgmodel.get_particle_sigma(particle)
                    epsilon = cgmodel.get_particle_epsilon(particle)
                    nonbonded_force.addParticle(charge, sigma, epsilon)

                # Add nonbonded exclusions:
                nonbonded_exclusion_list = cgmodel.get_nonbonded_exclusion_list()
                for pair in nonbonded_exclusion_list:
                    nonbonded_force.addException(pair[0],pair[1],0,0,0)
                   
                # Check for binary interaction parameters:
                if cgmodel.binary_interaction_parameters:
                    # If not an empty dict, use the input binary_interaction_parameters
                    # The cross epsilons are rescaled by (1-kappa) with addException
                    
                    for pair in cgmodel.nonbonded_interaction_list:
                        if pair not in nonbonded_exclusion_list and reversed(pair) not in nonbonded_exclusion_list:
                            # If already excluded, no further action needed
                            
                            pair_type0 = cgmodel.get_particle_type_name(pair[0])
                            pair_type1 = cgmodel.get_particle_type_name(pair[1])
                            
                            kappa_name = f"{pair_type0}_{pair_type1}_binary_interaction"
                            kappa_name_reverse = f"{pair_type1}_{pair_type0}_binary_interaction"
                            
                            if pair_type0 == pair_type1:
                                # Same type particle interactions should not be modified
                                kappa = 0
                            
                            else:
                                if kappa_name in cgmodel.binary_interaction_parameters:
                                    # Apply the binary interaction parameter to this interaction group
                                    kappa = cgmodel.binary_interaction_parameters[kappa_name]
                                    
                                elif kappa_name_reverse in cgmodel.binary_interaction_parameters:
                                    # Apply the binary interaction parameter to this interaction group
                                    kappa = cgmodel.binary_interaction_parameters[kappa_name_reverse]
                                    
                                else:
                                    # The binary interaction parameter is not set for this interaction
                                    # group. By default, we will use a value of 0 (standard mixing rules)
                                    
                                    print(f'Warning: no binary interaction parameter set for pair type {pair_type0}, {pair_type1}')
                                    print(f'Applying default value of zero (standard mixing rules)')
                                    
                                    kappa = 0

                            charge1 = cgmodel.get_particle_charge(pair[0])
                            sigma1 = cgmodel.get_particle_sigma(pair[0]).in_units_of(unit.nanometer)
                            epsilon1 = cgmodel.get_particle_epsilon(pair[0]).in_units_of(unit.kilojoule_per_mole)
                            
                            charge2 = cgmodel.get_particle_charge(pair[1])
                            sigma2 = cgmodel.get_particle_sigma(pair[1]).in_units_of(unit.nanometer)
                            epsilon2 = cgmodel.get_particle_epsilon(pair[1]).in_units_of(unit.kilojoule_per_mole)
                            
                            charge_ij = charge1*charge2
                            sigma_ij = (sigma1+sigma2) / 2.0
                            epsilon_ij = (1-kappa)*np.sqrt(epsilon1*epsilon2)
                            nonbonded_force.addException(
                                pair[0], pair[1], charge_ij, sigma_ij, epsilon_ij
                            )
                                
                # For rosetta, apply a 0.2 weight to 1-5 interactions:
                # This should not be used with the binary interaction parameters.
                if rosetta_functional_form:
                    for torsion in cgmodel.torsion_list:
                        for bond in cgmodel.bond_list:
                            if bond[0] not in torsion:
                                if bond[1] == torsion[0]:
                                    nonbonded_force = add_rosetta_exception_parameters(
                                        cgmodel, nonbonded_force, bond[0], torsion[3]
                                    )
                                if bond[1] == torsion[3]:
                                    nonbonded_force = add_rosetta_exception_parameters(
                                        cgmodel, nonbonded_force, bond[0], torsion[0]
                                    )
                            if bond[1] not in torsion:
                                if bond[0] == torsion[0]:
                                    nonbonded_force = add_rosetta_exception_parameters(
                                        cgmodel, nonbonded_force, bond[1], torsion[3]
                                    )
                                if bond[0] == torsion[3]:
                                    nonbonded_force = add_rosetta_exception_parameters(
                                        cgmodel, nonbonded_force, bond[1], torsion[0]
                                    )

                cgmodel.system.addForce(nonbonded_force)
                force = nonbonded_force
            elif cgmodel.go_model: 
                # Go 12-6 LJ potential

                # We can only set kappa as a global parameter, or as a per-particle parameter.
                # One way is to use a separate nonbonded force object for each kappa.
                
                # First, construct a dictionary mapping kappa to the pairs it should be applied to:
                kappa_map = {}
                
                for pair in cgmodel.nonbonded_interaction_list:
                    if pair not in cgmodel.nonbonded_exclusion_list and reversed(pair) not in cgmodel.nonbonded_exclusion_list:
                        # If already excluded, no further action needed
                        
                        pair_type0 = cgmodel.get_particle_type_name(pair[0])
                        pair_type1 = cgmodel.get_particle_type_name(pair[1])
                        
                        kappa_name = f"{pair_type0}_{pair_type1}_binary_interaction"
                        kappa_name_reverse = f"{pair_type1}_{pair_type0}_binary_interaction"
                        
                        if pair_type0 == pair_type1:
                            # Same type particle interactions should not be modified
                            kappa_pair = 0
                        
                        else:
                            if kappa_name in cgmodel.binary_interaction_parameters:
                                kappa_pair = cgmodel.binary_interaction_parameters[kappa_name]

                            elif kappa_name_reverse in cgmodel.binary_interaction_parameters:
                                kappa_pair = cgmodel.binary_interaction_parameters[kappa_name_reverse]
                                
                            else:
                                # The binary interaction parameter is not set for this pair.
                                # By default, we will use a value of 0 (standard mixing rules)
                                
                                print(f'Warning: no binary interaction parameter set for pair type {pair_type0}, {pair_type1}')
                                print(f'Applying default value of zero (standard mixing rules)')
                                
                                kappa_pair = 0

                    # Add pair to dict:
                    if kappa_pair in kappa_map:
                        kappa_map[kappa_pair].append(pair)
                    else:
                        kappa_map[kappa_pair] = []
                        kappa_map[kappa_pair].append(pair)
                            
                
                # Now, create a customNonbondedForce for each unique kappa:
                force = []
                for kappa, pair_list in kappa_map.items():
                    particles_added = []
                    
                    # Full repulsive potential with scaled attractive potential
                    
                    if kappa == 0 or cgmodel.go_repulsive_epsilon is None:
                        # This is either a native interaction, or we use mixing rules for the repulsive epsilon
                        nonbonded_force = mm.CustomNonbondedForce(f"4*epsilon_rep*(sigma/r)^12-4*epsilon_att*(sigma/r)^6; sigma=0.5*(sigma1+sigma2); epsilon_rep=sqrt(epsilon1*epsilon2); epsilon_att=(1-{kappa})*sqrt(epsilon1*epsilon2)")
                    
                    else:
                        # This is a non-native interaction and we use a uniform repulsive epsilon
                        nonbonded_force = mm.CustomNonbondedForce(f"4*epsilon_rep_uni*(sigma/r)^12-4*epsilon_att*(sigma/r)^6; sigma=0.5*(sigma1+sigma2); epsilon_att=(1-{kappa})*sqrt(epsilon1*epsilon2)")
                        nonbonded_force.addGlobalParameter("epsilon_rep_uni",cgmodel.go_repulsive_epsilon.in_units_of(unit.kilojoule_per_mole))
                    
                    nonbonded_force.addPerParticleParameter("sigma")
                    nonbonded_force.addPerParticleParameter("epsilon")   
                    
                    # TODO: add the rosetta_functional_form switching function
                    nonbonded_force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
                    
                    for particle in range(cgmodel.num_beads):
                        # We don't need to define charge here, though we should add it in the future
                        # We also don't need to define kappa since it is a global parameter
                        # Each nonbonded force must have the same number of particles as the system,
                        # so we add all particles here:
                        
                        sigma = cgmodel.get_particle_sigma(particle)
                        epsilon = cgmodel.get_particle_epsilon(particle)
                        nonbonded_force.addParticle((sigma, epsilon))

                    # We can't have different numbers of exclusions for each CustomNonbondedForce
                    # Instead add pairs as interaction groups:
                    for pair in pair_list:
                        nonbonded_force.addInteractionGroup([pair[0]],[pair[1]])

                    # Exclude pairs in the nonbonded exclusions list (commmon to all kappa):        
                    for pair in cgmodel.nonbonded_exclusion_list:
                        nonbonded_force.addExclusion(pair[0],pair[1])

                    cgmodel.system.addForce(nonbonded_force)
                    force.append(nonbonded_force)
                        

    if force_type == "Angle":

        if cgmodel.angle_style == 'harmonic':
            # Use standard harmonic angle potential
            angle_force = mm.HarmonicAngleForce()
            for angle in cgmodel.bond_angle_list:
                bond_angle_force_constant = cgmodel.get_bond_angle_force_constant(angle)
                equil_bond_angle = cgmodel.get_equil_bond_angle(angle)
                angle_force.addAngle(
                    angle[0],
                    angle[1],
                    angle[2],
                    equil_bond_angle.value_in_unit(unit.radian),
                    bond_angle_force_constant.value_in_unit(
                        unit.kilojoule_per_mole / unit.radian ** 2
                    ),
                )
            cgmodel.system.addForce(angle_force)
            force = angle_force

        elif cgmodel.angle_style == 'restricted':
            # Use restricted angle bending potential as CustomAngleForce
            angle_force = mm.CustomAngleForce("0.5*k*(cos(theta)-cos(theta_0))^2/(sin(theta)^2)")
            angle_force.addPerAngleParameter("k")
            angle_force.addPerAngleParameter("theta_0")
            
            for angle in cgmodel.bond_angle_list:
                bond_angle_force_constant = cgmodel.get_bond_angle_force_constant(angle)
                equil_bond_angle = cgmodel.get_equil_bond_angle(angle)
                angle_force.addAngle(
                    angle[0],
                    angle[1],
                    angle[2],
                    [bond_angle_force_constant.value_in_unit(unit.kilojoule_per_mole),
                    equil_bond_angle.value_in_unit(unit.radian)],
                )
            cgmodel.system.addForce(angle_force)
            force = angle_force
            
        elif cgmodel.angle_style == 'cosine':
            # Use cosine angle bending potential as CustomAngleForce
            angle_force = mm.CustomAngleForce("0.5*k*(1-cos(theta-theta_0))")
            angle_force.addPerAngleParameter("k")
            angle_force.addPerAngleParameter("theta_0")
            
            for angle in cgmodel.bond_angle_list:
                bond_angle_force_constant = cgmodel.get_bond_angle_force_constant(angle)
                equil_bond_angle = cgmodel.get_equil_bond_angle(angle)
                angle_force.addAngle(
                    angle[0],
                    angle[1],
                    angle[2],
                    [bond_angle_force_constant.value_in_unit(unit.kilojoule_per_mole),
                    equil_bond_angle.value_in_unit(unit.radian)],
                )
            cgmodel.system.addForce(angle_force)
            force = angle_force
            
        else:
            print(f'Error: unknown angle style {cgmodel.angle_style}')
            exit()


    if force_type == "Torsion":
        torsion_force = mm.PeriodicTorsionForce()
        for torsion in cgmodel.torsion_list:
            torsion_force_constant = cgmodel.get_torsion_force_constant(torsion)
            torsion_phase_angle = cgmodel.get_torsion_phase_angle(torsion)
            periodicity = cgmodel.get_torsion_periodicity(torsion)
            
            if type(periodicity) == list:
                # Check periodic torsion parameter lists:
                # These can be either a list of quantities, or a quantity with a list as its value
                
                # Check torsion_phase_angle parameters:
                if type(torsion_phase_angle) == unit.quantity.Quantity:
                    # This is either a single quantity, or quantity with a list value
                    if type(torsion_phase_angle.value_in_unit(unit.radian)) == list:
                        # Check if there are either 1 or len(periodicity) elements
                        if len(torsion_phase_angle) != len(periodicity) and len(torsion_phase_angle) != 1:
                            # Mismatch is list lengths
                            print('ERROR: incompatible periodic torsion parameter lists')
                            exit()
                        if len(torsion_phase_angle) == 1:
                            # This happens when input is '[value]*unit.radian'
                            torsion_phase_angle_list = []
                            for i in range(len(periodicity)):
                                torsion_phase_angle_list.append(torsion_phase_angle[0])
                            # This is a list of quantities
                            torsion_phase_angle = torsion_phase_angle_list                            
                    else:
                        # Single quantity - apply same angle to all periodic terms:
                        torsion_phase_angle_list = []
                        for i in range(len(periodicity)):
                            torsion_phase_angle_list.append(torsion_phase_angle)
                        # This is a list of quantities
                        torsion_phase_angle = torsion_phase_angle_list
                else:
                    # This is a list of quantities or incorrect input
                    if len(torsion_phase_angle) == 1:
                        # This is a list containing a single quantity
                        torsion_phase_angle_list = []
                        for i in range(len(periodicity)):
                            torsion_phase_angle_list.append(torsion_phase_angle[0])
                        # This is a list of quantities
                        torsion_phase_angle = torsion_phase_angle_list  

                # Check torsion_force_constant parameters:
                if type(torsion_force_constant) == unit.quantity.Quantity:
                    # This is either a single quantity, or quantity with a list value
                    if type(torsion_force_constant.value_in_unit(unit.kilojoule_per_mole)) == list:
                        # Check if there are either 1 or len(periodicity) elements
                        if len(torsion_force_constant) != len(periodicity) and len(torsion_force_constant) != 1:
                            # Mismatch is list lengths
                            print('ERROR: incompatible periodic torsion parameter lists')
                            exit()
                        if len(torsion_force_constant) == 1:
                            # This happens when input is '[value]*unit.kilojoule_per_mole'
                            torsion_force_constant_list = []
                            for i in range(len(periodicity)):
                                torsion_force_constant_list.append(torsion_force_constant[0])
                            # This is a list of quantities
                            torsion_force_constant = torsion_force_constant_list      
                    else:
                        # Single quantity - apply same angle to all periodic terms:
                        torsion_force_constant_list = []
                        for i in range(len(periodicity)):
                            torsion_force_constant_list.append(torsion_force_constant)
                        # This is a list of quantities
                        torsion_force_constant = torsion_force_constant_list
                else:
                    # This is a list of quantities or incorrect input
                    if len(torsion_force_constant) == 1:
                        # This is a list containing a single quantity
                        torsion_force_constant_list = []
                        for i in range(len(periodicity)):
                            torsion_force_constant_list.append(torsion_force_constant[0])
                        # This is a list of quantities
                        torsion_force_constant = torsion_force_constant_list  
                
                # Add torsion force:
                for i in range(len(periodicity)):
                    # print(f'Adding torsion term to particles [{torsion[0]} {torsion[1]} {torsion[2]} {torsion[3]}]')
                    # print(f'periodicity: {periodicity[i]}')
                    # print(f'torsion_phase_angle: {torsion_phase_angle[i]}')
                    # print(f'torsion_force_constant: {torsion_force_constant[i]}\n')
                    torsion_force.addTorsion(
                        torsion[0],
                        torsion[1],
                        torsion[2],
                        torsion[3],
                        periodicity[i],
                        torsion_phase_angle[i].value_in_unit(unit.radian),
                        torsion_force_constant[i].value_in_unit(unit.kilojoule_per_mole),
                    )
                    
            else:
                # Single periodic torsion term:
                torsion_force.addTorsion(
                    torsion[0],
                    torsion[1],
                    torsion[2],
                    torsion[3],
                    periodicity,
                    torsion_phase_angle.value_in_unit(unit.radian),
                    torsion_force_constant.value_in_unit(unit.kilojoule_per_mole),
                )                
                
        cgmodel.system.addForce(torsion_force)
        
        force = torsion_force

    return (cgmodel, force)


def check_forces(cgmodel):
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
    >>> pass_forces_test = check_forces(cgmodel)

    """
    if cgmodel.topology is None:
        cgmodel.topology = build_topology(cgmodel)
        
    simulation = build_mm_simulation(
        cgmodel.topology,
        cgmodel.system,
        cgmodel.positions,
        simulation_time_step=5.0*unit.femtosecond,
        print_frequency=1,
    )
    
    forces = simulation.context.getState(getForces=True).getForces()
    success = True
    
    for force in forces:
        for component in force:
            if "nan" in str(component):
                print("Detected 'nan' force value")
                print(f"for particle {forces.index(force)}")
                success = False
                return success
                
            if component > 9.9E9 * component.unit:
                print("Detected unusually large forces")
                print(f"for particle {forces.index(force)}")
                print(f"The force is: {component:.2e}")
                success = False
                return success
                
    return success


def build_system(cgmodel, rosetta_functional_form=False, verify=True):
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
    for particle in cgmodel.particle_list:
        system.addParticle(particle["type"]["mass"])
    cgmodel.system = system

    if cgmodel.include_nonbonded_forces:
        # Create nonbonded forces
        cgmodel, nonbonded_force = add_force(
            cgmodel, force_type="Nonbonded", rosetta_functional_form=rosetta_functional_form,
        )
        if verify and cgmodel.positions is not None:
            if not check_force(cgmodel, nonbonded_force, force_type="Nonbonded"):
                print("ERROR: There was a problem with the nonbonded force definitions")
                exit()

    if cgmodel.hbonds:
        # Create directional hbond forces
        cgmodel, hbond_force = add_force(cgmodel, force_type="HBond")
        
        if verify and cgmodel.positions is not None:
            # ***TODO: add the force check here (manual vs openmm):
            if not check_force(cgmodel, hbond_force, force_type="HBond"):
                print("ERROR: There was a problem with the nonbonded force definitions")
                exit()

    if cgmodel.include_bond_forces or cgmodel.constrain_bonds:
        if len(cgmodel.bond_list) > 0:
            # Create bond (harmonic) potentials
            cgmodel, bond_force = add_force(cgmodel, force_type="Bond")
            
            if verify and cgmodel.positions is not None:
            # ***This check is currently doing nothing:
                if not check_force(cgmodel, bond_force, force_type="Bond"):
                    print("ERROR: The bond force definition is giving 'nan'")
                    exit()

    if cgmodel.include_bond_angle_forces:
        if len(cgmodel.bond_angle_list) > 0:
            # Create bond angle potentials
            cgmodel, bond_angle_force = add_force(cgmodel, force_type="Angle")

            if verify and cgmodel.positions is not None:
            # ***This check is currently doing nothing:
                if not check_force(cgmodel, bond_angle_force, force_type="Angle"):
                    print("ERROR: There was a problem with the bond angle force definitions.")
                    exit()

    if cgmodel.include_torsion_forces:
        if len(cgmodel.torsion_list) > 0:
            # Create torsion potentials
            cgmodel, torsion_force = add_force(cgmodel, force_type="Torsion")

            if verify and cgmodel.positions is not None:
            # ***This check is currently doing nothing:
                if not check_force(cgmodel, torsion_force, force_type="Torsion"):
                    print("ERROR: There was a problem with the torsion definitions.")
                    exit()

    if verify and cgmodel.positions is not None:
        if not check_forces(cgmodel):
            print("ERROR: There was a problem with the forces.")
            exit()

    return system
