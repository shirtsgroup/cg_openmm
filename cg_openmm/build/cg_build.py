import os
import datetime
import tempfile
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

    if cgmodel.num_beads != cgmodel.topology.getNumAtoms():
        print("ERROR: The number of particles in the coarse grained model\n")
        print("does not match the number of particles in the OpenMM topology.\n")
        print("There are " + str(cgmodel.num_beads) + " particles in the coarse grained model\n")
        print("and " + str(cgmodel.topology.getNumAtoms()) + " particles in the OpenMM topology.")
        exit()

    if cgmodel.polymer_length != cgmodel.topology.getNumResidues():
        print("ERROR: The number of monomers in the coarse grained model\n")
        print("does not match the number of residues in the OpenMM topology.\n")
        print(
            "There are " + str(cgmodel.polymer_length) + " monomers in the coarse grained model\n"
        )
        print(
            "and " + str(cgmodel.topology.getNumResidues()) + " monomers in the OpenMM topology."
        )
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

    if cgmodel.include_bond_forces or cgmodel.constrain_bonds:
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
        total_forces = total_forces + 1
    if cgmodel.include_nonbonded_forces:
        total_forces = total_forces + 1
    if cgmodel.include_bond_angle_forces:
        total_forces = total_forces + 1
    if cgmodel.include_torsion_forces:
        total_forces = total_forces + 1
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

    if get_num_forces(cgmodel) != cgmodel.system.getNumForces():
        print("ERROR: the number of forces included in the coarse grained model\n")
        print("does not match the number of forces in the OpenMM system object.\n")
        print(
            " There are " + str(get_num_forces(cgmodel)) + " forces in the coarse grained model\n"
        )
        print("and " + str(cgmodel.system.getNumForces()) + " forces in the OpenMM System().")
        exit()

    if cgmodel.num_beads != cgmodel.system.getNumParticles():
        print("ERROR: The number of particles in the coarse grained model\n")
        print("does not match the number of particles in the OpenMM system.\n")
        print("There are " + str(cgmodel.num_beads) + " particles in the coarse grained model\n")
        print("and " + str(cgmodel.ssytem.getNumParticles()) + " particles in the OpenMM system.")
        exit()

    if cgmodel.constrain_bonds:
        if len(cgmodel.bond_list) != cgmodel.system.getNumConstraints():
            print("ERROR: Bond constraints were requested, but the\n")
            print("number of constraints in the coarse grained model\n")
            print("does not match the number of constraintes in the OpenMM system.\n")
            print(
                "There are "
                + str(cgmodel.bond_list)
                + " bond constraints in the coarse grained model\n"
            )
            print(
                "and "
                + str(cgmodel.system.getNumConstraints())
                + " constraints in the OpenMM system."
            )
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
        if cgmodel.num_beads != force.getNumParticles():
            print("ERROR: The number of particles in the coarse grained model is different")
            print(
                "from the number of particles with nonbonded force definitions in the OpenMM NonbondedForce.\n"
            )
            print("There are " + str(cgmodel.num_beads) + " particles in the coarse grained model")
            print(
                "and " + str(force.getNumParticles()) + " particles in the OpenMM NonbondedForce."
            )
            success = False

        total_nonbonded_energy = 0.0 * unit.kilojoule_per_mole
        # print(cgmodel.nonbonded_interaction_list)
        for nonbonded_interaction in cgmodel.nonbonded_interaction_list:
            particle_1_positions = cgmodel.positions[nonbonded_interaction[0]]
            particle_2_positions = cgmodel.positions[nonbonded_interaction[1]]
            sigma = cgmodel.get_particle_sigma(nonbonded_interaction[0])
            epsilon = cgmodel.get_particle_epsilon(nonbonded_interaction[0])
            int_energy = lj_v(particle_1_positions, particle_2_positions, sigma, epsilon)
            total_nonbonded_energy = total_nonbonded_energy.__add__(int_energy)

        cgmodel.include_bond_forces = False
        cgmodel.include_bond_angle_forces = False
        cgmodel.include_torsion_forces = False
        cgmodel.topology = build_topology(cgmodel)
        cgmodel.simulation = build_mm_simulation(
            cgmodel.topology,
            cgmodel.system,
            cgmodel.positions,
            simulation_time_step=5.0 * unit.femtosecond,
            print_frequency=1,
        )
        potential_energy = cgmodel.simulation.context.getState(getEnergy=True).getPotentialEnergy()

        # if potential_energy.__sub__(total_nonbonded_energy).__gt__(0.1 * unit.kilojoule_per_mole):
        # print("Warning: The nonbonded potential energy computed by hand does not agree")
        # print("with the value computed by OpenMM.")
        # print("The value computed by OpenMM was: "+str(potential_energy))
        # print("The value computed by hand was: "+str(total_nonbonded_energy))
        # print("Check the units for your model parameters.  If the problem persists, there")
        # print("could be some other problem with the configuration of your coarse grained model.")
        # success = False
        # else:
        # print("The OpenMM nonbonded energy matches the energy computed by hand:")
        # print(str(potential_energy))

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

    if [particle_index_1, particle_index_2] not in exception_list and [
        particle_index_2,
        particle_index_1,
    ] not in exception_list:
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

    if force_type == "Nonbonded":

        if cgmodel.binary_interaction_parameters:
            # If not an empty dictionary, use the parameters within
            
            for key, value in cgmodel.binary_interaction_parameters.items():
                # TODO: make kappa work for systems with more than 2 bead types
                kappa = value
            
            # Use custom nonbonded force with binary interaction parameter
            nonbonded_force = mm.CustomNonbondedForce(f"4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=(1-kappa)*sqrt(epsilon1*epsilon2)")
            nonbonded_force.addPerParticleParameter("sigma")
            nonbonded_force.addPerParticleParameter("epsilon")           

            # We need to specify a default value of kappa when adding global parameter
            nonbonded_force.addGlobalParameter("kappa",kappa)
            
            # TODO: add the rosetta_function_form switching function
            nonbonded_force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
            
            for particle in range(cgmodel.num_beads):
                # We don't need to define charge here, though we should add it in the future
                # We also don't need to define kappa since it is a global parameter
                sigma = cgmodel.get_particle_sigma(particle)
                epsilon = cgmodel.get_particle_epsilon(particle)
                nonbonded_force.addParticle((sigma, epsilon))   

            if len(cgmodel.bond_list) >= 1:
                #***Note: customnonbonded force uses 'Exclusion' rather than 'Exception'
                # Each of these also takes different arguments
                if not rosetta_functional_form:
                    # This should not be applied if there are no angle forces.
                    if cgmodel.include_bond_angle_forces:
                        bond_cut = 2 # Particles separated by this many bonds or fewer are excluded
                        # A value of 2 means that 1-2, 1-3 interactions are 0, 1-4 interactions are 1
                        nonbonded_force.createExclusionsFromBonds(cgmodel.bond_list, bond_cut)
                    else:
                        # Just remove the 1-2 nonbonded interactions.
                        # For customNonbondedForce, don't need to set charge product and epsilon here
                        for bond in cgmodel.bond_list:
                            nonbonded_force.addExclusion(bond[0], bond[1])
            
        else:
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

            if len(cgmodel.bond_list) >= 1:
                if not rosetta_functional_form:
                    # This should not be applied if there are no angle forces.
                    if cgmodel.include_bond_angle_forces:
                        nonbonded_force.createExceptionsFromBonds(cgmodel.bond_list, 1.0, 1.0)
                    else:
                        # Just remove the 1-2 nonbonded interactions.
                        # If charge product and epsilon are 0, the interaction is omitted.
                        for bond in cgmodel.bond_list:
                            nonbonded_force.addException(bond[0], bond[1], 0.0, 1.0, 0.0)
                if rosetta_functional_form:
                    # Remove i+3 interactions
                    nonbonded_force.createExceptionsFromBonds(cgmodel.bond_list, 0.0, 0.0)
                    # Reduce the strength of i+4 interactions
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

    if force_type == "Angle":
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
        
        # print(f"Number of torsion forces: {cgmodel.system.getForces()[3].getNumTorsions()}")
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
        simulation_time_step=5.0 * unit.femtosecond,
        print_frequency=1,
    )
    forces = simulation.context.getState(getForces=True).getForces()
    success = True
    for force in forces:
        for component in force:
            if "nan" in str(component):
                print("Detected 'nan' force value")
                print("for particle " + str(forces.index(force)))
                success = False
                return success
            if component.__gt__(9.9e9 * component.unit):
                print("Detected unusually large forces")
                print("for particle " + str(forces.index(force)))
                print(
                    "The force is: "
                    + str("{:.2e}".format(component._value))
                    + " "
                    + str(component.unit)
                )
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

    if cgmodel.include_bond_forces or cgmodel.constrain_bonds:
        if len(cgmodel.bond_list) > 0:
            # Create bond (harmonic) potentials
            cgmodel, bond_force = add_force(cgmodel, force_type="Bond")
            if cgmodel.positions is not None:
                if not check_force(cgmodel, bond_force, force_type="Bond"):
                    print("ERROR: The bond force definition is giving 'nan'")
                    exit()

    if cgmodel.include_bond_angle_forces:
        if len(cgmodel.bond_angle_list) > 0:
            # Create bond angle potentials
            cgmodel, bond_angle_force = add_force(cgmodel, force_type="Angle")
            if cgmodel.positions is not None:
                if not check_force(cgmodel, bond_angle_force, force_type="Angle"):
                    print("ERROR: There was a problem with the bond angle force definitions.")
                    exit()

    if cgmodel.include_torsion_forces:
        if len(cgmodel.torsion_list) > 0:
            # Create torsion potentials
            cgmodel, torsion_force = add_force(cgmodel, force_type="Torsion")
            if cgmodel.positions is not None:
                if not check_force(cgmodel, torsion_force, force_type="Torsion"):
                    print("ERROR: There was a problem with the torsion definitions.")
                    exit()

    if verify:
        if cgmodel.positions is not None:
            if not check_forces(cgmodel):
                print("ERROR: There was a problem with the forces.")
                exit()

    return system
