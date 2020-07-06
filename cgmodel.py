import simtk.unit as unit
import sys, os
from collections import Counter
from foldamers.utilities.util import get_random_positions
from cg_openmm.build.cg_build import *
from cg_openmm.utilities.iotools import *
from itertools import chain, combinations, product
import pickle


class CGModel(object):
    """

        Build a coarse-grained model class object.

        :Example:

        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> cgmodel = CGModel()

        :Example:

        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> from simtk import unit
        >>> bond_length = 7.5 * unit.angstrom
        >>> bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
        >>> constrain_bonds = False
        >>> cgmodel = CGModel(bond_lengths=bond_lengths,constrain_bonds=constrain_bonds)

        :Example:

        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> from simtk import unit
        >>> backbone_length=1
        >>> sidechain_length=1
        >>> sidechain_positions=0
        >>> bond_length = 7.5 * unit.angstrom
        >>> sigma = 2.0 * bond_length
        >>> epsilon = 0.2 * unit.kilocalorie_per_mole
        >>> sigmas = {'bb_bb_sigma': sigma,'sc_sc_sigma': sigma}
        >>> epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': epsilon}
        >>> A = {'monomer_name': "A", 'backbone_length': backbone_length, 'sidechain_length': sidechain_length, 'sidechain_positions': sidechain_positions, 'num_beads': num_beads, 'bond_lengths': bond_lengths, 'epsilons': epsilons, 'sigmas': sigmas}
        >>> B = {'monomer_name': "B", 'backbone_length': backbone_length, 'sidechain_length': sidechain_length, 'sidechain_positions': sidechain_positions, 'num_beads': num_beads, 'bond_lengths': bond_lengths, 'epsilons': epsilons, 'sigmas': sigmas}
        >>> monomer_types = [A,B]
        >>> sequence = [A,A,A,B,A,A,A,B,A,A,A,B]
        >>> cgmodel = CGModel(monomer_types=monomer_types,sequence=sequence)

        """

    def __init__(
        self,
        positions=None,
        charges={},
        masses={},
        bond_lengths={},
        bond_force_constants={},
        bond_angle_force_constants={},
        torsion_force_constants={},
        torsion_periodicities={},
        equil_bond_angles={},
        equil_torsion_angles={},
        constrain_bonds=True,
        include_bond_forces=False,
        include_nonbonded_forces=True,
        include_bond_angle_forces=True,
        include_torsion_forces=True,
        exclusions=True,
        rosetta_functional_form=False,
        check_energy_conservation=True,
        use_structure_library=False,
        random_positions=False,
        system=None,
        topology=None,
        simulation=None,
        monomer_types=None,
        sequence=None,
    ):

        """
          Initialize definitions for all of the properties of a coarse-grained model

          ..warning:: A large number of default definitions are applied when constructing the CGModel object, in an effort to make intiation of the class easier.  Please review these defaults (listed below) carefully before using this class.

          :param positions: Positions for the particles in the coarse-grained model, default = None
          :type positions: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit )

          :param masses: Masses of all particle types, default = None
          :type masses: dict( 'backbone_bead_masses': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ , 'sidechain_bead_masses': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )

          :param sigmas: Non-bonded bead Lennard-Jones equilibrium interaction distances, default = None
          :type sigmas: dict( 'bb_bb_sigma': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ,'bb_sc_sigma': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ,'sc_sc_sigma': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )

          :param epsilons: Non-bonded Lennard-Jones equilibrium interaction strengths, default = None
          :type epsilons: dict( 'bb_bb_eps': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ,'bb_sc_eps': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ,'sc_sc_eps': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )

          :param bond_lengths: Bond lengths for all bonds, default = None
          :type bond_lengths: dict( 'bb_bb_bond_length': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ,'bb_sc_bond_length': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ,'sc_sc_bond_length': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )

          :param bond_angle_force_constants: Bond angle force constants for all bond types, default = None
          :type bond_angle_force_constants: dict( 'bb_bb__bb_angle_k': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ,'bb_bb_sc_angle_k': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ , 'bb_sc_sc_angle_k': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ , 'sc_sc_sc_angle_k': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ , 'sc_bb_sc_angle_k': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ , 'sc_sc_bb_angle_k': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )

          :param bond_force_constants: Bond force constants for all bond types, default = None
          :type bond_force_constants: dict( 'bb_bb_bond_k': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ,'bb_sc_bond_k': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ , 'sc_sc_bond_k': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )

          :param equil_bond_angles: Equilibrium bond angle for all bond angle types, default = None
          :type equil_bond_angles: dict('bb_bb_bb_angle_0': float,'bb_bb_sc_angle_0': float,'bb_sc_sc_angle_0': float,'sc_sc_sc_angle_0': float, 'sc_bb_sc_angle_0': float,'sc_sc_bb_angle_0': float )

          :param torsion_force_constants: Torsion force constants for all unique torsion definitions, default = None
          :type torsion_force_constants: dict( 'bb_bb_bb_bb_torsion_k': float,'bb_bb_bb_sc_torsion_k': float,'bb_bb_sc_sc_torsion_k': float, 'bb_sc_sc_sc_torsion_k': float, 'sc_bb_bb_sc_torsion_k': float, 'bb_sc_sc_bb_torsion_k': float, 'sc_sc_sc_sc_torsion_k': float,  'sc_bb_bb_bb_torsion_k': float )

          :param equil_torsion_angles: Equilibrium torsion angle for all unique torsion angle definitions, default = 0
          :type equil_torsion_angles: dict( 'bb_bb_bb_bb_torsion_0': float,'bb_bb_bb_sc_torsion_0': float,'bb_bb_sc_sc_torsion_0': float, 'bb_sc_sc_sc_torsion_0': float, 'sc_bb_bb_sc_torsion_0': float, 'bb_sc_sc_bb_torsion_0': float, 'sc_sc_sc_sc_torsion_0': float, 'sc_bb_bb_bb_torsion_0': float )

          :param charges: Charges for all particles, default = None
          :type charges: dict( 'backbone_bead_charges': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ,'sidechain_bead_charges': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )

          :param system: OpenMM System() object, which stores the forces for the coarse grained model, default = None
          :type system: `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_

          :param topology: OpenMM Topology() object, which stores bonds, angles, and other structural attributes of the coarse-grained model, default = None
          :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_

          :param constrain_bonds: Logical variable determining whether bond constraints are applied during a simulation of the energy for the system, default = True
          :type constrain_bonds: Logical

          :param include_bond_forces: Include contributions from bond potentials when calculating the potential energy, default = False
          :type include_bond_forces: Logical

          :param include_nonbonded_forces: Include contributions from nonbonded interactions when calculating the potential energy, default = True
          :type include_nonbonded_forces: Logical

          :param include_bond_angle_forces: Include contributions from bond angle forces when calculating the potential energy, default = True
          :type include_bond_angle_forces: Logical

          :param include_torsion_forces: Include contributions from torsions when calculating the potential energy, default = True
          :type include_torsion_forces: Logical

          :param check_energy_conservation: Flag designating whether or not to perform a test OpenMM simulation with this coarse-grained model.
          :type check_energy_conservation: Logical

          :param use_structure_library: Flag designating whether or not to use a structure from the foldamers ensemble as the initial positions for the particles in the coarse-grained model, default = False
          :type use_structure_library: Logical

          :param monomer_types: A list of dictionary objects containing the properties for unique monomer types (used to construct a heteropolymeric coarse-grained model, default = None
          :type monomer_types: List( dict( 'monomer_name': str, 'backbone_length': int, 'sidechain_length': int, 'sidechain_positions': List( int ), 'num_beads': int, 'bond_lengths': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'epsilons': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'sigmas': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) ) )

          :param backbone_lengths: List of the number of beads in the backbone for unique monomer types within the coarse grained model, default = 1
          :type backbone_lengths: int

          :param sidechain_lengths: List of the number of beads in the sidechain for unique monomer types within the coarse grained model, default = 1
          :type sidechain_lengths: int

          :param sidechain_positions: List of the indices of backbone beads upon which we will place sidechains, default = [0] (add a sidechain to the first backbone bead in each monomer)
          :type sidechain_positions: List( integer )


          :param sequence: The sequence from which to build a heteropolymer.  Defined using a list of 'monomer_types', each of which contains the properties for that monomer, default = None (Homopolymer)
          :type sequence: List( dict( 'monomer_name': str, 'backbone_length': int, 'sidechain_length': int, 'sidechain_positions': List( int ), 'num_beads': int, 'bond_lengths': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'epsilons': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'sigmas': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) ) )

          :param random_positions: Flag designating whether or not to generate a set of random coordinates for the coarse-grained model, default = None


          """

        # define some default units
        self.default_mass = 72 * unit.amu  # from martini 3.0 C1
        self.default_length = 0.47 * unit.nanometers  # from martini 3.0 C1 particle
        self.default_angle = 0.0 * unit.degrees
        self.default_energyscale = 3.5 * unit.kilojoule_per_mole  # from martini 3.0 C1 particle
        self.default_bond_k = (
            1250.0 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
        )  # from martini 3.0
        self.default_torsion_k = 0.0 * unit.kilojoule_per_mole
        self.default_angle_k = (
            0.0 * unit.kilojoule_per_mole / unit.radian / unit.radian
        )  # from martini 3.0
        self.default_charge = 0.0 * unit.elementary_charge
        self.default_periodicity = 1

        """
        Initialize user-defined input.
        """

        # Assign forces based upon input flags
        self.rosetta_functional_form = rosetta_functional_form
        self.include_bond_forces = include_bond_forces
        self.constrain_bonds = constrain_bonds
        self.include_bond_angle_forces = include_bond_angle_forces
        self.include_nonbonded_forces = include_nonbonded_forces
        self.exclusions = exclusions
        self.include_torsion_forces = include_torsion_forces
        self.check_energy_conservation = check_energy_conservation
        self.monomer_types = monomer_types

        # Build a polymer with these model settings
        self.build_polymer(sequence)

        # Assign particle properties
        self.masses = masses
        self.charges = charges
        self.particle_types = add_new_elements(self)
        # Assign bonded force properties
        self.bond_force_constants = bond_force_constants
        self.bond_angle_force_constants = bond_angle_force_constants
        self.equil_bond_angles = equil_bond_angles
        self.torsion_force_constants = torsion_force_constants
        self.torsion_periodicities = torsion_periodicities
        self.equil_torsion_angles = equil_torsion_angles

        # Assign positions
        if positions == None:
            if random_positions:
                self.positions = get_random_positions(self, use_library=use_structure_library)
            else:
                self.positions = None
        else:
            self.positions = positions

        # Define storage for simulation objects from OpenMM
        self.simulation = simulation

        # Define OpenMM topology
        if topology == None:
            if self.positions != None:
                self.topology = build_topology(self, use_pdbfile=True)
        else:
            self.topology = topology

        # Define OpenMM system
        if system == None:
            self.system = build_system(self, rosetta_functional_form=rosetta_functional_form)
        else:
            self.system = system

    def export(self, filename):
        """
        export to a pickle file.

        :param filename: filename for exporting the cgmodel
        :type CGModel: str
        """

        pickle_out = open(filename, "wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()

    def build_polymer(self, sequence):
        """
          Used to build a polymer, or reset the properties for a polymer after parameters such as the polymer_length or sequence have been modified.
          """
        self.polymer_length = len(sequence)
        self.sequence = sequence
        self.process_monomer_types()
        self.num_beads = self.get_num_beads()
        self.particle_list = self.get_particle_list()

        if self.include_bond_forces or self.constrain_bonds:
            self.bond_list = self.get_bond_list()
        else:
            self.exclusions = False
            self.bond_list = []

        self.bond_angle_list = self.get_bond_angle_list()
        self.torsion_list = self.get_torsion_list()
        if self.exclusions:
            self.nonbonded_exclusion_list = self.get_nonbonded_exclusion_list(
                rosetta_functional_form=self.rosetta_functional_form
            )
        else:
            self.nonbonded_exclusion_list = []

        self.nonbonded_interaction_list = self.get_nonbonded_interaction_list()

    def process_monomer_types(self):
        """
          Clean up a list of 'monomer_types' for all unique monomers.

          :param CGModel: CGModel() class object
          :type CGModel: class

          """

        for monomer in self.monomer_types:
            if monomer["monomer_name"] is None:
                print("Error: monomers must have names!")
                exit()  # figure out how to handle with exceptions.

            mn = monomer["monomer_name"]
            if "backbone_length" not in monomer:
                print(f"Using default length 1 for backbone length monomer {mn}")
                monomer["backbone_length"] = 1

            if "sidechain_length" not in monomer:
                print(f"Using default length 1 for sidechain length monomer {mn}")
                monomer["sidechain_length"] = 1

            if "sidechain_positions" not in monomer:
                print(f"Using single sidechain at position [0] for monomer {mn}")
                monomer["sidechain_positions"] = [0]

            if type(monomer["sidechain_positions"]) is not list:
                monomer["sidechain_positions"] = [monomer["sidechain_positions"]]

            for position in monomer["sidechain_positions"]:
                if position >= monomer["backbone_length"]:
                    print(
                        f"Error: side chain position {position} is too large for backbone length {monomer['backbone_length']}"
                    )
                    exit()
            if "bond_lengths" not in monomer:
                print(f"Using default bond length {self.default_length} at for monomer {mn}")
                # could use some error checking if these are not all defined.
                monomer["bond_lengths"] = {
                    "bb_bb_bond_length": self.default_length,
                    "bb_sc_bond_length": self.default_length,
                    "sc_sc_bond_length": self.default_length,
                }

            if "epsilons" not in monomer:
                print(f"Using default epsilon {self.default_energy_scale} for monomer {mn}")
                # could use some error checking if these are not all defined.
                monomer["epsilons"] = {
                    "bb_eps": self.default_energyscale,
                    "sc_eps": self.default_energyscale,
                }

            if "sigmas" not in monomer:
                print(f"Using default sigmas {self.default_length} for monomer {mn}")
                # could use some error checking if these are not all defined.
                monomer["sigmas"] = {
                    "bb_sigma": self.default_length,
                    "sc_sigma": self.default_length,
                }

            num_beads = monomer["backbone_length"]
            for sidechain_position in monomer["sidechain_positions"]:
                num_beads = num_beads + monomer["sidechain_length"]
            if num_beads not in monomer:
                monomer["num_beads"] = num_beads
            elif monomer["num_beads"] != num_beads:
                print(
                    f"Warning: calculated number of beads {num_beads} is not equal to stated {monomer['num_beads']}. Using calculated number of beads."
                )
                monomer["num_beads"] = num_beads

    def get_num_beads(self):
        """
          Calculate the number of beads in a coarse-grained model class object

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
            - num_beads (int) - The total number of beads in the coarse-grained model

          """
        num_beads = 0
        for monomer in self.sequence:
            num_beads = num_beads + monomer["num_beads"]
        return num_beads

    def get_particle_list(self):
        """
          Get a list of particles, where the indices correspond to those in the system/topology.

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
            - particle_list ( List( str ) ) - A list of unique particles in the coarse-grained model

          """
        cg_particle_index = 0
        particle_list = []
        for i, monomer in enumerate(self.sequence):
            for backbone_bead in range(monomer["backbone_length"]):
                particle = dict()
                particle["type"] = "backbone"
                # will need to come up with a better naming scheme than X
                # X for backbones and A for monomers
                particle["name"] = f"X{cg_particle_index}"
                particle["index"] = cg_particle_index
                particle["monomer"] = i
                particle["monomer_type"] = monomer
                particle_list.append(particle)
                cg_particle_index += 1
                if backbone_bead in monomer["sidechain_positions"]:
                    for sidechain in range(monomer["sidechain_length"]):
                        particle = dict()
                        particle["type"] = "sidechain"
                        particle["name"] = f"A{cg_particle_index}"
                        particle["index"] = cg_particle_index
                        particle["monomer"] = i
                        particle["monomer_type"] = monomer
                        particle_list.append(particle)
                        cg_particle_index += 1
        return particle_list

    def get_bond_list(self):
        """
          Construct a bond list for the coarse-grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
             - bond_list ( List( List( int, int ) ) ) - A list of the bonds in the coarse-grained model.

          """
        bond_list = []
        bead_index = 0
        if self.include_bond_forces or self.constrain_bonds:
            for monomer in range(len(self.sequence)):
                monomer_type = self.sequence[monomer]
                for backbone_bead in range(monomer_type["backbone_length"]):
                    if (monomer != 0 and backbone_bead == 0): 
                        parent_index = last_backbone_bead # first backbone bead is attached to the last backbone bead of previous monomer.
                    elif (backbone_bead - 1) in monomer_type["sidechain_positions"]:
                        # if the previous backbone bead has a sidechain
                        parent_index = bead_index - monomer_type["sidechain_length"] - 1
                    else:
                        parent_index = bead_index - 1  # otherwise, the parent of this bead is the previous bead.
                    if bead_index != 0: # if it's the first bead, obviously don't count it.
                        indices = [parent_index, bead_index]
                        bond_list.append([min(indices),max(indices)])
                    if backbone_bead == monomer_type["backbone_length"] - 1:
                        last_backbone_bead = bead_index
                    bead_index = bead_index + 1 # move on to next bead
                    if backbone_bead in monomer_type["sidechain_positions"]:
                        for sidechain_bead in range(monomer_type["sidechain_length"]):
                            parent_index = bead_index - 1
                            indices = [parent_index, bead_index]
                            bond_list.append([min(indices),max(indices)])
                            bead_index = bead_index + 1
        return bond_list

    def get_nonbonded_interaction_list(self):
        """
          Construct a nonbonded interaction list for the coarse-grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
             - interaction_list ( List( List( int, int ) ) ) - A list of the nonbonded interactions (which don't violate exclusion rules) in the coarse-grained model

          """

        interaction_list = []

        if self.include_bond_forces or self.constrain_bonds:
            bond_list = self.get_bond_list()
            for particle_1 in range(self.num_beads):
                for particle_2 in range(particle_1 + 1, self.num_beads):
                    if [particle_1, particle_2] not in bond_list and [
                        particle_2,
                        particle_1,
                    ] not in bond_list:
                        if [particle_1, particle_2] not in interaction_list:
                            if [particle_2, particle_1] not in interaction_list:
                                interaction_list.append([particle_1, particle_2])
                        if [particle_2, particle_1] not in interaction_list:
                            if [particle_1, particle_2] not in interaction_list:
                                interaction_list.append([particle_2, particle_1])
            exclusion_list = self.nonbonded_exclusion_list
            if exclusion_list != None:
                for exclusion in exclusion_list:
                    if exclusion in interaction_list:
                        interaction_list.remove(exclusion)
                    if [exclusion[1], exclusion[0]] in exclusion_list:
                        print([exclusion[1], exclusion[0]])
                        interaction_list.remove([exclusion[1], exclusion[0]])
        else:
            for particle_1 in range(self.num_beads):
                for particle_2 in range(particle_1 + 1, self.num_beads):
                    interaction_list.append([particle_1, particle_2])
        return interaction_list

    def get_nonbonded_exclusion_list(self, rosetta_functional_form=False):
        """
          Get a list of the nonbonded interaction exclusions, which are assigned if two particles are separated by less than three bonds

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
             - exclusion_list ( List( List( int, int ) ) ) - A list of the nonbonded particle interaction exclusions for the coarse-grained model

          """
        bond_list = self.bond_list
        exclusion_list = []

        # for now, we are INCLUDING intraresidue interactions, even though
        # this isn't traditional Rosetta functional form.
        remove_intraresidue_interactions = False
        if rosetta_functional_form and remove_intraresidue_interactions:
            # Remove interactions between particles in the same monomer
            bead_index = 0
            for monomer in self.sequence:
                for beadi in range(monomer["num_beads"]):
                    for beadj in range(beadi + 1, monomer["num_beads"]):
                        exclusion_list.append([bead_index + beadi, bead_index + beadj])
                bead_index = bead_index + monomer["num_beads"]

        for bond in self.bond_list:
            if bond not in exclusion_list and bond.reverse() not in exclusion_list:
                exclusion_list.append(bond)
        for angle in self.bond_angle_list:
            angle_ends = [angle[0], angle[2]]
            if angle_ends not in exclusion_list and angle_ends.reverse() not in exclusion_list:
                exclusion_list.append(angle_ends)
        if rosetta_functional_form:
            for torsion in self.torsion_list:
                torsion_ends = [torsion[0], torsion[3]]
                if (
                    torsion_ends not in exclusion_list
                    and torsion_ends.reverse() not in exclusion_list
                ):
                    exclusion_list.append(torsion_ends)

        return exclusion_list

    def get_bond_angle_list(self):
        """
          Construct a list of bond angles, which can be used to build bond angle potentials for the coarse-grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
             - bond_angles ( List( List( int, int, int ) ) ) - A list of indices for all of the bond angles in the coarse-grained model

          """

        bond_list = self.bond_list
        bond_angles = []
        # Choose the first bond we will use to define a bond angle
        for bond_1 in bond_list:
            # Choose a second bond with which to attempt a bond angle
            # definition.
            for bond_2 in bond_list:
                # Make sure the bonds are different
                if bond_2 != bond_1 and [bond_2[1], bond_2[0]] != bond_1:
                    # Make sure the bonds share a common atom
                    bond_angle = []
                    if bond_2[0] in bond_1 or bond_2[1] in bond_1:
                        if bond_2[0] == bond_1[1]:
                            bond_angle = [bond_1[0], bond_1[1], bond_2[1]]
                        if bond_2[0] == bond_1[0]:
                            bond_angle = [bond_1[1], bond_1[0], bond_2[1]]
                        if bond_2[1] == bond_1[1]:
                            bond_angle = [bond_1[0], bond_1[1], bond_2[0]]
                        if bond_2[1] == bond_1[0]:
                            bond_angle = [bond_1[1], bond_1[0], bond_2[0]]
                    if len(bond_angles) > 0 and len(bond_angle) > 0:
                        unique = True
                        bond_angle_set = set(tuple(angle) for angle in bond_angles)
                        bond_angles_temp = [list(angle) for angle in bond_angle_set]
                        bond_angle_reverse = [bond_angle[2], bond_angle[1], bond_angle[0]]
                        if any(
                            [
                                bond_angle in bond_angles_temp,
                                bond_angle_reverse in bond_angles_temp,
                            ]
                        ):
                            unique = False
                        if unique:
                            bond_angles.append(bond_angle)
                    if len(bond_angles) == 0 and len(bond_angle) != 0:
                        bond_angles.append(bond_angle)
        return bond_angles

    def get_torsion_list(self):
        """
          Construct a list of particle indices from which to define torsions for the coarse-grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
            - torsions ( List( List( int, int, int, int ) ) ) - A list of the particle indices for the torsions in the coarse-grained model

          """

        bond_list = self.bond_list
        angle_list = self.bond_angle_list
        torsions = []
        # Choose the first bond in the torsion
        for bond_1 in bond_list:
            # Choose a second bond with which to attempt to define a torsion.
            torsion = []
            for bond_2 in bond_list:
                # Make sure bonds 1 and 2 are different
                if bond_2 != bond_1 and [bond_2[1], bond_2[0]] != bond_1:
                    if bond_1[0] == bond_2[1]:
                        torsion = [bond_2[0], bond_2[1], bond_1[1]]
                    if bond_1[1] == bond_2[0]:
                        torsion = [bond_1[0], bond_1[1], bond_2[1]]

                if len(torsion) == 3:
                    # Choose a third bond with which to attempt a torsion definition
                    for bond_3 in bond_list:
                        # Make sure the third bond is different from the first two bonds.
                        if (
                            bond_3 != bond_1
                            and [bond_3[1], bond_3[0]] != bond_1
                            and bond_3 != bond_2
                            and [bond_3[1], bond_3[0]] != bond_2
                        ):
                            if bond_3[0] in torsion or bond_3[1] in torsion:
                                if bond_3[0] == torsion[0] and len(torsion) < 4:
                                    torsion.insert(0, bond_3[1])
                                if bond_3[0] == torsion[1] and len(torsion) == 4:
                                    torsion[0] = bond_3[1]

                                if bond_3[1] == torsion[0] and len(torsion) < 4:
                                    torsion.insert(0, bond_3[0])
                                if bond_3[1] == torsion[1] and len(torsion) == 4:
                                    torsion[0] = bond_3[0]
                                if bond_3[0] == torsion[2] and len(torsion) < 4:
                                    torsion.append(bond_3[1])
                                if bond_3[0] == torsion[2] and len(torsion) == 4:
                                    torsion[3] = bond_3[1]
                                if bond_3[1] == torsion[2] and len(torsion) < 4:
                                    torsion.append(bond_3[0])
                                if bond_3[1] == torsion[2] and len(torsion) == 4:
                                    torsion[3] = bond_3[0]
                        if len(torsion) == 4:
                            # Determine if the particles defining this torsion are suitable.
                            if len(torsions) == 0:
                                torsions.append(torsion)
                            else:
                                unique = True
                                for existing_torsion in torsions:
                                    if Counter(torsion) == Counter(existing_torsion):
                                        unique = False
                                if unique:
                                    torsions.append(torsion)

        for angle in angle_list:
            for bond in bond_list:
                if (
                    bond[0] in angle
                    and bond[1] not in angle
                    or bond[1] in angle
                    and bond[0] not in angle
                ):
                    if bond[0] == angle[0]:
                        torsion = [bond[1], angle[0], angle[1], angle[2]]
                        unique = True
                        for existing_torsion in torsions:
                            if Counter(torsion) == Counter(existing_torsion):
                                unique = False
                        if unique:
                            torsions.append(torsion)
                    if bond[0] == angle[2]:
                        torsion = [angle[0], angle[1], angle[2], bond[1]]
                        unique = True
                        for existing_torsion in torsions:
                            if Counter(torsion) == Counter(existing_torsion):
                                unique = False
                        if unique:
                            torsions.append(torsion)
                    if bond[1] == angle[0]:
                        torsion = [bond[0], angle[0], angle[1], angle[2]]
                        unique = True
                        for existing_torsion in torsions:
                            if Counter(torsion) == Counter(existing_torsion):
                                unique = False
                        if unique:
                            torsions.append(torsion)
                    if bond[1] == angle[2]:
                        torsion = [angle[0], angle[1], angle[2], bond[0]]
                        unique = True
                        for existing_torsion in torsions:
                            if Counter(torsion) == Counter(existing_torsion):
                                unique = False
                        if unique:
                            torsions.append(torsion)

        torsion_set = set(tuple(torsion) for torsion in torsions)
        torsions = [list(torsion) for torsion in torsion_set]
        return torsions

    def get_particle_name(self, particle_index):
        """
          Returns the name of a particle, given its index within the model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :returns: 
            - particle_name ( str ) - The name of the particle

          """
        particle_name = self.particle_list[particle_index]["name"]
        return particle_name

    def get_particle_type(self, particle_index):
        """
          Indicates if a particle is a backbone bead or a sidechain bead

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :returns: 
             - particle_type (str) - 'backbone' or 'sidechain'

          """
        if self.particle_list[particle_index]["type"] == "backbone":
            particle_type = "backbone"
        elif self.particle_list[particle_index]["type"] == "sidechain":
            particle_type = "sidechain"
        else:
            print(f"ERROR: The particle type definition could not be found for {particle_index}")

        return particle_type

    def get_particle_monomer(self, particle_index):
        """
          Indicates which type of monomer a particle belongs to

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the monomer type
          :type particle_index: int

          :returns: 
             - monomer_type (dict) : monomer type

          """
        return self.particle_list[particle_index]["monomer_type"]

    def get_particle_mass(self, particle_index):
        """
          Get the mass for a particle, given its index within the coarse-grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :returns: 
            - particle_mass ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The mass for the provided particle index

          """
        particle_type = self.get_particle_type(particle_index)

        particle_mass = None

        if particle_type == "backbone":
            try:
                particle_mass = self.masses["backbone_bead_masses"]
            except:
                print("No particle mass definition found for particle type: backbone")
                print(f"Applying a default definition: 'backbone_bead_masses'={self.default_mass}")
                print("If you observe unusual behavior, it is most likely because")
                print("this default definition is inappropriate for your model.")
                particle_mass = self.default_mass
                self.masses.update({"backbone_bead_masses": particle_mass})

        if particle_type == "sidechain":
            try:
                particle_mass = self.masses["sidechain_bead_masses"]
            except:
                print("No particle mass definition found for particle type: sidechain")
                print("Applying a default definition: 'sidechain_bead_masses'={self.default_mass")
                print("If you observe unusual behavior, it is most likely because")
                print("this default definition is inappropriate for your model.")
                particle_mass = self.default_mass
                self.masses.update({"sidechain_bead_masses": particle_mass})
        return particle_mass

    def get_particle_charge(self, particle_index):
        """
          Returns the charge for a particle, given its index within the coarse-grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :returns: 
            - particle_charge ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The charge for the provided particle index

          """
        particle_type = self.get_particle_type(particle_index)

        if particle_type == "backbone":
            try:
                particle_charge = self.charges["backbone_bead_charges"]
            except:
                print("No particle charge definition was found for particle type: backbone")
                print(f"Applying a default definition: charge = {self.default_charge}")
                self.charges.update({"backbone_bead_charges": self.default_charge})
                particle_charge = self.charges["backbone_bead_charges"]

        if particle_type == "sidechain":
            try:
                particle_charge = self.charges["sidechain_bead_charges"]
            except:
                print("No particle charge definition was found for particle type: sidechain")
                print(f"Applying a default definition: charge = {self.default_charge}")
                self.charges.update({"sidechain_bead_charges": self.default_charge})
                particle_charge = self.charges["sidechain_bead_charges"]

        return particle_charge

    def get_sigma(self, particle_index):
        """
          Returns the Lennard-Jones potential sigma value for a particle, given its index within the coarse-grained model.

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :param particle_type: Designates a particle as "backbone" or "sidechain"
          :type particle_type: str

          :returns: 
            - sigma ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned Lennard-Jones sigma value for the provided particle index

          """

        particle_type = self.get_particle_type(particle_index)
        monomer_type = self.get_particle_monomer(particle_index)

        sigma = None

        abbrev = {"backbone": "bb", "sidechain": "sc"}
        for ptype in ["backbone", "sidechain"]:
            sigma_type = abbrev[ptype] + "_sigma"
            if particle_type == ptype:
                try:
                    sigma = monomer_type["sigmas"][sigma_type]
                except:
                    print(
                        f"No Lennard-Jones potential 'sigma' definition found for particle type: {monomer_type['monomer_name']}:{sigma_type}"
                    )
                    print(
                        "Applying a definition based upon the default between particles of this type:"
                    )
                    print(f"{sigma_type} = {self.default_length}")
                    print("If you observe unusual behavior, it is most likely because")
                    print("this default definition is inappropriate for your model.")
                    monomer_type["sigmas"].update({sigma_type: self.default_length})
                    sigma = self.default_length

        return sigma

    def get_epsilon(self, particle_index):
        """
          Returns the Lennard-Jones potential epsilon value for a particle, given its index within the coarse-grained model.

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :param particle_type: Designates a particle as "backbone" or "sidechain"
          :type particle_type: str

          :returns: 
             - epsilon ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned Lennard-Jones epsilon value for the provided particle index


          """
        particle_type = self.get_particle_type(particle_index)
        monomer_type = self.get_particle_monomer(particle_index)

        epsilon = None

        abbrev = {"backbone": "bb", "sidechain": "sc"}
        for ptype in ["backbone", "sidechain"]:
            epsilon_type = abbrev[ptype] + "_eps"
            if particle_type == ptype:
                try:
                    epsilon = monomer_type["epsilons"][epsilon_type]
                except:
                    print(
                        f"No Lennard-Jones potential 'epsilon' definition found for particle type: {monomer_type['monomer_name']}:{epsilon_type}"
                    )
                    print(
                        "Applying a definition based upon the default between particles of this type:"
                    )
                    print(f"{epsilon_type} = {self.default_energyscale}")
                    print("If you observe unusual behavior, it is most likely because")
                    print("this default definition is inappropriate for your model.")
                    monomer_type["epsilons"].update({epsilon_type: self.default_energyscale})
                    epsilon = self.default_energyscale

        return epsilon

    def get_bond_length(self, bond):
        """
          Determines the correct bond length for two particles, given their indices.

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_1_index: Index for the first particle
          :type particle_1_index: int

          :param particle_2_index: Index for the second particle
          :type particle_2_index: int

          :returns: 
             - bond_length ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned bond length for the provided particles

          """

        particle_1_type = self.get_particle_type(bond[0])
        particle_2_type = self.get_particle_type(bond[1])

        # CONVENTION: the joining bond belongs to the first monomer.
        monomer_type = self.particle_list[bond[0]]["monomer_type"]

        bond_length = None

        if particle_1_type == "backbone" and particle_2_type == "backbone":
            string_name = reverse_string_name = "bb_bb_bond_length"
        if particle_1_type == "backbone" and particle_2_type == "sidechain":
            string_name = "bb_sc_bond_length"
            reverse_string_name = "sc_bb_bond_length"
        if particle_1_type == "sidechain" and particle_2_type == "backbone":
            string_name = "sc_bb_bond_length"
            reverse_string_name = "bb_sc_bond_length"
        if particle_1_type == "sidechain" and particle_2_type == "sidechain":
            string_name = reverse_string_name = "sc_sc_bond_length"
        try:
            bond_length = monomer_type["bond_lengths"][string_name]
        except:
            try:
                bond_length = monomer_type["bond_lengths"][reverse_string_name]
            except:
                print(
                    f"No bond length definition provided for '{string_name}', setting '{string_name}'={self.default_length}"
                )
                monomer_type["bond_lengths"].update({string_name: self.default_length})
                monomer_type["bond_lengths"].update({reverse_string_name: self.default_length})
                bond_length = monomer_type["bond_lengths"][string_name]

        return bond_length

    def get_bond_force_constant(self, bond):
        """
          Determines the correct bond force constant for two particles, given their indices

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_1_index: Index for the first particle
          :type particle_1_index: int

          :param particle_2_index: Index for the second particle
          :type particle_2_index: int

          :returns: 
             - bond_force_constant ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned bond force constant for the provided particles

          """

        particle_1_type = self.get_particle_type(bond[0])
        particle_2_type = self.get_particle_type(bond[1])

        if particle_1_type == "backbone" and particle_2_type == "backbone":
            string_name = reverse_string_name = "bb_bb_bond_k"
        if particle_1_type == "backbone" and particle_2_type == "sidechain":
            string_name = "bb_sc_bond_k"
            reverse_string_name = "sc_bb_bond_k"
        if particle_1_type == "sidechain" and particle_2_type == "backbone":
            string_name = "sc_bb_bond_k"
            reverse_string_name = "bb_sc_bond_k"
        if particle_1_type == "sidechain" and particle_2_type == "sidechain":
            string_name = reverse_string_name = "sc_sc_bond_k"

        bond_force_constant = None

        try:
            bond_force_constant = self.bond_force_constants[string_name]
        except:
            try:
                bond_force_constant = self.bond_force_constants[reverse_string_name]
            except:
                print(
                    f"No bond force constant provided for '{string_name}', setting '{string_name}'={self.default_bond_k}"
                )
                self.bond_force_constants.update({string_name: self.default_bond_k})
                self.bond_force_constants.update({reverse_string_name: self.default_bond_k})
                bond_force_constant = self.bond_force_constants[string_name]

        return bond_force_constant

    def get_equil_bond_angle(self, angle):
        """
          Determines the correct equilibrium bond angle between three particles, given their indices within the coarse-grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_1_index: Index for the first particle
          :type particle_1_index: int

          :param particle_2_index: Index for the second particle
          :type particle_2_index: int

          :param particle_3_index: Index for the third particle
          :type particle_3_index: int

          :returns: 
            - equil_bond_angle (float) - The assigned equilibrium bond angle for the provided particles

          """

        particle_types = [
            self.get_particle_type(angle[0]),
            self.get_particle_type(angle[1]),
            self.get_particle_type(angle[2]),
        ]

        equil_bond_angle = None

        string_name = ""
        reverse_string_name = ""
        for i in range(3):
            string_name += particle_types[i][0] + particle_types[i][4] + "_"
        for i in range(3):
            reverse_string_name += particle_types[2 - i][0] + particle_types[2 - i][4] + "_"

        string_name += "angle_0"
        reverse_string_name += "angle_0"

        try:
            equil_bond_angle = self.equil_bond_angles[string_name]
        except:
            try:
                equil_bond_angle = self.equil_bond_angles[reverse_string_name]
            except:
                print(
                    f"No equilibrium bond angle definition provided for '{string_name}', setting '{string_name}'={self.default_angle}"
                )
                self.equil_bond_angles.update({string_name: self.default_angle})
                self.equil_bond_angles.update({reverse_string_name: self.default_angle})
                equil_bond_angle = self.equil_bond_angles[string_name]

        if equil_bond_angle == None:
            print(
                "ERROR: No equilibrium bond angle definition was found for the following particle types:"
            )
            print(f"{particle_types[0]}-{particle_types[1]}-{particle_types[2]}")
            print("This means that at least one of the particle types has not been defined.")
            print("Check the names and definitions for the particle types in your model.")
            exit()

        return equil_bond_angle

    def get_bond_angle_force_constant(self, angle):
        """
          Determines the correct bond angle force constant for a bond angle between three particles, given their indices within the coarse-grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_1_index: Index for the first particle
          :type particle_1_index: int

          :param particle_2_index: Index for the second particle
          :type particle_2_index: int

          :param particle_3_index: Index for the third particle
          :type particle_3_index: int

          :returns: 
             - bond_angle_force_constant ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned bond angle force constant for the provided particles


          """
        particle_types = [
            self.get_particle_type(angle[0]),
            self.get_particle_type(angle[1]),
            self.get_particle_type(angle[2]),
        ]

        bond_angle_force_constant = None

        string_name = ""
        reverse_string_name = ""
        for i in range(3):
            string_name += particle_types[i][0] + particle_types[i][4] + "_"
        for i in range(3):
            reverse_string_name += particle_types[2 - i][0] + particle_types[2 - i][4] + "_"

        string_name += "angle_k"
        reverse_string_name += "angle_k"

        try:
            bond_angle_force_constant = self.bond_angle_force_constants[string_name]
        except:
            try:
                bond_angle_force_constant = self.bond_angle_force_constants[reverse_string_name]
            except:
                print(
                    f"No bond angle force constant definition provided for '{string_name}', setting '{string_name}'={self.default_angle_k}"
                )
                self.bond_angle_force_constants.update({string_name: self.default_angle_k})
                self.bond_angle_force_constants.update({reverse_string_name: self.default_angle_k})
                bond_angle_force_constant = self.bond_angle_force_constants[string_name]

        if bond_angle_force_constant == None:
            print(
                "ERROR: No bond angle force constant definition was found for the following particle types:"
            )
            print(f"{particle_types[0]}-{particle_types[1]}-{particle_types[2]}")
            print("This means that at least one of the particle types has not been defined.")
            print("Check the names and definitions for the particle types in your model.")
            exit()

        return bond_angle_force_constant

    def get_torsion_periodicity(self, torsion):
        """         
        Determines the correct periodicity for a torsion (bond angle involving four particles), given their indices within the coarse-grained model

        :param CGModel: CGModel() class object
        :type CGModel: class

        :param torsion: A list of the indices for the particles in a torsion
        :type torsion: List( int )

        :returns:
        - torsion_periodicity ( int ) - The periodicity for the input torsion

        """

        particle_types = [
            self.get_particle_type(torsion[0]),
            self.get_particle_type(torsion[1]),
            self.get_particle_type(torsion[2]),
            self.get_particle_type(torsion[3]),
        ]

        torsion_periodicity = None

        string_name = ""
        reverse_string_name = ""
        for i in range(4):
            string_name += particle_types[i][0] + particle_types[i][4] + "_"
        for i in range(4):
            reverse_string_name += particle_types[3 - i][0] + particle_types[3 - i][4] + "_"

        string_name += "period"
        reverse_string_name += "period"

        try:
            torsion_periodicity = self.torsion_periodicities[string_name]
        except:
            try:
                torsion_periodicity = self.torsion_periodicities[reverse_string_name]
            except:
                print(
                    f"No torsion periodicity definition provided for '{string_name}', setting '{string_name}'={self.default_periodicity}"
                )
                self.torsion_periodicities.update({string_name: self.default_periodicity})
                self.torsion_periodicities.update({reverse_string_name: self.default_periodicity})
                torsion_periodicity = self.torsion_periodicities[string_name]

        # does it reach here?
        if torsion_periodicity == None:
            print(
                "ERROR: No torsion periodicity definition was found for the following particle types:"
            )
            print(
                f"{particle_types[0]}-{particle_types[1]}-{particle_types[2]}-{particle_types[3]}"
            )
            print("This means that at least one of the particle types has not been defined.")
            print("Check the names and definitions for the particle types in your model.")
            exit()

        return torsion_periodicity

    def get_torsion_force_constant(self, torsion):
        """         
          Determines the correct torsion force constant for a torsion (bond angle involving four particles), given their indices within the coarse-grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param torsion: A list of the indices for the particles in a torsion
          :type torsion: List( int )

          :returns: 
             - torsion_force_constant ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned torsion force constant for the provided particles

          """
        particle_types = [
            self.get_particle_type(torsion[0]),
            self.get_particle_type(torsion[1]),
            self.get_particle_type(torsion[2]),
            self.get_particle_type(torsion[3]),
        ]

        torsion_force_constant = None

        string_name = ""
        reverse_string_name = ""
        for i in range(4):
            string_name += particle_types[i][0] + particle_types[i][4] + "_"
        for i in range(4):
            reverse_string_name += particle_types[3 - i][0] + particle_types[3 - i][4] + "_"

        string_name += "torsion_k"
        reverse_string_name += "torsion_k"

        try:
            torsion_force_constant = self.torsion_force_constants[string_name]
        except:
            try:
                torsion_force_constant = self.torsion_force_constants[reverse_string_name]
            except:
                print(
                    f"No torsion force constant definition provided for '{string_name}', setting '{string_name}'={self.default_torsion_k}"
                )
                self.torsion_force_constants.update({string_name: self.default_torsion_k})
                self.torsion_force_constants.update({reverse_string_name: self.default_torsion_k})
                torsion_force_constant = self.torsion_force_constants[string_name]

        # can it reach here?
        if torsion_force_constant == None:
            print(
                "ERROR: No torsion force constant definition was found for the following particle types:"
            )
            print(
                f"{particle_types[0]}-{particle_types[1]}-{particle_types[2]}-{particle_types[3]}"
            )
            print("This means that at least one of the particle types has not been defined.")
            print("Check the names and definitions for the particle types in your model.")
            exit()

        return torsion_force_constant

    def get_equil_torsion_angle(self, torsion):
        """         
          Determines the correct equilibrium angle for a torsion (bond angle involving four particles), given their indices within the coarse-grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param torsion: A list of the indices for the particles in a torsion
          :type torsion: List( int )

          :returns: 
             - equil_torsion_angle (float) - The assigned equilibrium torsion angle for the provided particles

          """
        particle_types = [
            self.get_particle_type(torsion[0]),
            self.get_particle_type(torsion[1]),
            self.get_particle_type(torsion[2]),
            self.get_particle_type(torsion[3]),
        ]

        equil_torsion_angle = None

        string_name = ""
        reverse_string_name = ""
        for i in range(4):
            string_name += particle_types[i][0] + particle_types[i][4] + "_"
        for i in range(4):
            reverse_string_name += particle_types[3 - i][0] + particle_types[3 - i][4] + "_"

        string_name += "torsion_0"
        reverse_string_name += "torsion_0"

        try:
            equil_torsion_angle = self.equil_torsion_angles[string_name]
        except:
            try:
                equil_torsion_angle = self.equil_torsion_angles[reverse_string_name]
            except:
                print(
                    f"No equilibrium torsion angle definition provided for '{string_name}', setting '{string_name}'={self.default_angle}"
                )
                self.equil_torsion_angles.update({string_name: self.default_angle})
                self.equil_torsion_angles.update({reverse_string_name: self.default_angle})
                equil_torsion_angle = self.equil_torsion_angles[string_name]

        # can it reach here?
        if equil_torsion_angle == None:
            print(
                "ERROR: No equilibrium torsion angle definition was found for the following particle types:"
            )
            print(
                f"{particle_types[0]}-{particle_types[1]}-{particle_types[2]}-{particle_types[3]}"
            )
            print("This means that at least one of the particle types has not been defined.")
            print("Check the names and definitions for the particle types in your model.")
            exit()

        return equil_torsion_angle
