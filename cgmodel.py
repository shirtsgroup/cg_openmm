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
        particle_type_list={},
        charges={},
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

          :param particle list: list of all particle types, default = None
          :type particle: list 

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
        self.default_bond_force_constant = (
            1250.0 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
        )  # from martini 3.0
        self.default_torsion_force_constant = 0.0 * unit.kilojoule_per_mole
        self.default_bond_angle_force_constant = (
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
        self.bond_lengths = bond_lengths

        # fill in defaults in particle list
        self.particle_type_list = self._validate_particle_type_list(particle_type_list)

        # Build a polymer with these model settings
        self.build_polymer(sequence)

        # Assign particle properties
        self.particle_types = add_new_elements(self)

        # Assign bonded force properties
        self.bond_force_constants = bond_force_constants
        self.bond_angle_force_constants = bond_angle_force_constants
        self.equil_bond_angles = equil_bond_angles
        self.torsion_force_constants = torsion_force_constants
        self.torsion_periodicities = torsion_periodicities
        self.equil_torsion_angles = equil_torsion_angles
        self._validate_bonded_forces()
        
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

    def _validate_bonded_forces(self):

        # check the names that are included in the dictionaries to make sure
        # there are no mispellings.

        # dictionary of the force attributes

        # for each force attribute, which appears in the dictionary defining the forces,
        # define certain properties;
        #     "default name" : the name to look for default definitions of those nonboded forces.
        #     "default value" : the value to store if the default is not given,
        #     "suffix" : the suffix that those forces should have
        # We are trying to minimize the number of places adding new forces changes the code,
        # and this should help with that.

        self.bonded_force_attributes = {
            'bond_lengths': {
                'default_name' : "default_bond_length",
                'default_value' : self.default_length,
                'suffix' : "bond_length"
            },
            'bond_force_constants' : {
                'default_name' : "default_bond_force_constant",
                'default_value' : self.default_bond_force_constant,
	 	'suffix' : 'bond_force_constant'
            },
            'equil_bond_angles' : {
                'default_name' : "default_equil_bond_angle",
                'default_value' : self.default_angle,
		'suffix' : 'equil_bond_angle'             
            },
            'bond_angle_force_constants' : {
                'default_name' : "default_angle_bond_force_constant",
                'default_value' : self.default_bond_angle_force_constant,
		'suffix' : 'bond_angle_force_constant'
            },
            'equil_torsion_angles' : {
                'default_name' : "default_equil_torsion_angle",
                'default_value' : self.default_angle,
		'suffix' : 'equil_torsion_angle'             
            },
            'torsion_force_constants' : {
                'default_name' : "default_torsion_force_constant",
                'default_value' : self.default_torsion_force_constant,
		'suffix' : 'torsion_force_constant'
            },
            'torsion_periodicities' : {
                'default_name' : "default_torsion_periodicity",
                'default_value' : self.default_periodicity,
		'suffix' : 'torsion_periodicity'
            },
        }

        # make sure all the property values are internally consistent
        for attribute in self.bonded_force_attributes:
            # for the bonded force attributes
            if hasattr(self,attribute):
                properties = self.bonded_force_attributes[attribute]
                default_name = properties['default_name']
                # if the default name hasn't been defined for this model
                if default_name not in getattr(self,attribute):
                    default_value = properties['default_value']
                    # set it to the the default for the program.
                    print(f"Warning: No {default_name}: setting to {default_value}")
                default_suffix = properties['suffix']
                for force in getattr(self,attribute):
                    # make sure all forces have the corresponding suffix.
                    if default_suffix not in force:
                        print(f"Warning: force term \'{force}\' does not have proper suffix of {default_suffix}")
                        exit()

    def _validate_particle_type_list(self,particle_type_list):
        """
        parameters: list of particle types

        Check each of the defined particles to make sure it's properly defined

        """
        if particle_type_list is None or len(particle_type_list) == 0:
            # we need a default particle.  Call it a.
            default_particle_type_name = "a"
            print(
                f"No particles defined: creating a default particle named: {default_particle_type_name}"
            )
            particle_type_list = list()
            particle = dict()
            particle["particle_type_name"] = default_particle_type_name
            particle_type_list.append(particle)

        for particle in particle_type_list:
            if "particle_type_name" not in particle:
                print(f'Particle has no attribute \"particle_type_name\": Exiting now')
                exit()
            name = particle["particle_type_name"]
            if "sigma" not in particle:
                print(
                    f"sigma not defined for particle type {name} using default sigma: {self.default_length}"
                )
                particle["sigma"] = self.default_length
            if "mass" not in particle:
                print(
                    f"mass not defined for particle type {name} using default mass: {self.default_mass:}"
                )
                particle["mass"] = self.default_mass
            if "epsilon" not in particle:
                print(
                    f"epsilon not defined for particle type {name} using default epsilon: {self.default_energyscale}"
                )
                particle["epsilon"] = self.default_energyscale
            if "charge" not in particle:
                print(
                    f"charge not defined for particle type {name} using default epsilon: {self.default_charge}"
                )
                particle["charge"] = self.default_charge
                
        return particle_type_list

    def build_polymer(self, sequence):
        """
          Used to build a polymer, or reset the properties for a polymer after parameters such as the polymer_length or sequence have been modified.
          """
        self.polymer_length = len(sequence)
        self.sequence = sequence
        self.process_monomer_types()
        self.num_beads = self.get_num_beads()
        self.particle_list = self.create_particle_list()

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
            if "particle_sequence" not in monomer:
                print(f"Error: monomer {mm} must have a list of particle types!")
                exit()  # figure out how to handle with exceptions.

            if "bond_list" not in monomer or (
                len(monomer["bond_list"]) == 0 and len(monomer["particle_sequence"] != 1)
            ):
                print(
                    f"Error: monomer {mm} is has more than one particle, so it must have a bond list of pairs of bonded particles!"
                )
                exit()  # figure out how to handle with exceptions.

            if "start" not in monomer:
                print(
                    f"Warning: no starting particle is indicated for monomer {mm}: I'm assuming it's the first particle in the sequence."
                )
                monomer["start"] = 0

            if "end" not in monomer:
                print(
                    f"Warning: no ending particle is indicated for monomer {mm}: I'm assuming it's the last one in the sequence."
                )
                monomer["end"] = len(monomer["particle_sequence"]) - 1

            monomer["num_beads"] = len(monomer["particle_sequence"])

            # double check the bonds are consistent with the particles:

            # are any of the bond particles too large?
            for bond in monomer["bond_list"]:
                if bond[0] >= monomer["num_beads"] or bond[1] >= monomer["num_beads"]:
                    print(
                        f"Error: monomer {mn} has a bond [{bond[0]},{bond[1]}] with a particle index too high (>={monomer['num_beads']})"
                    )
                    exit()  # figure out how to handle with exceptions.
                    
            # are there any particles with no bonds?
            unbonded = True
            for i in range(monomer["num_beads"]):
                for bond in monomer["bond_list"]:
                    if i in (bond[0],bond[1]):
                        unbonded = False
                        break
                if unbonded == False:
                    break
                    
            if unbonded:
                    print(
                        f"Error: particle {i} in monomer {mm} has no bonds."
                    )
                    exit()  
            
                
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

    def create_particle_list(self):
        """
          Get a list of particles, where the indices correspond to those in the system/topology.

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
            - particle_list ( List( str ) ) - A list of unique particles in the coarse-grained model

          """
        particle_index = 0
        particle_list = []
        for i, monomer in enumerate(self.sequence):
            seq = monomer["particle_sequence"]
            for j, bead in enumerate(seq):
                particle = dict()
                particle["type"] = bead
                # will need to come up with a better naming scheme than X
                # X for backbones and A for monomers
                if 'particle_type_name' not in bead:
                    print("\'particle_type_name\' not defined, cannot contiue") 
                    exit()
                particle["name"] = f"{bead['particle_type_name']}{particle_index}"
                particle["index"] = particle_index
                particle["monomer"] = i
                particle["monomer_type"] = monomer
                particle_list.append(particle)
                particle_index += 1
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
            for i, monomer in enumerate(self.sequence):
                monomer_bond_list = []
                for bond in monomer["bond_list"]:
                    monomer_bond_list.append([bond[0] + bead_index, bond[1] + bead_index])

                for j, bead in enumerate(monomer["particle_sequence"]):
                    # first, connect the monomer to the last monomer
                    if i != 0 and j == monomer["start"]:
                        # first backbone bead is attached to the last backbone bead of previous monomer.
                        bond_list.append([last_backbone_bead, bead_index])
                    if j == monomer["end"]:
                        last_backbone_bead = bead_index
                    bead_index = bead_index + 1 # increment for bookkeeping
                bond_list += monomer_bond_list
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

    def get_torsion_list(self):   # MRS: really slow, should be looked at.
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

    def get_particle_attribute(self, particle, attribute):
        """
        get various attributes of a particle, given either the index or the particle dictionary

        """
        
        if attribute in ['monomer', 'monomer_type', 'name', 'index', 'type']:
            # these are attributes of the particles in the list
            if type(particle) == dict:
                return particle[attribute] 
            elif type(particle) == int:
                return self.particle_list[particle][attribute]
        elif attribute in ['mass','charge','epsilon','sigma','particle_type_name']:
            # these are attributes of the partilce type
            if type(particle) == dict:
                return particle["type"][attribute] 
            elif type(particle) == int:
                return self.particle_list[particle]["type"][attribute]        
        return

    def get_particle_name(self, particle):
        """
          Returns the name of a particle, given its index within the model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :returns: 
            - particle_name ( str ) - The name of the particle

          """
        return self.get_particle_attribute(particle,"name")


    def get_particle_index(self, particle):
        """
          Returns the index of a particle, given its index within the model or the particle dictionary. Obviously,
          kind of redundant if using the index instead of the particle dictionary

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :returns: 
            - particle_name ( str ) - The name of the particle

          """
        return self.get_particle_attribute(particle,"index")

    def get_particle_type(self, particle):
        """
          Gives the type of a particle (a dictionary)

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle: Index of the particle for which we would like to determine the type OR partice dictionary
          :type particle: int or dict()

          :returns: 
             - particle_type (str):

          """
        
        return self.get_particle_attribute(particle,"type")

    def get_particle_type_name(self, particle):
        """
          Gives the type of a particle.

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type OR partice dictionary
          :type particle_index: int or dict()

          :returns: 
             - particle_type (str):

          """
        
        return self.get_particle_attribute(particle,"particle_type_name")

    
    def get_particle_monomer_type(self, particle):
        """
          Indicates which type of monomer a particle belongs to

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the monomer type
          :type particle_index: int

          :returns: 
             - monomer_type (dict) : monomer type

          """
        return self.get_particle_attribute(particle,"monomer_type")

    def get_particle_monomer(self, particle):
        
        """
          Indicates which monomer index a particle belongs to

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the monomer type
          :type particle_index: int

          :returns: 
             - monomer_type (dict) : monomer type

          """
        return self.get_particle_attribute(particle,"monomer")

    def get_particle_mass(self, particle):
        """
          Returns the mass of a particle, given its index within the coarse-grained model or the particle dictionary

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle: Index of the particle for which we would like to determine the type, or dict()
          :type particle: int or dict()

          :returns: 
             - epsilon ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned Lennard-Jones epsilon value for the provided particle index


          """

        return self.get_particle_attribute(particle,"epsilon")        

    
    def get_particle_charge(self, particle):
        """
          Returns the charge for a particle, given its index within the coarse-grained model, or the dict

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :returns: 
            - particle_charge ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The charge for the provided particle index

          """
        return self.get_particle_attribute(particle,"charge")

    def get_particle_sigma(self, particle):
        
        """
          Returns the Lennard-Jones potential sigma value for a particle, given the particle index

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :returns: 
            - sigma ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned Lennard-Jones sigma value for the provided particle index

          """
        return self.get_particle_attribute(particle,"sigma")

    def get_particle_epsilon(self, particle):
        """
          Returns the Lennard-Jones potential epsilon value for a particle, given its index within the coarse-grained model.

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :returns: 
             - epsilon ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned Lennard-Jones epsilon value for the provided particle index


          """

        return self.get_particle_attribute(particle,"epsilon")        

    def _get_bonded_parameter(self, particle_types, force):

        '''
        internal function for returning any force value.

        parameters: the string name of the bonded force of interest

        returns: the value of the parameter for the atoms involved in the interaction
        '''

        # get the details for this force
        properties = self.bonded_force_attributes[force]
        suffix = properties['suffix']
        default_name = properties['default_name']
        default_value = properties['default_value']
        
        # first, construct the name of the force that is needed.
        string_name = ""
        reverse_string_name = ""
        for particle in particle_types:
            string_name += f"{particle}_"
        for particle in reversed(particle_types):
            reverse_string_name += f"{particle}_"

        string_name += suffix
        reverse_string_name += suffix
            
        parameter_value = None
        forces = getattr(self,force)
        try:
            parameter_value = forces[string_name]
        except:
            try:
                parameter_value = forces[reverse_string_name]
            except:
                print(
                    f"No {force} definition provided for '{string_name}', setting to {default_value}"
                )
                forces.update({string_name: default_value})
                forces.update({reverse_string_name: default_value})
                parameter_value = forces[string_name]

        return parameter_value
    
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

        particle_types = [
            self.get_particle_type_name(bond[0]),
            self.get_particle_type_name(bond[1]),
        ]

        return self._get_bonded_parameter(particle_types, 'bond_lengths')
        

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

        particle_types = [
            self.get_particle_type_name(bond[0]),
            self.get_particle_type_name(bond[1]),
        ]

        return self._get_bonded_parameter(particle_types, 'bond_force_constants')
    
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
            self.get_particle_type_name(angle[0]),
            self.get_particle_type_name(angle[1]),
            self.get_particle_type_name(angle[2]),
        ]

        return self._get_bonded_parameter(particle_types, 'equil_bond_angles')        

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
            self.get_particle_type_name(angle[0]),
            self.get_particle_type_name(angle[1]),
            self.get_particle_type_name(angle[2]),
        ]

        return self._get_bonded_parameter(particle_types, 'bond_angle_force_constants')

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
            self.get_particle_type_name(torsion[0]),
            self.get_particle_type_name(torsion[1]),
            self.get_particle_type_name(torsion[2]),
            self.get_particle_type_name(torsion[3]),
        ]

        return self._get_bonded_parameter(particle_types, 'torsion_periodicities')

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
            self.get_particle_type_name(torsion[0]),
            self.get_particle_type_name(torsion[1]),
            self.get_particle_type_name(torsion[2]),
            self.get_particle_type_name(torsion[3]),
        ]

        return self._get_bonded_parameter(particle_types, 'torsion_force_constants')

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
            self.get_particle_type_name(torsion[0]),
            self.get_particle_type_name(torsion[1]),
            self.get_particle_type_name(torsion[2]),
            self.get_particle_type_name(torsion[3]),
        ]

        return self._get_bonded_parameter(particle_types, 'equil_torsion_angles')
