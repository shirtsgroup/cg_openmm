import os
import pickle
import sys
from collections import Counter
from itertools import chain, combinations, product

from cg_openmm.build.cg_build import *
from cg_openmm.utilities.iotools import *
from cg_openmm.utilities.random_builder import get_random_positions
from openmm import unit


class CGModel(object):
    """
    Coarse-grained model class object containing:
    
        - particle and residue definitions
        - monomer sequence
        - bonded force field parameters
        - nonbonded force field parameters
        - initial particle positions
    
    :Example:
    .. code-block:: python
    
        from openmm import unit
        from cg_openmm.cg_model.cgmodel import CGModel
        
        # Specify backbone (bb) and sidechain (sc) particle parameters:
        sigma = 0.3 * unit.nanometer
        epsilon = 2 * unit.kilojoule_per_mole
        mass = 100 * unit.amu
        
        bb = {"particle_type_name": "bb", "sigma": sigma, "epsilon": epsilon, "mass": mass}
        sc = {"particle_type_name": "sc", "sigma": sigma, "epsilon": epsilon, "mass": mass}
        
        # Define monomer (residue):
        A = {
            "monomer_name": "A",
            "particle_sequence": [bb, sc],
            "bond_list": [[0, 1]],
            "start": 0,
            "end": 0}
        
        # Specify bonded parameters:
        bond_lengths = {
            "default_bond_length": 0.35 * unit.nanometer,
            "bb_bb_bb_bond_length": 0.40 * unit.nanometer}
            
        bond_force_constants = {
            "default_bond_force_constant": 1000 * unit.kilojoule_per_mole / unit.nanometer**2}
        
        equil_bond_angles = {
            "default_equil_bond_angle": 120.0 * unit.degrees,
            "bb_bb_bb_equil_bond_angle": 150.0 * unit.degrees}        
        
        bond_angle_force_constants = {
            "default_bond_angle_force_constant": 100.0 * unit.kilojoule_per_mole / unit.radian**2}

        torsion_phase_angles = {
            "default_torsion_phase_angle": 150 * unit.degrees}        
        
        torsion_force_constants = {
            "default_torsion_force_constant": 2.0 * unit.kilojoule_per_mole,
            "bb_bb_bb_bb_torsion_force_constant": 5.0 * unit.kilojoule_per_mole}

        torsion_periodicities = {
            "default_torsion_periodicity": 1}

        # Define oligomer sequence:
        sequence = 12 * [A]
        
        # Initial particle positions determined from random builder
        
        cgmodel = CGModel(
            particle_type_list=[bb, sc],
            bond_lengths=bond_lengths,
            bond_force_constants=bond_force_constants,
            bond_angle_force_constants=bond_angle_force_constants,
            torsion_force_constants=torsion_force_constants,
            equil_bond_angles=equil_bond_angles,
            torsion_phase_angles=torsion_phase_angles,
            torsion_periodicities=torsion_periodicities,
            include_nonbonded_forces=True,
            include_bond_forces=True,
            include_bond_angle_forces=True,
            include_torsion_forces=True,
            constrain_bonds=False,
            sequence=sequence,
            monomer_types=[A],
        )

    """

    def __init__(
        self,
        particle_type_list={},
        charges={},
        monomer_types=None,
        sequence=None,
        positions=None,
        bond_lengths={},
        bond_force_constants={},
        bond_angle_force_constants={},
        torsion_force_constants={},
        torsion_periodicities={},
        equil_bond_angles={},
        torsion_phase_angles={},
        binary_interaction_parameters={},
        go_model=False,
        go_repulsive_epsilon=None,
        constrain_bonds=False,
        include_nonbonded_forces=True,
        include_bond_forces=True,
        include_bond_angle_forces=True,
        include_torsion_forces=True,
        angle_style='harmonic',
        nonbond_repulsive_exp=12,
        nonbond_attractive_exp=6,
        exclusions={},
        hbonds={},
        rosetta_functional_form=False,
        check_energy_conservation=True,
        use_structure_library=False,
        random_positions=False,
        system=None,
        topology=None,
        simulation=None,
        ):

        """
        Initialize definitions for all of the attributes of a coarse-grained model

        .. Note:: Default definitions are applied to force field parameters not explicitly defined.

        :param positions: Positions for the particles in the coarse-grained model (default = None)
        :type positions: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit )

        :param particle_type_list: list of all particle types (default = None)
        :type particle_type_list: list 

        :param monomer_types: A list of dictionary objects containing the properties for unique monomer types used to construct a heteropolymeric coarse-grained model (default = None)
        :type monomer_types: List( dict( 'monomer_name': str, 'backbone_length': int, 'sidechain_length': int, 'sidechain_positions': List( int ), 'num_beads': int, 'bond_lengths': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'epsilons': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'sigmas': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) ) )        
        
        :param sequence: The sequence from which to build a heteropolymer.  Defined using a list of 'monomer_types', each of which contains the properties for that monomer (default = None (Homopolymer))
        :type sequence: List( dict( 'monomer_name': str, 'backbone_length': int, 'sidechain_length': int, 'sidechain_positions': List( int ), 'num_beads': int, 'bond_lengths': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'epsilons': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'sigmas': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) ) )        
        
        :param bond_lengths: Equilibrium bond lengths for all bond types (default = None)
        :type bond_lengths: dict( 'type_name1_type_name2_bond_length': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )

        :param bond_angle_force_constants: Harmonic bond angle-bending force constants for all bond types (default = None)
        :type bond_angle_force_constants: dict( 'type_name1_type_name2_type_name3_angle_k': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )

        :param bond_force_constants: Harmonic bond-stretching force constants for all bond types (default = None)
        :type bond_force_constants: dict( 'type_name1_type_name2_bond_k': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )

        :param equil_bond_angles: Equilibrium bond bending angle for all angle types (default = None)
        :type equil_bond_angles: dict('type_name1_type_name2_type_name3_angle_0': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ))

        :param torsion_force_constants: Torsion force constants for all torsion types (default = None)
        :type torsion_force_constants: dict( 'type_name1_type_name2_type_name3_type_name4_torsion_k': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ))

        :param torsion_phase_angles: Periodic torsion phase angle for all unique torsion angle definitions (default = 0)
        :type torsion_phase_angles: dict( 'type_name1_type_name2_type_name3_type_name4_torsion_0': `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) )

        :param binary_interaction_parameters: Binary interaction parameters used to scale nonbonded interactions between unlike particles (default=None)
        :type binary_interaction_parameters: dict( 'type_name1_type_name2_binary_interaction': float )
        
        :param go_model: If True, the binary interaction parameters will be applied only to the attractive component of the nonbonded potential. Otherwise, binary_interaction_parameters will be applied to the total potential (default=False)
        :type go_model: bool
        
        :param go_repulsive_epsilon: If not None and go_model=True, use a fixed value for repulsive interactions, applied only to pairs that also have a binary interaction parameters < 1. (default=None)
        :type go_repulsive_epsilon: Quantity ( float*unit.kilojoule_per_mole )
        
        :param constrain_bonds: Option to use rigid bond constaints during a simulation of the energy for the system (default = False)
        :type constrain_bonds: Bool

        :param include_bond_forces: Include bond stretching potentials when calculating the potential energy (default = True)
        :type include_bond_forces: Bool

        :param include_nonbonded_forces: Include nonbonded interactions when calculating the potential energy (default = True)
        :type include_nonbonded_forces: Bool

        :param include_bond_angle_forces: Include contributions from bond angle forces when calculating the potential energy (default = True)
        :type include_bond_angle_forces: Bool

        :param include_torsion_forces: Include contributions from torsions when calculating the potential energy (default = True)
        :type include_torsion_forces: Bool
        
        :param angle_style: Functional form to use for bond-bending angle potential ('harmonic', 'restricted', or 'cosine') (default='harmonic')
        :type angle_style: str  

        :param nonbond_repulsive_exp: Repulsive exponent for custom nonbonded Mie potential. For now this same exponent is applied to all pair types. (default=12)
        :type nonbond_repulsive_exp: float
        
        :param nonbond_attractive_exp: Attractive exponent for custom nonbonded Mie potential. For now this same exponent is applied to all pair types. (default=6)
        :type nonbond_attractive_exp: float
        
        :param exclusions: Nonbonded weights for [1-2, 1-3, 1-4] interactions (default = [0,0,1])
        :type exclusions: dict( list( int ) )
        
        :param hbonds: Dictionary containing directional CustomHbondedForce potential information. 
        :type hbonds: dict( 'donors': list(); 'acceptors': list(); 'epsilon_hb': Quantity * unit.kilojoule_per_mole; 'sigma_hb': Quantity * unit.angstrom; 'theta_d': Quantity * unit.degrees; 'theta_a': Quantity * unit.degrees )  
        
        :param rosetta_functional_form: Option to use nonbonded exclusions consistent with Rosetta
        :type rosetta_functional_form: Bool

        :param check_energy_conservation: Flag designating whether or not to perform a test OpenMM simulation with this coarse-grained model (default = True).
        :type check_energy_conservation: Bool

        :param use_structure_library: Flag designating whether or not to use a structure from the foldamers ensemble as the initial positions for the particles in the coarse-grained model (default = False)
        :type use_structure_library: Bool

        :param random_positions: Flag designating whether or not to generate a set of random coordinates for the coarse-grained model (default = None)
        :type random_positions: Bool
        
        :param system: OpenMM System() object, which stores the forces for the coarse grained model (default = None)
        :type system: `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_

        :param topology: OpenMM Topology() object, which stores bonds, angles, and torsions coarse-grained model (default = None)
        :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_
        
        :param simulation: OpenMM Simulation() object (default = None)
        :type simulation: `Simulation() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1simulation_1_1Simulation.html>`_        

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
        self.default_exclusion_rules = [0,0,1]

        # Initialize user-defined input:

        # Assign forces based upon input flags
        self.rosetta_functional_form = rosetta_functional_form
        self.include_bond_forces = include_bond_forces
        self.constrain_bonds = constrain_bonds
        self.include_bond_angle_forces = include_bond_angle_forces
        self.include_nonbonded_forces = include_nonbonded_forces
        self.include_torsion_forces = include_torsion_forces
        self.angle_style = angle_style
        self.nonbond_repulsive_exp = nonbond_repulsive_exp
        self.nonbond_attractive_exp = nonbond_attractive_exp
        self.check_energy_conservation = check_energy_conservation
        self.monomer_types = monomer_types
        self.bond_lengths = bond_lengths

        # Validate bonded force and exclusion input
        self.bond_force_constants = bond_force_constants
        self.bond_angle_force_constants = bond_angle_force_constants
        self.equil_bond_angles = equil_bond_angles
        self.torsion_force_constants = torsion_force_constants
        self.torsion_periodicities = torsion_periodicities
        self.torsion_phase_angles = torsion_phase_angles
        self.exclusions = exclusions
        self._validate_bonded_forces()

        # fill in defaults in particle list
        self.particle_type_list = self._validate_particle_type_list(particle_type_list)

        # Build a polymer with these model settings
        self.build_polymer(sequence)

        # Assign particle properties
        self.particle_types = add_new_elements(self)
        
        # Assign binary interaction parameters
        self.go_model = go_model
        self.go_repulsive_epsilon = go_repulsive_epsilon
        self.binary_interaction_parameters = binary_interaction_parameters
        self._validate_binary_interaction()
        
        # Assign directional H-bond parameters:
        self.hbonds = hbonds
        self._validate_hbonds()

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
        Export a cgmodel to a pickle file.

        :param filename: filename for exported cgmodel
        :type filename: str
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
            "bond_lengths": {
                "default_name": "default_bond_length",
                "default_value": self.default_length,
                "suffix": "bond_length",
            },
            "bond_force_constants": {
                "default_name": "default_bond_force_constant",
                "default_value": self.default_bond_force_constant,
                "suffix": "bond_force_constant",
            },
            "equil_bond_angles": {
                "default_name": "default_equil_bond_angle",
                "default_value": self.default_angle,
                "suffix": "equil_bond_angle",
            },
            "bond_angle_force_constants": {
                "default_name": "default_bond_angle_force_constant",
                "default_value": self.default_bond_angle_force_constant,
                "suffix": "bond_angle_force_constant",
            },
            "torsion_phase_angles": {
                "default_name": "default_torsion_phase_angle",
                "default_value": self.default_angle,
                "suffix": "torsion_phase_angle",
            },
            "torsion_force_constants": {
                "default_name": "default_torsion_force_constant",
                "default_value": self.default_torsion_force_constant,
                "suffix": "torsion_force_constant",
            },
            "torsion_periodicities": {
                "default_name": "default_torsion_periodicity",
                "default_value": self.default_periodicity,
                "suffix": "torsion_periodicity",
            },
            "exclusions": {
                "default_name": "default_exclusions",
                "default_value": self.default_exclusion_rules,
                "suffix": "exclusions",
            },
        }

        # make sure all the property values are internally consistent
        for attribute in self.bonded_force_attributes:
            # for the bonded force attributes
            if hasattr(self, attribute):
                properties = self.bonded_force_attributes[attribute]
                default_name = properties["default_name"]
                # if the default name hasn't been defined for this model
                if default_name not in getattr(self, attribute):
                    default_value = properties["default_value"]
                    # set it to the the default for the program.
                    print(f"Warning: No {default_name}: setting to {default_value}")
                    parameter_dict = getattr(self,attribute)
                    # actually add the default force to the dictionary.
                    default_dict = {default_name : default_value}
                    parameter_dict.update(default_dict)
                    setattr(self,attribute,parameter_dict)
                default_suffix = properties["suffix"]
                for force in getattr(self, attribute):
                    # make sure all forces have the corresponding suffix.
                    if ('_'+default_suffix) not in force:
                        print(
                            f"Warning: force term '{force}' does not have proper suffix of {default_suffix}"
                        )
                        exit()
                        

    def _validate_particle_type_list(self, particle_type_list):
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
                print(f'Particle has no attribute "particle_type_name": Exiting now')
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
                    f"charge not defined for particle type {name} using default charge: {self.default_charge}"
                )
                particle["charge"] = self.default_charge

        return particle_type_list
        
        
    def _validate_binary_interaction(self):    
        """
        Check that the binary interaction definitions are valid.
        Each entry in the dictionary should be 'type_name1_type_name2_binary_interaction': float
        """
        
        if self.binary_interaction_parameters:
            for key, value in self.binary_interaction_parameters.items():
                # Extract the particle types:
                kappa_list = []
                string = ""
                for c in key:
                    if c == '_':
                        kappa_list.append(string)
                        string = ""
                    else:
                        string += c
                kappa_list.append(string)
                
                if kappa_list[-2] != 'binary' or kappa_list[-1] != 'interaction':
                    print(f'Incorrect suffix for binary interaction parameter for {self.binary_interaction_parameters[key]}')
                    exit()
  
                # Use the first two particles in the dictionary key:
                kappa_particle_list = kappa_list[0:2]
                
                # Get the names of all particles in particle_type_list:
                particle_strings = []
                for particle in self.particle_type_list:
                    particle_strings.append(particle['particle_type_name'])
                
                for particle in kappa_particle_list:
                    if particle in particle_strings:
                        pass
                    else:
                        print(f'Invalid particle name {particle} in binary interaction parameter definition') 
                        exit()
                        
        return
        
        
    def _validate_hbonds(self):    
        """
        Check that the direction hydrogen bond force definitions are valid.
        The hbond dictionary must have the following entries and types:
        'donors': list of donor residue indices,
        'acceptors: list of acceptor residue indices,
        'epsilon_hb': Quantity with energy units (hydrogen bond interaction strength between contacts)
        'sigma_hb': Quantity with distance units (size parameter for hydrogen bond potential)
        'theta_d': Quantity with angle units (angle between (sc_donor-bb_donor)---(bb_acceptor))
        'theta_a': Quantity with angle units (angle between (bb_donor)---(bb_acceptor-sc_acceptor))
        """
        
        if self.hbonds:
            hbonds = self.hbonds
            # Check for required parameter keys:
            for param in ['donors','acceptors','epsilon_hb','sigma_hb','theta_d','theta_a']:
                if param in hbonds:
                    pass
                else:
                    print(f'Error: incomplete hbond potential parameters provided (missing {param})')
                    exit()
                
            # Check acceptor and donor lists:
            if (type(hbonds['donors']) == list and type(hbonds['acceptors']) == list and \
                len(hbonds['donors']) == len(hbonds['acceptors'])):
                pass
            else:
                print(f'Error: invalid hbond donor and acceptor residue lists')
                exit()
                
            # Check hbond parameters:
            for param in ['epsilon_hb','sigma_hb','theta_d','theta_a']:
                if type(hbonds[param]) == unit.quantity.Quantity:
                    pass
                else:
                    print(f'Error: invalid {param} parameter - check units')
                    exit()
         
        return        
        
        
    def build_polymer(self, sequence):
        """
        Used to build a polymer, or reset the properties for a polymer after parameters such as the polymer_length or sequence have been modified.
        
        :param sequence: The sequence from which to build a heteropolymer.  Defined using a list of 'monomer_types', each of which contains the properties for that monomer (default = None (Homopolymer))
        :type sequence: List( dict( 'monomer_name': str, 'backbone_length': int, 'sidechain_length': int, 'sidechain_positions': List( int ), 'num_beads': int, 'bond_lengths': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'epsilons': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'sigmas': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) ) )        
        
        :returns:
           - self.nonbonded_interaction_list ( List( List( int, int ) ) ) - A list of the nonbonded interactions (which don't violate exclusion rules) in the coarse-grained model
        """
        
        self.polymer_length = len(sequence)
        self.sequence = sequence
        self.process_monomer_types()
        self.num_beads = self.get_num_beads()
        self.particle_list = self.create_particle_list()

        # Get the number of unique particle types:
        n_particle_types = len(self.particle_type_list)
        
        # Get the number of unique nonbonded pair types:
        n_pair_types = 0
        for i in range(1,n_particle_types+1):
            n_pair_types += i

        if self.include_bond_forces or self.constrain_bonds:
            self.bond_list = self.get_bond_list()
        else:
            # Check if any bond will have zero interaction:

            # Bond forces are excluded using rosetta_functional_form, which overrides any other specified exclusion rules:
            if self.rosetta_functional_form:
                print(f"Error: bonded particles must have either bond forces or 1-2 nonbonded forces")
                exit()
                
            # Default exclusions have no bond forces and not all explicit pair types were specified:
            # (default will always be part of the exclusions dict)
            if self.exclusions["default_exclusions"][0] == 0 and len(self.exclusions) < n_pair_types+1:
                print(f"Error: bonded particles must have either bond forces or 1-2 nonbonded forces")
                exit()
                
            # At least one specific pair type has no 1-2 nonbonded interactions:
            for key,value in self.exclusions.items():
                if key != "default_exclusions":
                    if value[0] == 0:
                        print(f"Error: bonded particles must have either bond forces or 1-2 nonbonded forces")
                        exit()
                        
            self.bond_list = []
            
        # Check for missing 1-3 interactions:
        if self.include_bond_angle_forces == False:
            if self.rosetta_functional_form:
                print(f"Warning: there are no 1-3 nonbonded or angle forces defined")
            
            # Default exclusions have no angle forces and not all explicit pair types were specified: 
            if self.exclusions["default_exclusions"][1] == 0 and len(self.exclusions) < n_pair_types+1:
                print(f"Warning: at least one pair type has no 1-3 nonbonded or angle forces defined")
                
            # At least one specific pair type has no 1-3 nonbonded interactions:
            for key,value in self.exclusions.items():
                if key != "default_exclusions":
                    if value[1] == 0:
                        print(f"Warning: at least one pair type has no 1-3 nonbonded or angle forces defined")
            
            
        # Check for missing 1-4 interactions:
        if self.include_torsion_forces == False:
            if self.rosetta_functional_form:
                print(f"Warning: there are no 1-4 nonbonded or torsion forces defined")
            
            # Default exclusions have no torsion forces and not all explicit pair types were specified: 
            if self.exclusions["default_exclusions"][2] == 0 and len(self.exclusions) < n_pair_types+1:
                print(f"Warning: at least one pair type has no 1-4 nonbonded or torsion forces defined")
                
            # At least one specific pair type has no 1-4 nonbonded interactions:
            for key,value in self.exclusions.items():
                if key != "default_exclusions":
                    if value[2] == 0:
                        print(f"Warning: at least one pair type has no 1-4 nonbonded or torsion forces defined")

        self.bond_angle_list = self.get_bond_angle_list()
        self.torsion_list = self.get_torsion_list()
        
        # Returns an empty list if no exclusions specified:
        self.nonbonded_exclusion_list = self.get_nonbonded_exclusion_list(
            rosetta_functional_form=self.rosetta_functional_form
        )

        self.nonbonded_interaction_list = self.get_nonbonded_interaction_list()
        

    def process_monomer_types(self):
        """
        Clean up a list of 'monomer_types' for all unique monomers.
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
                    if i in (bond[0], bond[1]):
                        unbonded = False
                        break
                if unbonded == False:
                    break

            if unbonded:
                print(f"Error: particle {i} in monomer {mm} has no bonds.")
                exit()

                
    def get_num_beads(self):
        """
        Calculate the number of beads in a coarse-grained model class object

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
                if "particle_type_name" not in bead:
                    print("'particle_type_name' not defined, cannot contiue")
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
                    bead_index = bead_index + 1  # increment for bookkeeping
                bond_list += monomer_bond_list
        return bond_list
        

    def get_nonbonded_interaction_list(self):
        """
        Construct a nonbonded interaction list for the coarse-grained model

        :returns: 
           - interaction_list ( List( List( int, int ) ) ) - A list of the nonbonded interactions (which don't violate exclusion rules) in the coarse-grained model
        """

        interaction_list = []

        # First, include all pairs. Then remove any excluded pairs.
        for particle_1 in range(self.num_beads):
            for particle_2 in range(particle_1+1, self.num_beads):
                if ([particle_1, particle_2] not in interaction_list and \
                    [particle_2, particle_1] not in interaction_list):
                    interaction_list.append([particle_1, particle_2])
                        
        exclusion_list = self.nonbonded_exclusion_list
        
        for exclusion in exclusion_list:
            if exclusion in interaction_list:
                interaction_list.remove(exclusion)
            if [exclusion[1], exclusion[0]] in interaction_list:
                interaction_list.remove([exclusion[1], exclusion[0]])

        return interaction_list
        

    def get_nonbonded_exclusion_list(self, rosetta_functional_form=False):
        """
        Get a list of the nonbonded interaction exclusions, which are assigned if two particles are separated by less than three bonds

        :param rosetta_functional_form: Option to use nonbonded exclusions consistent with Rosetta
        :type rosetta_functional_form: Bool

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

        if rosetta_functional_form:
            # Exclude all bonds:
            for bond in self.bond_list:
                if bond not in exclusion_list and bond.reverse() not in exclusion_list:
                    exclusion_list.append(bond)
                    
            # Exclude all angles:
            for angle in self.bond_angle_list:
                angle_ends = [angle[0], angle[2]]
                if angle_ends not in exclusion_list and angle_ends.reverse() not in exclusion_list:
                    exclusion_list.append(angle_ends)
                    
            # Exclude all torsions:
            for torsion in self.torsion_list:
                torsion_ends = [torsion[0], torsion[3]]
                if (
                    torsion_ends not in exclusion_list
                    and torsion_ends.reverse() not in exclusion_list
                ):
                    exclusion_list.append(torsion_ends)     
                    
        else:
            # Check for pair-specific exclusions:
            for bond in self.bond_list:
                exclusion_rules = self.get_exclusions(bond)
                if exclusion_rules[0] == 0:
                    if bond not in exclusion_list and bond.reverse() not in exclusion_list:
                        exclusion_list.append(bond)
        
            for angle in self.bond_angle_list:
                angle_ends = [angle[0], angle[2]]
                exclusion_rules = self.get_exclusions(angle_ends)
                if exclusion_rules[1] == 0:
                    if angle_ends not in exclusion_list and angle_ends.reverse() not in exclusion_list:
                        exclusion_list.append(angle_ends)
                    
            for torsion in self.torsion_list:
                torsion_ends = [torsion[0], torsion[3]]
                exclusion_rules = self.get_exclusions(torsion_ends)
                if exclusion_rules[2] == 0:
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
        

    def get_torsion_list(self):  # MRS: really slow, should be looked at.
        """
        Construct a list of particle indices from which to define torsions for the coarse-grained model

        :returns: 
           - torsions ( List( List( int, int, int, int ) ) ) - A list of the particle indices for the torsions in the coarse-grained model
        """

        angle_list = self.bond_angle_list
        torsion_list = []
        
        # New method - just use two overlapping angles:
        for i in range(len(angle_list)):
            for j in range(i,len(angle_list)):
                angle_1 = angle_list[i]
                angle_2 = angle_list[j]

                # Check overlap:
                if [angle_1[1],angle_1[2]] == [angle_2[0],angle_2[1]]:
                    # 0 1 2
                    #   0 1 2
                    torsion = [angle_1[0], angle_1[1], angle_1[2], angle_2[2]]
                    # Check that torsion is new and that ends are not the same particle:
                    if torsion not in torsion_list and reversed(torsion) not in torsion_list and torsion[0] != torsion[3]:
                        torsion_list.append(torsion)
                        
                elif [angle_1[1],angle_1[2]] == [angle_2[2],angle_2[1]]:
                    # 0 1 2
                    #   2 1 0
                    torsion = [angle_1[0], angle_1[1], angle_1[2], angle_2[0]]
                    if torsion not in torsion_list and reversed(torsion) not in torsion_list and torsion[0] != torsion[3]:
                        torsion_list.append(torsion)
                        
                elif [angle_1[1],angle_1[0]] == [angle_2[0],angle_2[1]]:
                    # 2 1 0
                    #   0 1 2
                    torsion = [angle_1[2], angle_1[1], angle_1[0], angle_2[2]]
                    if torsion not in torsion_list and reversed(torsion) not in torsion_list and torsion[0] != torsion[3]:
                        torsion_list.append(torsion)
                        
                elif [angle_1[1],angle_1[0]] == [angle_2[2],angle_2[1]]:
                    # 2 1 0
                    #   2 1 0
                    torsion = [angle_1[2], angle_1[1], angle_1[0], angle_2[0]]
                    if torsion not in torsion_list and reversed(torsion) not in torsion_list and torsion[0] != torsion[3]:
                        torsion_list.append(torsion)
        
        return torsion_list
        

    def get_particle_attribute(self, particle, attribute):
        """
        Get various attributes of a particle, given either the index or the particle dictionary
        
        :param particle: Index of the particle of interest OR particle dictionary
        :type particle: int or dict()
        
        :param attribute: options are "monomer", "monomer_type", "name", "index", "type", "mass", "charge", "epsilon", "sigma", "particle_type_name"
        :type attribute: str
        
        :returns:
           - attribute of interest

        """

        if attribute in ["monomer", "monomer_type", "name", "index", "type"]:
            # these are attributes of the particles in the list
            if type(particle) == dict:
                return particle[attribute]
            elif type(particle) == int:
                return self.particle_list[particle][attribute]
        elif attribute in ["mass", "charge", "epsilon", "sigma", "particle_type_name"]:
            # these are attributes of the partilce type
            if type(particle) == dict:
                return particle["type"][attribute]
            elif type(particle) == int:
                return self.particle_list[particle]["type"][attribute]
        return
        

    def get_particle_name(self, particle):
        """
        Returns the name of a particle, given its index within the model

        :param particle_index: Index of the particle for which we would like to determine the type
        :type particle_index: int

        :returns:
           - particle_name ( str ) - The name of the particle
        """
        
        return self.get_particle_attribute(particle, "name")

        
    def get_particle_index(self, particle):
        """
        Returns the index of a particle, given its index within the model or the particle dictionary. Obviously,
        kind of redundant if using the index instead of the particle dictionary

        :param particle_index: Index of the particle for which we would like to determine the type
        :type particle_index: int

        :returns: 
           - particle_name ( str ) - The name of the particle
        """
        
        return self.get_particle_attribute(particle, "index")
        

    def get_particle_type(self, particle):
        """
        Gives the type of a particle (a dictionary)

        :param particle: Index of the particle for which we would like to determine the type OR particle dictionary
        :type particle: int or dict()

        :returns: 
           - particle_type (str):
        """

        return self.get_particle_attribute(particle, "type")
        

    def get_particle_type_name(self, particle):
        """
        Gives the type name of a particle.

        :param particle_index: Index of the particle for which we would like to determine the type name OR particle dictionary
        :type particle_index: int or dict()

        :returns: 
           - particle_type_name (str):
        """

        return self.get_particle_attribute(particle, "particle_type_name")
        

    def get_particle_monomer_type(self, particle):
        """
        Indicates which type of monomer a particle belongs to

        :param particle_index: Index of the particle for which we would like to determine the monomer type
        :type particle_index: int

        :returns: 
           - monomer_type (dict) : monomer type
        """
          
        return self.get_particle_attribute(particle, "monomer_type")
        

    def get_particle_monomer(self, particle):
        """
        Indicates which monomer index a particle belongs to

        :param particle_index: Index of the particle for which we would like to determine the monomer type
        :type particle_index: int

        :returns: 
           - monomer_type (dict) : monomer type

        """
        
        return self.get_particle_attribute(particle, "monomer")
        

    def get_particle_mass(self, particle):
        """
        Returns the mass of a particle, given its index within the coarse-grained model or the particle dictionary

        :param particle: Index of the particle for which we would like to determine the type, or dict()
        :type particle: int or dict()

        :returns: 
           - epsilon ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned Lennard-Jones epsilon value for the provided particle index
        """

        return self.get_particle_attribute(particle, "mass")
        

    def get_particle_charge(self, particle):
        """
        Returns the charge for a particle, given its index within the coarse-grained model, or the dict

        :param particle_index: Index of the particle for which we would like to determine the type
        :type particle_index: int

        :returns: 
           - particle_charge ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The charge for the provided particle index

        """
        return self.get_particle_attribute(particle, "charge")
        

    def get_particle_sigma(self, particle):

        """
        Returns the Lennard-Jones potential sigma value for a particle, given the particle index

        :param particle_index: Index of the particle for which we would like to determine the type
        :type particle_index: int

        :returns: 
           - sigma ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned Lennard-Jones sigma value for the provided particle index

        """
        return self.get_particle_attribute(particle, "sigma")
        

    def get_particle_epsilon(self, particle):
        """
        Returns the Lennard-Jones epsilon value for a particle, given its index within the coarse-grained model.

        :param particle_index: Index of the particle for which we would like to determine the type
        :type particle_index: int

        :returns: 
           - epsilon ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned Lennard-Jones epsilon value for the provided particle index
        """

        return self.get_particle_attribute(particle, "epsilon")

        
    def _get_bonded_parameter(self, particle_types, force):

        """
        internal function for returning any force value.

        parameters: the string name of the bonded force of interest

        returns: the value of the parameter for the atoms involved in the interaction
        """

        # get the details for this force
        properties = self.bonded_force_attributes[force]
        suffix = properties["suffix"]

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
        forces = getattr(self, force)

        default_name = properties["default_name"]
        default_value = forces[default_name]

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

        :param bond: A list of the indices for the particles in a bond
        :type bond: List ( int )

        :returns: 
           - bond_length ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned bond length for the provided particles

        """

        particle_types = [
            self.get_particle_type_name(bond[0]),
            self.get_particle_type_name(bond[1]),
        ]

        return self._get_bonded_parameter(particle_types, "bond_lengths")

        
    def get_bond_force_constant(self, bond):
        """
        Determines the correct bond force constant for two particles, given their indices

        :param bond: A list of the indices for the particles in a bond
        :type bond: List ( int )
        
        :returns: 
           - bond_force_constant ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned bond force constant for the provided particles

        """

        particle_types = [
            self.get_particle_type_name(bond[0]),
            self.get_particle_type_name(bond[1]),
        ]

        return self._get_bonded_parameter(particle_types, "bond_force_constants")
        

    def get_equil_bond_angle(self, angle):
        """
        Determines the correct equilibrium bond angle between three particles, given their indices within the coarse-grained model

        :param angle: A list of the indices for the particles in an angle
        :type angle: List ( int )
        
        :returns: 
           - equil_bond_angle (float) - The assigned equilibrium bond angle for the provided particles

        """

        particle_types = [
            self.get_particle_type_name(angle[0]),
            self.get_particle_type_name(angle[1]),
            self.get_particle_type_name(angle[2]),
        ]

        return self._get_bonded_parameter(particle_types, "equil_bond_angles")
        

    def get_bond_angle_force_constant(self, angle):
        """
        Determines the correct bond angle force constant for a bond angle between three particles, given their indices within the coarse-grained model

        :param angle: A list of the indices for the particles in an angle
        :type angle: List ( int )

        :returns: 
           - bond_angle_force_constant ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned bond angle force constant for the provided particles
        """
        
        particle_types = [
            self.get_particle_type_name(angle[0]),
            self.get_particle_type_name(angle[1]),
            self.get_particle_type_name(angle[2]),
        ]

        return self._get_bonded_parameter(particle_types, "bond_angle_force_constants")
        

    def get_torsion_periodicity(self, torsion):
        """         
        Determines the periodicity for a torsion, given a quadruplet of particle indices. 
        For sums of periodic torsions, this returns a list.
        
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

        return self._get_bonded_parameter(particle_types, "torsion_periodicities")
        

    def get_torsion_force_constant(self, torsion):
        """         
        Determines the correct torsion force constant for a torsion (bond angle involving four particles), given their indices within the coarse-grained model.
        For sums of periodic torsions, this returns a list.
        
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

        return self._get_bonded_parameter(particle_types, "torsion_force_constants")
        

    def get_torsion_phase_angle(self, torsion):
        """         
        Determines the phase_angle for a torsion, given indices of the 4 particles within the coarse-grained model
        For sums of periodic torsions, this returns a list.
        
        :param torsion: A list of the indices for the particles in a torsion
        :type torsion: List( int )

        :returns: 
           - torsion_phase_angle (float) - The assigned periodic torsion phase angle for the provided particles
        """
        particle_types = [
            self.get_particle_type_name(torsion[0]),
            self.get_particle_type_name(torsion[1]),
            self.get_particle_type_name(torsion[2]),
            self.get_particle_type_name(torsion[3]),
        ]

        return self._get_bonded_parameter(particle_types, "torsion_phase_angles")
        
        
    def get_exclusions(self, pair):
        """
        Gets the exclusion rules applied to this specific pair of particles.
        
        :param pair: A list of 2 indices defining a pair
        :type pair: List ( int )

        :returns:
           - exclusion_rules (list(int)) - List of exclusions for [1-2, 1-3, 1-4] nonbonded interactions (0 = no interaction)
        """

        particle_types = [
            self.get_particle_type_name(pair[0]),
            self.get_particle_type_name(pair[1]),
        ]

        return self._get_bonded_parameter(particle_types, "exclusions")  
           