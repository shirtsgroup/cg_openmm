from simtk import unit
import sys, os
from collections import Counter
from simtk import openmm as mm
from simtk.openmm.app.pdbfile import PDBFile
from simtk.openmm.app.topology import Topology
from simtk.openmm.app.topology import Residue
import simtk.openmm.app.element as elem
import foldamers
from foldamers.utilities import util
from cg_openmm.build.cg_build import *
from itertools import chain, combinations, product

def basic_cgmodel(polymer_length=12,backbone_length=1,sidechain_length=1,sidechain_positions=[0],mass=100.0 * unit.amu,bond_length=0.75 * unit.nanometer,sigma=1.85*unit.nanometer,epsilon=0.5 * unit.kilocalorie_per_mole,positions=None):

        """
        :param polymer_length: Number of monomer units, default = 8
        :type polymer_length: int

        :param backbone_length: Number of beads in the backbone for individual monomers within a coarse grained model, default = 1
        :type backbone_length: int

        :param sidechain_length: Number of beads in the sidechain for individual monomers within a coarse grained model, default = 1
        :type sidechain_length: int

        :param sidechain_positions: Designates the indices of backbone beads upon which we will place sidechains, default = [0] (add a sidechain to the first backbone bead in each monomer)
        :type sidechain_positions: List( int )

        :param mass: Mass for all coarse grained beads, default = 100.0 * unit.amu
        :type mass: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :param bond_length: Defines the length for all bond types, default = 7.5 * unit.angstrom
        :type bond_length: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_
             
        :param sigma: Lennard-Jones equilibrium interaction distance (by default, calculated for particles that are separated by 3 or more bonds), default = 18.5 * bond_length (for all interaction types)
        :type sigma: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :param epsilon: Lennard-Jones equilibrium interaction energy (by default, calculated for particles that are separated by 3 or more bonds), default = 0.5 * unit.kilocalorie_per_mole
        :type espilon: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :param positions: Positions for coarse grained particles in the model, default = None
        :type positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit )

        :returns: 
          - cgmodel (class) - CGModel() class object

        .. warning:: this function has significant limitations, in comparison with building a coarse grained model with the CGModel() class.  In particular, this function makes it more difficult to build heteropolymers, and is best-suited for the simulation of homopolymers.

        :Example:

        >>> from simtk import unit
        >>> polymer_length = 20
        >>> backbone_length = 1
        >>> sidechain_length = 1
        >>> sidechain_positions = [0]
        >>> mass = 100.0 * unit.amu
        >>> bond_length=0.75 * unit.nanometer
        >>> sigma=1.85*unit.nanometer
        >>> epsilon=0.5 * unit.kilocalorie_per_mole
        >>> cgmodel = basic_cgmodel(polymer_length=polymer_length,backbone_length=backbone_length,sidechain_length=sidechain_length,sidechain_positions=sidechain_positions,mass=mass,bond_length=bond_length,sigma=sigma,epsilon=epsilon) 

        """
        backbone_lengths = [backbone_length] # Number of backbone beads in unique monomer types
        sidechain_lengths = [sidechain_length] # Number of sidechain beads in unique monomer types
        masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass} # List of bead masses
        sigmas = {'bb_bb_sigma': sigma,'bb_sc_sigma': sigma,'sc_sc_sigma': sigma} # Lennard-Jones interaction distances.  List of unique interaction types
        bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length} # bond length
        epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': epsilon} # Lennard-Jones interaction strength.  List of unique interaction types
        cgmodel = CGModel(positions=positions,polymer_length=polymer_length,backbone_lengths=backbone_lengths, sidechain_lengths=sidechain_lengths, sidechain_positions = sidechain_positions, masses = masses, sigmas = sigmas, epsilons = epsilons, bond_lengths = bond_lengths)
        return(cgmodel)


class CGModel(object):
        """

        Build a coarse grained model class object.

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
        >>> cgmodel = CGModel(heteropolymer=True,monomer_types=monomer_types,sequence=sequence)

        """
        _BUILT_IN_REGIONS = ('polymer_length','backbone_lengths','sidechain_lengths','sidechain_positions','masses','sigmas','epsilons','bond_lengths','bond_force_constants','bond_angle_force_constants','torsion_force_constants','equil_torsion_angles','equil_bond_angles','charges','num_beads','positions','system','topology','simulation','constrain_bonds','bond_list','nonbonded_interaction_list','nonbonded_exclusion_list','bond_angle_list','torsion_list','include_bond_forces','include_nonbonded_forces','include_bond_angle_forces','include_torsion_forces','use_structure_library','check_energy_conservation')

        def __init__(self,
                     positions=None,
                     polymer_length=12,
                     backbone_lengths=[1],
                     sidechain_lengths=[1],
                     sidechain_positions=[0],
                     masses=None, 
                     sigmas=None,
                     epsilons=None, 
                     bond_lengths=None, 
                     bond_force_constants=None, 
                     bond_angle_force_constants=None, 
                     torsion_force_constants=None, 
                     equil_bond_angles=None,
                     equil_torsion_angles=None, 
                     charges=None, 
                     constrain_bonds=True,
                     include_bond_forces=False,
                     include_nonbonded_forces=True,
                     include_bond_angle_forces=True,
                     include_torsion_forces=True,
                     exclusions=True,
                     check_energy_conservation=True,
                     use_structure_library=False,
                     heteropolymer=False,
                     monomer_types=None,
                     sequence=None,
                     random_positions=False,
                     system=None,
                     topology=None):

          """
          Initialize definitions for all of the properties of a coarse grained model

          ..warning:: A large number of default definitions are applied when constructing the CGModel object, in an effort to make intiation of the class easier.  Please review these defaults (listed below) carefully before using this class.

          :param positions: Positions for the particles in the coarse grained model, default = None
          :type positions: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit )

          :param polymer_length: Number of monomer units, default = 12
          :type polymer_length: integer

          :param backbone_lengths: List of the number of beads in the backbone for unique monomer types within the coarse grained model, default = [1]
          :type backbone_lengths: List( int )

          :param sidechain_lengths: List of the number of beads in the sidechain for unique monomer types within the coarse grained model, default = [1]
          :type sidechain_lengths: List( int )

          :param sidechain_positions: List of the indices of backbone beads upon which we will place sidechains, default = [0] (add a sidechain to the first backbone bead in each monomer)
          :type sidechain_positions: List( integer )

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

          :param num_beads: Total number of particles in the coarse grained model, default = None (The total number of particles in a length=12 1-1 coarse-grained model)
          :type num_beads: int

          :param system: OpenMM System() object, which stores the forces for the coarse grained model, default = None
          :type system: `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_

          :param topology: OpenMM Topology() object, which stores bonds, angles, and other structural attributes of the coarse grained model, default = None
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

          :param check_energy_conservation: Flag designating whether or not to perform a test OpenMM simulation with this coarse grained model.
          :type check_energy_conservation: Logical

          :param use_structure_library: Flag designating whether or not to use a structure from the foldamers ensemble as the initial positions for the particles in the coarse grained model, default = False
          :type use_structure_library: Logical

          :param heteropolymer: Flag designating whether or not to build the coarse grained model from multiple monomer types, default = False
          :type heteropolymer: Logical

          :param monomer_types: A list of dictionary objects containing the properties for unique monomer types (used to construct a heteropolymeric coarse grained model, default = None
          :type monomer_types: List( dict( 'monomer_name': str, 'backbone_length': int, 'sidechain_length': int, 'sidechain_positions': List( int ), 'num_beads': int, 'bond_lengths': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'epsilons': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'sigmas': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) ) )

          :param sequence: The sequence from which to build a heteropolymer.  Defined using a list of 'monomer_types', each of which contains the properties for that monomer, default = None (Homopolymer)
          :type sequence: List( dict( 'monomer_name': str, 'backbone_length': int, 'sidechain_length': int, 'sidechain_positions': List( int ), 'num_beads': int, 'bond_lengths': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'epsilons': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'sigmas': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) ) )

          :param random_positions: Flag designating whether or not to generate a set of random coordinates for the coarse grained model, default = None

          """
          if bond_force_constants == None:
            bond_force_constants = {'bb_bb_bond_k': 1250.0,'bb_sc_bond_k': 1250.0, 'sc_sc_bond_k': 1250.0}
          if bond_angle_force_constants == None:
            bond_angle_force_constants={'bb_bb_bb_angle_k': 0.0002,'bb_bb_sc_angle_k': 0,'bb_sc_sc_angle_k': 0,'sc_sc_sc_angle_k': 0.0002}
          if torsion_force_constants == None:
            torsion_force_constants={'bb_bb_bb_bb_torsion_k': 0.0002,'bb_bb_bb_sc_torsion_k': 0,'bb_bb_sc_sc_torsion_k': 0, 'bb_sc_sc_sc_torsion_k': 0, 'sc_bb_bb_sc_torsion_k': 0, 'sc_sc_sc_sc_torsion_k': 0, 'sc_bb_bb_bb_torsion_k': 0}
          if equil_bond_angles == None:
            equil_bond_angles = {'bb_bb_bb_angle_0': 1.61,'bb_bb_sc_angle_0': 1.61}
          if equil_torsion_angles == None:
            equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': 0.91,'bb_bb_bb_sc_torsion_0': 0,'bb_bb_sc_sc_torsion_0': 0.0, 'bb_sc_sc_sc_torsion_0': 0.0, 'sc_bb_bb_sc_torsion_0': 0.0, 'bb_sc_sc_bb_torsion_0': 0.0, 'sc_sc_sc_sc_torsion_0': 0.0, 'sc_bb_bb_bb_torsion_0': 0}
          if charges == None:
            charges = {'backbone_bead_charges': 0.0 * unit.elementary_charge,'sidechain_bead_charges': 0.0 * unit.elementary_charge}
          if masses == None:
            masses = {'backbone_bead_masses': 100.0 * unit.amu, 'sidechain_bead_masses': 100.0 * unit.amu}
          if sigmas == None:
            sigmas = {'bb_bb_sigma': 1.875 * unit.nanometer,'bb_sc_sigma': 1.875 * unit.nanometer,'sc_sc_sigma': 1.875 * unit.nanometer}
          if epsilons == None:
            epsilons = {'bb_bb_eps': 0.05 * unit.kilocalorie_per_mole,'sc_sc_eps': 0.05 * unit.kilocalorie_per_mole}
          if bond_lengths == None:
            bond_lengths = {'bb_bb_bond_length': 0.75 * unit.nanometer,'bb_sc_bond_length': 0.75 * unit.nanometer,'sc_sc_bond_length': 0.75 * unit.nanometer}

          """
          Initialize user-defined input.
          """

          self.polymer_length = polymer_length
          self.backbone_lengths = backbone_lengths
          self.sidechain_lengths = sidechain_lengths
          self.sidechain_positions = sidechain_positions
          self.bond_lengths = bond_lengths
          self.sigmas = sigmas
          self.epsilons = epsilons
          if monomer_types == None:
            self.monomer_types = self.get_monomer_types()
          else:
            self.monomer_types = monomer_types


          if heteropolymer == True:
            if sequence == None:
              print("ERROR: The 'heteropolymer'=True flag was selected, but no")
              print("'sequence' was provided.  Please rerun with an input 'sequence'.")
              exit()
            else:
              self.sequence = sequence
          else:
           sequence = []
           monomer_type = self.monomer_types[0]
           for monomer in range(self.polymer_length):
            sequence.append(monomer_type)

          self.sequence = sequence
          self.num_beads = self.get_num_beads()
          self.particle_list = self.get_particle_list()
          self.masses = masses
          self.bond_force_constants = bond_force_constants
          self.bond_angle_force_constants = bond_angle_force_constants
          self.equil_bond_angles = equil_bond_angles
          self.torsion_force_constants = torsion_force_constants
          self.equil_torsion_angles = equil_torsion_angles
          self.charges = charges

          self.include_bond_forces = include_bond_forces
          self.include_bond_angle_forces = include_bond_angle_forces
          self.include_nonbonded_forces = include_nonbonded_forces
          self.include_torsion_forces = include_torsion_forces
          self.check_energy_conservation = check_energy_conservation

          self.constrain_bonds = constrain_bonds
          self.bond_list = self.get_bond_list()
          self.bond_angle_list = self.get_bond_angle_list()
          self.torsion_list = self.get_torsion_list()
          if exclusions == True:
           self.nonbonded_exclusion_list = self.get_nonbonded_exclusion_list()
          else:
           self.nonbonded_exclusion_list = None
          self.nonbonded_interaction_list = self.get_nonbonded_interaction_list()

          self.particle_types = add_new_elements(self)

          if positions == None: 
           if random_positions:
            if use_structure_library:
              self.positions = util.random_positions(self,use_library=True)
            else:
              self.positions = util.random_positions(self)
           else:
            if use_structure_library:
             if polymer_length == 12:
              positions_file = str(str(str(os.path.abspath(__file__)).split('/cg_model')[0])+"/structure_library/12_1_1_0/helix.pdb")
              self.positions = PDBFile(positions_file).getPositions()
             else:
              self.positions = util.random_positions(self,use_library=True)
            else:
              self.positions = None
          else:
           self.positions = positions

          self.simulation = None

          #print("Assigning topology")          
          if topology == None:
            if self.positions != None:
              self.topology = build_topology(self,use_pdbfile=True)
          else:
              self.topology = topology
 
          #print("Assigning system")
          if system == None:
            self.system = build_system(self)
          else:
            self.system = system

        def get_monomer_types(self):
          """
          Get a list of 'monomer_types' for all unique monomers.

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
             - monomer_types ( List( dict( 'monomer_name': str, 'backbone_length': int, 'sidechain_length': int, 'sidechain_positions': List( int ), 'num_beads': int, 'bond_lengths': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'epsilons': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ), 'sigmas': List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) ) ) ) - A list of unique monomer types in the coarse grained model

          """
          monomer_name_modifier = ['A','B','C','D','E','F','G','H']
          monomer_types = []
          monomer_type_index = 0
          for backbone_length,sidechain_length in zip(self.backbone_lengths,self.sidechain_lengths):
            num_beads = backbone_length
            for sidechain_position in self.sidechain_positions:
             num_beads = num_beads + sidechain_length
            monomer_name = str('CG'+str(backbone_length)+str(sidechain_length))
            if monomer_name in monomer_types:
             modifier_index = 0
             while monomer_name in monomer_types:
              monomer_name = str('CG'+str(backbone_length)+str(sidechain_length)+str(modifier_index))
              modifier_index = modifier_index + 1
            monomer_type = {'monomer_name': monomer_name, 'backbone_length': backbone_length, 'sidechain_length': sidechain_length, 'sidechain_positions': sidechain_position, 'num_beads': num_beads, 'bond_lengths': self.bond_lengths, 'epsilons': self.epsilons, 'sigmas': self.sigmas}
            monomer_types.append(monomer_type)
          return(monomer_types)

        def get_num_beads(self):
          """
          Calculate the number of beads in a coarse grained model class object

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
            - num_beads (int) - The total number of beads in the coarse grained model

          """
          num_beads = 0
          for monomer_type in self.sequence:
           num_beads = num_beads + monomer_type['num_beads']
          return(num_beads)

        def get_particle_list(self):
          """
          Get a list of particles, where the indices correspond to those in the system/topology.

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
            - particle_list ( List( str ) ) - A list of unique particles in the coarse grained model

          """
          particle_list = []
          for monomer_type in self.sequence:
           cg_particle_index = 1
           for backbone_bead in range(monomer_type['backbone_length']):
            particle_symbol = str("X"+str(cg_particle_index))
            particle_list.append(particle_symbol)
            cg_particle_index = cg_particle_index + 1
            if type(monomer_type['sidechain_positions']) == int:
             sidechain_positions = [monomer_type['sidechain_positions']]
            else:
             sidechain_positions = monomer_type['sidechain_positions']
            if backbone_bead in sidechain_positions:
             for sidechain in range(monomer_type['sidechain_length']):
              particle_symbol = str("A"+str(cg_particle_index))
              particle_list.append(particle_symbol)
              cg_particle_index = cg_particle_index + 1
          return(particle_list)


        def get_bond_list(self):
          """
          Construct a bond list for the coarse grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
             - bond_list ( List( List( int, int ) ) ) - A list of the bonds in the coarse grained model.

          """
          bond_list = []
          bead_index = 0
          for monomer in range(len(self.sequence)):
            monomer_type = self.sequence[monomer]
            for backbone_bead in range(monomer_type['backbone_length']):
             if (int(monomer) != 0 and backbone_bead == 0 and monomer_type['backbone_length']-1 in [monomer_type['sidechain_positions']]) or (backbone_bead != 0 and backbone_bead-1 in [monomer_type['sidechain_positions']]):
               parent_index = bead_index - monomer_type['sidechain_length'] - 1
             else: 
               parent_index = bead_index - 1
             if bead_index != 0:
              if parent_index < bead_index:
               bond_list.append([parent_index,bead_index])
              else:
               bond_list.append([bead_index,parent_index])
             bead_index = bead_index + 1
             
             if int(backbone_bead) in [monomer_type['sidechain_positions']]:
                for sidechain_bead in range(monomer_type['sidechain_length']):
                  parent_index = bead_index - 1
                  if parent_index < bead_index:
                   bond_list.append([parent_index,bead_index])
                  else:
                   bond_list.append([bead_index,parent_index])
                  bead_index = bead_index + 1
          return(bond_list)

        def get_nonbonded_interaction_list(self):
          """
          Construct a nonbonded interaction list for the coarse grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
             - interaction_list ( List( List( int, int ) ) ) - A list of the nonbonded interactions (which don't violate exclusion rules) in the coarse grained model

          """

          interaction_list = []
          bond_list = self.get_bond_list()
          for particle_1 in range(self.num_beads):
               for particle_2 in range(particle_1+1,self.num_beads):
                   if [particle_1,particle_2] not in bond_list and [particle_2,particle_1] not in bond_list:
                     if [particle_1,particle_2] not in interaction_list:
                       if [particle_2,particle_1] not in interaction_list:
                         interaction_list.append([particle_1,particle_2])
                     if [particle_2,particle_1] not in interaction_list:
                       if [particle_1,particle_2] not in interaction_list:
                         interaction_list.append([particle_2,particle_1])
          exclusion_list = self.nonbonded_exclusion_list
          if exclusion_list != None:
           for interaction in interaction_list:
            if interaction in exclusion_list or [interaction[1],interaction[0]] in exclusion_list:
              interaction_list.remove(interaction)
          #interaction_list = [[0,1]]
          return(interaction_list)

        def get_nonbonded_exclusion_list(self):
          """
          Get a list of the nonbonded interaction exclusions, which are assigned if two particles are separated by less than three bonds

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
             - exclusion_list ( List( List( int, int ) ) ) - A list of the nonbonded particle interaction exclusions for the coarse grained model

          """
          bond_list = self.bond_list
          exclusion_list = []
          for i in range(self.num_beads):
            for j in range(i+1,self.num_beads):
              if [i,j] in bond_list or [j,i] in bond_list:
                if [i,j] not in exclusion_list:
                  exclusion_list.append([i,j])
              for angle in self.bond_angle_list:
                if i in angle and j in angle:
                  if [i,j] not in exclusion_list:
                    exclusion_list.append([i,j]) 
          return(exclusion_list)

        def get_bond_angle_list(self):
          """
          Construct a list of bond angles, which can be used to build bond angle potentials for the coarse grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
             - bond_angles ( List( List( int, int, int ) ) ) - A list of indices for all of the bond angles in the coarse grained model

          """

          bond_list = self.bond_list
          bond_angles = []
          # Choose the first bond we will use to define a bond angle
          for bond_1 in bond_list:
            # Choose a second bond with which to attempt a bond angle
            # definition.
            for bond_2 in bond_list:
             # Make sure the bonds are different
             if bond_2 != bond_1 and [bond_2[1],bond_2[0]] != bond_1:
              # Make sure the bonds share a common atom
              bond_angle = []
              if bond_2[0] in bond_1 or bond_2[1] in bond_1:
               if bond_2[0] == bond_1[1]:
                bond_angle = [bond_1[0],bond_1[1],bond_2[1]]
               if bond_2[0] == bond_1[0]:
                bond_angle = [bond_1[1],bond_1[0],bond_2[1]]
               if bond_2[1] == bond_1[1]:
                bond_angle = [bond_1[0],bond_1[1],bond_2[0]]
               if bond_2[1] == bond_1[0]:
                bond_angle = [bond_1[1],bond_1[0],bond_2[0]]
              if len(bond_angles) > 0 and len(bond_angle) > 0:
               unique = True
               bond_angle_set = set(tuple(angle) for angle in bond_angles)
               bond_angles_temp = [ list(angle) for angle in bond_angle_set ]
               bond_angle_reverse = [bond_angle[2],bond_angle[1],bond_angle[0]]
               if any([bond_angle in bond_angles_temp, bond_angle_reverse in bond_angles_temp]):
                 unique = False
               if unique: bond_angles.append(bond_angle)
              if len(bond_angles) == 0 and len(bond_angle) != 0:
               bond_angles.append(bond_angle)
          return(bond_angles)

        def get_torsion_list(self):
          """
          Construct a list of particle indices from which to define torsions for the coarse grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
            - torsions ( List( List( int, int, int, int ) ) ) - A list of the particle indices for the torsions in the coarse grained model

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
             if bond_2 != bond_1 and [bond_2[1],bond_2[0]] != bond_1:
              if bond_1[0] == bond_2[1]:
                torsion = [bond_2[0],bond_2[1],bond_1[1]]
              if bond_1[1] == bond_2[0]:
                torsion = [bond_1[0],bond_1[1],bond_2[1]]

             if len(torsion) == 3:
               # Choose a third bond with which to attempt a torsion definition
               for bond_3 in bond_list:
                 # Make sure the third bond is different from the first two bonds.
                 if bond_3 != bond_1 and [bond_3[1],bond_3[0]] != bond_1 and bond_3 != bond_2 and [bond_3[1],bond_3[0]] != bond_2:
                   if bond_3[0] in torsion or bond_3[1] in torsion:
                     if bond_3[0] == torsion[0] and len(torsion) < 4:
                       torsion.insert(0,bond_3[1])
                     if bond_3[0] == torsion[1] and len(torsion) == 4:
                       torsion[0] = bond_3[1]

                     if bond_3[1] == torsion[0] and len(torsion) < 4:
                       torsion.insert(0,bond_3[0])
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
                     if unique: torsions.append(torsion)

          for angle in angle_list:
            for bond in bond_list:
             if bond[0] in angle and bond[1] not in angle or bond[1] in angle and bond[0] not in angle:
              if bond[0] == angle[0]:
                torsion = [bond[1],angle[0],angle[1],angle[2]]
                unique = True
                for existing_torsion in torsions:
                 if Counter(torsion) == Counter(existing_torsion):
                   unique = False
                if unique: torsions.append(torsion)
              if bond[0] == angle[2]:
                torsion = [angle[0],angle[1],angle[2],bond[1]]
                unique = True
                for existing_torsion in torsions:
                 if Counter(torsion) == Counter(existing_torsion):
                   unique = False
                if unique: torsions.append(torsion)
              if bond[1] == angle[0]:
                torsion = [bond[0],angle[0],angle[1],angle[2]]
                unique = True
                for existing_torsion in torsions:
                 if Counter(torsion) == Counter(existing_torsion):
                   unique = False
                if unique: torsions.append(torsion)
              if bond[1] == angle[2]:
                torsion = [angle[0],angle[1],angle[2],bond[0]]
                unique = True
                for existing_torsion in torsions:
                 if Counter(torsion) == Counter(existing_torsion):
                   unique = False
                if unique: torsions.append(torsion)

          torsion_set = set(tuple(torsion) for torsion in torsions)
          torsions = [ list(torsion) for torsion in torsion_set ]            
          return(torsions)

        def get_particle_name(self,particle_index):
          """
          Returns the name of a particle, given its index within the model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :returns: 
            - particle_name ( str ) - The name of the particle

          """
          particle_name = self.particle_list[particle_index]
          return(particle_name)

        def get_particle_type(self,particle_index,particle_name=None):
          """
          Indicates if a particle is a backbone bead or a sidechain bead

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :param particle_name: Name of the particle that we would like to "type".
          :type particle_name: str

          :returns: 
             - particle_type (str) - 'backbone' or 'sidechain'

          """
          if particle_name == None: particle_name = self.particle_list[particle_index]
          if 'X' in particle_name: particle_type = 'backbone'
          if 'A' in particle_name: particle_type = 'sidechain'

          return(particle_type)

        def get_particle_mass(self,particle_index):
          """
          Get the mass for a particle, given its index within the coarse grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :returns: 
            - particle_mass ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The mass for the provided particle index

          """
          particle_type = self.get_particle_type(particle_index)
          if particle_type == 'backbone': particle_mass = self.masses['backbone_bead_masses']
          if particle_type == 'sidechain': particle_mass = self.masses['sidechain_bead_masses']
          return(particle_mass)

        def get_particle_charge(self,particle_index):
          """
          Returns the charge for a particle, given its index within the coarse grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :returns: 
            - particle_charge ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The charge for the provided particle index

          """
          particle_type = self.get_particle_type(particle_index)
          if particle_type == 'backbone': particle_charge = self.charges['backbone_bead_charges']
          if particle_type == 'sidechain': particle_charge = self.charges['sidechain_bead_charges']
          return(particle_charge)

        def get_sigma(self,particle_index,particle_type=None):
          """
          Returns the Lennard-Jones potential sigma value for a particle, given its index within the coarse grained model.

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :param particle_type: Designates a particle as "backbone" or "sidechain"
          :type particle_type: str

          :returns: 
            - sigma ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned Lennard-Jones sigma value for the provided particle index

          """

          if particle_type == None: particle_type = self.get_particle_type(particle_index)
          if particle_type == 'backbone': sigma = self.sigmas['bb_bb_sigma']
          if particle_type == 'sidechain': sigma = self.sigmas['sc_sc_sigma']
          return(sigma)

        def get_epsilon(self,particle_index,particle_type=None):
          """
          Returns the Lennard-Jones potential epsilon value for a particle, given its index within the coarse grained model.

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_index: Index of the particle for which we would like to determine the type
          :type particle_index: int

          :param particle_type: Designates a particle as "backbone" or "sidechain"
          :type particle_type: str

          :returns: 
             - epsilon ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned Lennard-Jones epsilon value for the provided particle index


          """
          if particle_type == None: particle_type = self.get_particle_type(particle_index)
          if particle_type == 'backbone': epsilon = self.epsilons['bb_bb_eps']
          if particle_type == 'sidechain': epsilon = self.epsilons['sc_sc_eps']
          return(epsilon)

        def get_all_particle_masses(self):
          """
          Returns a list of all unique particle masses

          :param CGModel: CGModel() class object
          :type CGModel: class

          :returns: 
            - list_of_masses ( List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) ) - List of unique particle masses

          """
          list_of_masses = []
          list_of_masses.append(self.masses['backbone_bead_masses'])
          list_of_masses.append(self.masses['sidechain_bead_masses'])
          return(list_of_masses)

        def get_bond_length_from_names(self,particle_1_name,particle_2_name):
          """
          Determines the correct bond length for two particles, given their symbols.

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param particle_1_name: Name for the first particle
          :type particle_1_name: str

          :param particle_2_name: Name for the second particle
          :type particle_2_name: str

          :returns: 
             - bond_length ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned bond length for the provided particles

          """
          if 'B' in particle_1_name: particle_1_type = 'backbone'
          else: particle_1_type = 'sidechain'
          if particle_1_type == 'backbone' and particle_2_type == 'backbone':
           bond_length = self.bond_lengths['bb_bb_bond_length']

          if particle_1_type == 'backbone' and particle_2_type == 'sidechain':
           bond_length = self.bond_lengths['bb_sc_bond_length']

          if particle_1_type == 'sidechain' and particle_2_type == 'sidechain':
           bond_length = self.bond_lengths['bb_bb_bond_length']

          return(bond_length)

        def get_bond_length(self,particle_1_index,particle_2_index):
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
          if 'B' in self.particle_list[particle_1_index]: particle_1_type = 'backbone'
          else: particle_1_type = 'sidechain'

          if 'B' in self.particle_list[particle_2_index]: particle_2_type = 'backbone'
          else: particle_2_type = 'sidechain'

          if particle_1_type == 'backbone' and particle_2_type == 'backbone':
           bond_length = self.bond_lengths['bb_bb_bond_length']

          if particle_1_type == 'backbone' and particle_2_type == 'sidechain':
           bond_length = self.bond_lengths['bb_sc_bond_length']

          if particle_1_type == 'sidechain' and particle_2_type == 'sidechain':
           bond_length = self.bond_lengths['bb_bb_bond_length']

          return(bond_length)

        def get_bond_force_constant(self,particle_1_index,particle_2_index):
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
          if 'B' in self.particle_list[particle_1_index]: particle_1_type = 'backbone'
          else: particle_1_type = 'sidechain'

          if 'B' in self.particle_list[particle_2_index]: particle_2_type = 'backbone'
          else: particle_2_type = 'sidechain'

          if particle_1_type == 'backbone' and particle_2_type == 'backbone':
           bond_force_constant = self.bond_force_constants['bb_bb_bond_k']

          if particle_1_type == 'backbone' and particle_2_type == 'sidechain':
           bond_force_constant = self.bond_force_constants['bb_sc_bond_k']

          if particle_1_type == 'sidechain' and particle_2_type == 'sidechain':
           bond_force_constant = self.bond_force_constants['bb_bb_bond_k']

          return(bond_force_constant)

        def get_equil_bond_angle(self,particle_1_index,particle_2_index,particle_3_index):
          """
          Determines the correct equilibrium bond angle between three particles, given their indices within the coarse grained model

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
          particle_1_type = self.get_particle_type(particle_1_index)
          particle_2_type = self.get_particle_type(particle_2_index)
          particle_3_type = self.get_particle_type(particle_3_index)

          if particle_1_type == 'backbone' and particle_2_type == 'backbone' and particle_3_type == 'backbone':
           equil_bond_angle = self.equil_bond_angles['bb_bb_bb_angle_0']
          if (particle_1_type == 'backbone' and particle_2_type == 'backbone' and particle_3_type == 'sidechain') or (particle_1_type == 'sidechain' and particle_2_type == 'backbone' and particle_3_type == 'backbone'):
           equil_bond_angle = self.equil_bond_angles['bb_bb_sc_angle_0']
          if particle_1_type == 'backbone' and particle_2_type == 'sidechain' and particle_3_type == 'sidechain':
           equil_bond_angle = self.equil_bond_angles['bb_sc_sc_angle_0']
          if particle_1_type == 'sidechain' and particle_2_type == 'backbone' and particle_3_type == 'sidechain':
           equil_bond_angle = self.equil_bond_angles['sc_bb_sc_angle_0']
          if particle_1_type == 'sidechain' and particle_2_type == 'sidechain' and particle_3_type == 'sidechain':
           equil_bond_angle = self.equil_bond_angles['sc_sc_sc_angle_0']
          if particle_1_type == 'sidechain' and particle_2_type == 'sidechain' and particle_3_type == 'backbone':
           equil_bond_angle = self.equil_bond_angles['sc_sc_bb_angle_0']


          return(equil_bond_angle)

        def get_bond_angle_force_constant(self,particle_1_index,particle_2_index,particle_3_index):
          """
          Determines the correct bond angle force constant for a bond angle between three particles, given their indices within the coarse grained model

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
          particle_1_type = self.get_particle_type(particle_1_index)
          particle_2_type = self.get_particle_type(particle_2_index)
          particle_3_type = self.get_particle_type(particle_3_index)

          if particle_1_type == 'backbone' and particle_2_type == 'backbone' and particle_3_type == 'backbone':
           bond_angle_force_constant = self.bond_angle_force_constants['bb_bb_bb_angle_k']
          if (particle_1_type == 'backbone' and particle_2_type == 'backbone' and particle_3_type == 'sidechain') or (particle_1_type == 'sidechain' and particle_2_type == 'backbone' and particle_3_type == 'backbone'):
           bond_angle_force_constant = self.bond_angle_force_constants['bb_bb_sc_angle_k']
          if particle_1_type == 'backbone' and particle_2_type == 'sidechain' and particle_3_type == 'sidechain':
           bond_angle_force_constant = self.bond_angle_force_constants['bb_sc_sc_angle_k']
          if particle_1_type == 'sidechain' and particle_2_type == 'backbone' and particle_3_type == 'sidechain':
           bond_angle_force_constant = self.bond_angle_force_constants['sc_bb_sc_angle_k']
          if particle_1_type == 'sidechain' and particle_2_type == 'sidechain' and particle_3_type == 'sidechain':
           bond_angle_force_constant = self.bond_angle_force_constants['sc_sc_sc_angle_k']
          if particle_1_type == 'sidechain' and particle_2_type == 'sidechain' and particle_3_type == 'backbone':
           bond_angle_force_constant = self.bond_angle_force_constants['sc_sc_bb_angle_k']


          return(bond_angle_force_constant)

        def get_torsion_force_constant(self,torsion):
          """         
          Determines the correct torsion force constant for a torsion (bond angle involving four particles), given their indices within the coarse grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param torsion: A list of the indices for the particles in a torsion
          :type torsion: List( int )

          :returns: 
             - torsion_force_constant ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The assigned torsion force constant for the provided particles

          """
          particle_1_type = self.get_particle_type(torsion[0])
          particle_2_type = self.get_particle_type(torsion[1])
          particle_3_type = self.get_particle_type(torsion[2])
          particle_4_type = self.get_particle_type(torsion[3])

          particle_types = [particle_1_type,particle_2_type,particle_3_type,particle_4_type]

          if particle_types[0] == 'sidechain':
           if particle_types[1] == 'backbone':
            if particle_types[2] == 'backbone':
             if particle_types[3] == 'backbone':
              torsion_force_constant = self.torsion_force_constants['sc_bb_bb_bb_torsion_k']
             if particle_types[3] == 'sidechain':
              torsion_force_constant = self.torsion_force_constants['sc_bb_bb_sc_torsion_k']
            if particle_types[2] == 'sidechain':
             if particle_types[3] == 'backbone':
              torsion_force_constant = self.torsion_force_constants['bb_sc_sc_bb_torsion_k']
             if particle_types[3] == 'sidechain':
              torsion_force_constant = self.torsion_force_constants['sc_bb_sc_sc_torsion_k']
           if particle_types[1] == 'sidechain':
            if particle_types[2] == 'backbone':
             if particle_types[3] == 'backbone':
              torsion_force_constant = self.torsion_force_constants['sc_sc_bb_bb_torsion_k']
             if particle_types[3] == 'sidechain':
              torsion_force_constant = self.torsion_force_constants['sc_sc_bb_sc_torsion_k']
            if particle_types[2] == 'sidechain':
             if particle_types[3] == 'backbone':
              torsion_force_constant = self.torsion_force_constants['sc_sc_sc_bb_torsion_k']
             if particle_types[3] == 'sidechain':
              torsion_force_constant = self.torsion_force_constants['sc_sc_sc_sc_torsion_k']
          if particle_types[0] == 'backbone':
           if particle_types[1] == 'backbone':
            if particle_types[2] == 'backbone':
             if particle_types[3] == 'backbone':
              torsion_force_constant = self.torsion_force_constants['bb_bb_bb_bb_torsion_k']
             if particle_types[3] == 'sidechain':
              torsion_force_constant = self.torsion_force_constants['bb_bb_bb_sc_torsion_k']
            if particle_types[2] == 'sidechain':
             if particle_types[3] == 'backbone':
              torsion_force_constant = self.torsion_force_constants['bb_bb_sc_bb_torsion_k']
             if particle_types[3] == 'sidechain':
              torsion_force_constant = self.torsion_force_constants['bb_bb_sc_sc_torsion_k']
           if particle_types[1] == 'sidechain':
            if particle_types[2] == 'backbone':
             if particle_types[3] == 'backbone':
              torsion_force_constant = self.torsion_force_constants['bb_sc_bb_bb_torsion_k']
             if particle_types[3] == 'sidechain':
              torsion_force_constant = self.torsion_force_constants['bb_sc_bb_sc_torsion_k']
            if particle_types[2] == 'sidechain':
             if particle_types[3] == 'backbone':
              torsion_force_constant = self.torsion_force_constants['bb_sc_sc_bb_torsion_k']
             if particle_types[3] == 'sidechain':
              torsion_force_constant = self.torsion_force_constants['bb_sc_sc_sc_torsion_k']
          return(torsion_force_constant)

        def get_equil_torsion_angle(self,torsion):
          """         
          Determines the correct equilibrium angle for a torsion (bond angle involving four particles), given their indices within the coarse grained model

          :param CGModel: CGModel() class object
          :type CGModel: class

          :param torsion: A list of the indices for the particles in a torsion
          :type torsion: List( int )

          :returns: 
             - equil_torsion_angle (float) - The assigned equilibrium torsion angle for the provided particles

          """
          particle_1_type = self.get_particle_type(torsion[0])
          particle_2_type = self.get_particle_type(torsion[1])
          particle_3_type = self.get_particle_type(torsion[2])
          particle_4_type = self.get_particle_type(torsion[3])

          particle_types = [particle_1_type,particle_2_type,particle_3_type,particle_4_type]

          if particle_types[0] == 'sidechain':
           if particle_types[1] == 'backbone':
            if particle_types[2] == 'backbone':
             if particle_types[3] == 'backbone':
              equil_torsion_angle = self.equil_torsion_angles['sc_bb_bb_bb_torsion_0']
             if particle_types[3] == 'sidechain':
              equil_torsion_angle = self.equil_torsion_angles['sc_bb_bb_sc_torsion_0']
            if particle_types[2] == 'sidechain':
             if particle_types[3] == 'backbone':
              equil_torsion_angle = self.equil_torsion_angles['bb_sc_sc_bb_torsion_0']
             if particle_types[3] == 'sidechain':
              equil_torsion_angle = self.equil_torsion_angles['sc_bb_sc_sc_torsion_0']
           if particle_types[1] == 'sidechain':
            if particle_types[2] == 'backbone':
             if particle_types[3] == 'backbone':
              equil_torsion_angle = self.equil_torsion_angles['sc_sc_bb_bb_torsion_0']
             if particle_types[3] == 'sidechain':
              equil_torsion_angle = self.equil_torsion_angles['sc_sc_bb_sc_torsion_0']
            if particle_types[2] == 'sidechain':
             if particle_types[3] == 'backbone':
              equil_torsion_angle = self.equil_torsion_angles['sc_sc_sc_bb_torsion_0']
             if particle_types[3] == 'sidechain':
              equil_torsion_angle = self.equil_torsion_angles['sc_sc_sc_sc_torsion_0']
          if particle_types[0] == 'backbone':
           if particle_types[1] == 'backbone':
            if particle_types[2] == 'backbone':
             if particle_types[3] == 'backbone':
              equil_torsion_angle = self.equil_torsion_angles['bb_bb_bb_bb_torsion_0']
             if particle_types[3] == 'sidechain':
              equil_torsion_angle = self.equil_torsion_angles['bb_bb_bb_sc_torsion_0']
            if particle_types[2] == 'sidechain':
             if particle_types[3] == 'backbone':
              equil_torsion_angle = self.equil_torsion_angles['bb_bb_sc_bb_torsion_0']
             if particle_types[3] == 'sidechain':
              equil_torsion_angle = self.equil_torsion_angles['bb_bb_sc_sc_torsion_0']
           if particle_types[1] == 'sidechain':
            if particle_types[2] == 'backbone':
             if particle_types[3] == 'backbone':
              equil_torsion_angle = self.equil_torsion_angles['bb_sc_bb_bb_torsion_0']
             if particle_types[3] == 'sidechain':
              equil_torsion_angle = self.equil_torsion_angles['bb_sc_bb_sc_torsion_0']
            if particle_types[2] == 'sidechain':
             if particle_types[3] == 'backbone':
              equil_torsion_angle = self.equil_torsion_angles['bb_sc_sc_bb_torsion_0']
             if particle_types[3] == 'sidechain':
              equil_torsion_angle = self.equil_torsion_angles['bb_sc_sc_sc_torsion_0']
          return(equil_torsion_angle)

