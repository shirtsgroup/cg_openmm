#!/usr/local/bin/env python

#foldamers/foldamers.py
#====

#Tools for building coarse grained models 

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================


from simtk import openmm as mm
from simtk import unit
import simtk.openmm.app.element as elem


def get_box_vectors(box_size):
        """

        Assign all side lengths for simulation box.

        Parameters
        ----------

        box_size: Simulation box length ( float * simtk.unit.length )

        """

        units = box_size.unit
        a = unit.Quantity(np.zeros([3]), units)
        a[0] = box_size
        b = unit.Quantity(np.zeros([3]), units)
        b[1] = box_size
        c = unit.Quantity(np.zeros([3]), units)
        c[2] = box_size
        return([a,b,c])

def set_box_vectors(system,box_size):
        """

        Build a simulation box.

        Parameters
        ----------

        system: OpenMM system object

        box_size: Simulation box length ( float * simtk.unit.length )

        """

        a,b,c = get_box_vectors(box_size)
        system.setDefaultPeriodicBoxVectors(a, b, c)
        return(system)

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


def build_mm_system(box_size,mass,num_beads,sigma,epsilon,charge):
        """

        Construct an OpenMM system for our coarse grained model

        Parameters
        ----------

        box_size: Simulation box length ( float * simtk.unit.length )

        mass: Coarse grained particle mass ( float * simtk.unit.length )

        num_beads: Total number of beads in our coarse grained model (int)

        sigma: Non-bonded bead Lennard-Jones interaction distances,
        ( float * simtk.unit.distance )

        epsilon: Non-bonded bead Lennard-Jones interaction strength,
        ( float * simtk.unit.energy )

        charge: Charge for all beads
        ( float * simtk.unit.charge ) 

        """

        system = mm.System()

        for particle in range(num_beads):
          system.addParticle(mass)

        nonbonded = build_mm_force(sigma,epsilon,charge,num_beads,cutoff=1*unit.nanometer)
        system.addForce(nonbonded)

        return(system)

def build_mm_topology(polymer_length,backbone_length,sidechain_length):
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

        topology = mm.Topology()

        chain = topology.addChain()

        for monomer in range(polymer_length):

            residue = topology.addResidue(str(particle), chain)

            for bead in range(backbone_length):
              topology.addAtom('X', 'test_1', residue)

            for bead in range(sidechain_length):
              topology.addAtom('Q', 'test_2', residue)

        return(topology)


def build_mm_simulation(topology,system,temperature,simulation_time_step,total_simulation_time,positions,output_data='output.dat',print_frequency=100):
        """
        Construct an OpenMM simulation object for our coarse grained model.

        Parameters
        ----------

        topology: OpenMM topology object

        system: OpenMM system object

        temperature: Simulation temperature ( float * simtk.unit.temperature )

        simulation_time_step: Simulation integration time step
        ( float * simtk.unit.time )

        total_simulation_time: Total simulation time ( float * simtk.unit.time )

        positions: Array containing the positions of all beads
        in the coarse grained model
        ( np.array( 'num_beads' x 3 , ( float * simtk.unit.distance ) )

        output_data: Name of output file where we will write the data from this
        simulation ( string )

        print_frequency: Number of simulation steps to skip when writing data
        to 'output_data' ( integer )
 
        """
        integrator = LangevinIntegrator(temperature, total_simulation_time, simulation_time_step)
        
        simulation = Simulation(topology, system, integrator)

        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature)

        simulation.reporters.append(StateDataReporter(output_data,print_frequency, \
        step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True))

        simulation.minimizeEnergy() # Set the simulation type to energy minimization

        return(simulation)

class cgmodel(object):
        """
        Construct all of the objects that OpenMM expects/requires 
        for simulations with a coarse grained model.

        Parameters
        ----------

        box_size: Simulation box length, 
        default = 10.00 * unit.nanometer

        polymer_length: Number of monomer units (integer), default = 8
      
        backbone_length: Number of beads in the backbone 
        portion of each (individual) monomer (integer), default = 1

        sidechain_length: Number of beads in the sidechain
        portion of each (individual) monomer (integer), default = 1

        sidechain_positions: List of integers defining the backbone
        bead indices upon which we will place the sidechains,
        default = [0] (Place a sidechain on the backbone bead with
        index "0" (first backbone bead) in each (individual) monomer

        mass: Mass of coarse grained beads ( float * simtk.unit.mass )
        default = 12.0 * unit.amu

        sigma: Non-bonded bead Lennard-Jones interaction distances,
        ( float * simtk.unit.distance )
        default = 8.4 * unit.angstrom

        epsilon: Non-bonded bead Lennard-Jones interaction strength,
        ( float * simtk.unit.energy )
        default = 0.5 * unit.kilocalorie_per_mole

        bond_length: Bond length for all beads that are bonded,
        ( float * simtk.unit.distance )
        default = 1.0 * unit.angstrom

        bb_bond_length: Bond length for all bonded backbone beads,
        ( float * simtk.unit.distance )
        default = 1.0 * unit.angstrom

        bs_bond_length: Bond length for all backbone-sidechain bonds,
        ( float * simtk.unit.distance )
        default = 1.0 * unit.angstrom

        ss_bond_length: Bond length for all beads within a sidechain,
        ( float * simtk.unit.distance )
        default = 1.0 * unit.angstrom

        charge: Charge for all beads
        ( float * simtk.unit.charge )
        default = 0.0 * unit.elementary_charge

        Attributes
        ----------

        box_size
        polymer_length
        backbone_length
        sidechain_length
        sidechain_positions
        mass
        sigma
        epsilon
        bond_length
        bb_bond_length
        bs_bond_length
        ss_bond_length
        charge
        num_beads
        topology
        system
        positions
        simulation

        Notes
        -----
        
        """

        # Built in class attributes
        _BUILT_IN_REGIONS = ('box_size','polymer_length','backbone_length','sidechain_length','sidechain_positions','mass','sigma','epsilon','bond_length','bs_bond_length','bb_bond_length','ss_bond_length','charge','topology','system','simulation')

        def __init__(self, box_size = 10.00 * unit.nanometer, polymer_length = 12, backbone_length = 1, sidechain_length = 1, sidechain_positions = [0], mass = 12.0 * unit.amu, sigma = 8.4 * unit.angstrom, epsilon = 0.5 * unit.kilocalorie_per_mole, bond_length = 1.0 * unit.angstrom, bb_bond_length = 1.0 * unit.angstrom, bs_bond_length = 1.0 * unit.angstrom, ss_bond_length = 1.0 * unit.angstrom, charge = 0.0 * unit.elementary_charge):

          """
          Initialize variables that were passed as input
          """

          self._box_size = box_size
          self._polymer_length = polymer_length
          self._backbone_length = backbone_length
          self._sidechain_length = sidechain_length
          self._sidechain_positions = sidechain_positions
          self._mass = mass
          self._sigma = sigma
          self._epsilon = epsilon
          self._bond_length = bond_length
          self._bb_bond_length = bb_bond_length
          self._bs_bond_length = bs_bond_length
          self._ss_bond_length = ss_bond_length
          self._charge = charge         

          """
          Initialize new (coarse grained) particle types:
          """
          elem.Element(117,'backbone','X',mass)
          elem.Element(118,'sidechain','Q',mass) 

          self._num_beads = polymer_length * ( backbone_length + sidechain_length )

          self._system = build_mm_system( box_size, mass, self._num_beads, sigma, epsilon, charge )

          self._topology = build_mm_topology( box_size,mass, self._num_beads )

          self._positions = util.random_positions( polymer_length, bacbone_length, sidechain_length, bond_length, sigma ) 

          self._simulation = build_mm_simulation( self._topology, self._system, self._positions, temperature = 500.0 * unit.kelvin, simulation_time_step = 0.002 * unit.picosecond, total_simulation_time = 1.000 * unit.picosecond )

          """
          Initialize attributes of our coarse grained model.
          """

        @property
        def box_size(self):
          return self._box_size

        @property
        def polymer_length(self):
          return self._polymer_length

        @property
        def backbone_length(self):
          return self._backbone_length

        @property
        def sidechain_length(self):
          return self._sidechain_length

        @property
        def sidechain_positions(self):
          return self._sidechain_positions

        @property
        def mass(self):
          return self._mass

        @property
        def sigma(self):
          return self._sigma

        @property
        def epsilon(self):
          return self._epsilon

        @property
        def bond_length(self):
          return self._bond_length

        @property
        def bb_bond_length(self):
          return self._bb_bond_length

        @property
        def bs_bond_length(self):
          return self._bs_bond_length

        @property
        def ss_bond_length(self):
          return self._ss_bond_length

        @property
        def charge(self):
          return self._charge

        @property
        def num_beads(self):
          return self._num_beads

        @property
        def topology(self):
          return self._topology

        @property
        def system(self):
          return self._system
         
        @property
        def simulation(self):
          return self._simulation
 
