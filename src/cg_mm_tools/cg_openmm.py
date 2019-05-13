import numpy as np
from simtk import openmm as mm
from simtk.openmm import *
from simtk import unit
import simtk.openmm.app.element as elem
from simtk.openmm.app import *


def distance(positions_1,positions_2):
        """
        Construct a matrix of the distances between all particles.

        Parameters
        ----------

        positions_1: Positions for a particle
        ( np.array( length = 3 ) )

        positions_2: Positions for a particle
        ( np.array( length = 3 ) )

        Returns
        -------

        distance
        ( float * unit )
        """

        direction_comp = np.zeros(3) * positions_1.unit

        for direction in range(len(direction_comp)):
          direction_comp[direction] = positions_1[direction].__sub__(positions_2[direction])

        direction_comb = np.zeros(3) * positions_1.unit.__pow__(2.0)
        for direction in range(3):
          direction_comb[direction] = direction_comp[direction].__pow__(2.0)

        sqrt_arg = direction_comb[0].__add__(direction_comb[1]).__add__(direction_comb[2])

        value = math.sqrt(sqrt_arg._value)
        units = sqrt_arg.unit.sqrt()
        distance = unit.Quantity(value=value,unit=units)

        return(distance)

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


def build_mm_simulation(topology,system,positions,temperature=300.0 * unit.kelvin,simulation_time_step=0.002 * unit.picosecond,total_simulation_time=1.0 * unit.picosecond,output_pdb='output.pdb',output_data='output.dat',print_frequency=100):
        """
        Construct an OpenMM simulation object for our coarse grained model.

        Parameters
        ----------

        topology: OpenMM topology object

        system: OpenMM system object

        positions: Array containing the positions of all beads
        in the coarse grained model
        ( np.array( 'num_beads' x 3 , ( float * simtk.unit.distance ) )

        temperature: Simulation temperature ( float * simtk.unit.temperature )

        simulation_time_step: Simulation integration time step
        ( float * simtk.unit.time )

        total_simulation_time: Total simulation time ( float * simtk.unit.time )

        output_data: Name of output file where we will write the data from this
        simulation ( string )

        print_frequency: Number of simulation steps to skip when writing data
        to 'output_data' ( integer )
 
        """
        integrator = LangevinIntegrator(temperature, total_simulation_time, simulation_time_step)
        
        simulation = Simulation(topology, system, integrator)

        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature)

        simulation.reporters.append(PDBReporter(output_pdb,print_frequency))
        simulation.reporters.append(StateDataReporter(output_data,print_frequency, \
        step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True))

        simulation.minimizeEnergy() # Set the simulation type to energy minimization

        return(simulation)

def lj_v(positions_1,positions_2,sigma,epsilon):
        dist = distance(positions_1,positions_2)
        quot = dist.__div__(sigma)
        attr = (quot.__pow__(6.0)).__mul__(2.0)
        rep = quot.__pow__(12.0)
        v = epsilon.__mul__(rep.__sub__(attr))
        return(v)


def get_nonbonded_interaction_list(cgmodel):
        interaction_list = []
        bond_list = [[bond[0]-1,bond[1]-1] for bond in cgmodel.get_bond_list()]
        for particle_1 in range(cgmodel.num_beads):
               for particle_2 in range(cgmodel.num_beads):
                 if particle_1 != particle_2:
                   if [particle_1,particle_2] not in bond_list and [particle_2,particle_1] not in bond_list:
                     if [particle_1,particle_2] not in interaction_list:
                       if [particle_2,particle_1] not in interaction_list:
                         interaction_list.append([particle_1,particle_2])
                     if [particle_2,particle_1] not in interaction_list:
                       if [particle_1,particle_2] not in interaction_list:
                         interaction_list.append([particle_2,particle_1])
        return(interaction_list)


def calculate_nonbonded_energy(cgmodel,particle1=None,particle2=None):
        nonbonded_interaction_list = get_nonbonded_interaction_list(cgmodel)
        positions = cgmodel.positions
        energy = unit.Quantity(0.0,cgmodel.epsilon.unit)
        if particle1 != None:
           dist = distance(positions[particle1],positions[particle2])
           inter_energy = lj_v(positions[particle1],positions[particle2],cgmodel.sigma,cgmodel.epsilon).in_units_of(unit.kilojoules_per_mole)
           energy = energy.__add__(inter_energy)
           return(energy)

        for interaction in nonbonded_interaction_list:
           dist = distance(positions[interaction[0]],positions[interaction[1]])
           inter_energy = lj_v(positions[interaction[0]],positions[interaction[1]],cgmodel.sigma,cgmodel.epsilon).in_units_of(unit.kilojoules_per_mole)
           energy = energy.__add__(inter_energy)
        return(energy)
