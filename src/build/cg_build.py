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
        return(topology)

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

        if cgmodel.include_bond_forces:
         # Create bond (harmonic) potentials
         bond_list = cgmodel.get_bond_list()
         new_bond_list = []
         bead_index = 1
         bond_force = mm.HarmonicBondForce()
         for bond in bond_list:
              new_bond = [bond[0]-1,bond[1]-1]
              new_bond_list.append(new_bond)
              bond_force_constant = cgmodel.get_bond_force_constant(new_bond[0],new_bond[1])
              bond_length = cgmodel.get_bond_length(new_bond[0],new_bond[1])
              bond_length = bond_length.in_units_of(unit.nanometer)._value
              bond_force.addBond(new_bond[0],new_bond[1],bond_length,bond_force_constant)
              if cgmodel.constrain_bonds:
               system.addConstraint(new_bond[0],new_bond[1], bond_length)
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
            system.addParticle(mass)
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
                system.addParticle(mass)
                bead_index = bead_index + 1
                sigma = sigma.in_units_of(unit.nanometer)._value
                charge = charge._value
                epsilon = epsilon.in_units_of(unit.kilojoule_per_mole)._value
                nonbonded_force.addParticle(charge,sigma,epsilon)
         system.addForce(nonbonded_force)
         nonbonded_force.createExceptionsFromBonds(new_bond_list,1.0,1.0)

        if cgmodel.include_bond_angle_forces:
         # Create bond angle potentials
         angle_list = cgmodel.get_bond_angle_list()
         angle_force = mm.HarmonicAngleForce()
         for angle in angle_list:
              bond_angle_force_constant = cgmodel.get_bond_angle_force_constant(angle[0],angle[1],angle[2])
              angle_force.addAngle(angle[0],angle[1],angle[2],cgmodel.equil_bond_angle,bond_angle_force_constant)
         system.addForce(angle_force)

        if cgmodel.include_torsion_forces:
         # Create torsion potentials
         torsion_list = cgmodel.get_torsion_list()
         torsion_force = mm.PeriodicTorsionForce()
         for torsion in torsion_list:
              torsion_force_constant = cgmodel.get_torsion_force_constant(torsion)
              torsion_force.addTorsion(torsion[0],torsion[1],torsion[2],torsion[3],1,cgmodel.equil_dihedral_angle,torsion_force_constant)
         system.addForce(torsion_force)

        return(system)


def get_mm_energy(topology,system,positions):
        """
        Get the OpenMM potential energy for a system, given a topology and set of positions.

        Parameters
        ----------

        topology: OpenMM topology object

        system: OpenMM system object

        positions: Array containing the positions of all beads
        in the coarse grained model
        ( np.array( 'num_beads' x 3 , ( float * simtk.unit.distance ) )

 
        """
        integrator = LangevinIntegrator(300.0 * unit.kelvin,0.0,1.0)
        simulation = Simulation(topology, system, integrator)
        simulation.context.setPositions(positions)
        potential_energy = simulation.context.getEnergy(potentialEnergy=True).getPotentialEnergy()

        return(potential_energy)

def build_mm_simulation(topology,system,positions,temperature=300.0 * unit.kelvin,simulation_time_step=None,total_simulation_time=1.0 * unit.picosecond,output_pdb='output.pdb',output_data='output.dat',print_frequency=100):
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
        if simulation_time_step == None:
#          print("No simulation time step provided.")
#          print("Going to attempt a range of time steps,")
#          print("to confirm their validity for these model settings,")
#          print("before performing a full simulation.")
          time_step_list = [(10.0 * (0.5 ** i)) * unit.femtosecond for i in range(0,14)]
          simulation_time_step,force_cutoff = get_simulation_time_step(topology,system,positions,temperature,total_simulation_time,time_step_list)
        friction = 0.0

        integrator = LangevinIntegrator(temperature._value,friction,simulation_time_step.in_units_of(unit.picosecond)._value)
        
        simulation = Simulation(topology, system, integrator)

        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature)

        simulation.reporters.append(PDBReporter(output_pdb,print_frequency))
        simulation.reporters.append(StateDataReporter(output_data,print_frequency, \
        step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True))

        simulation.minimizeEnergy() # Set the simulation type to energy minimization

        try:
          simulation_temp = simulation.__deepcopy__(memo={})
          simulation_temp.step(100)
        except:
#          print("Simulation attempt failed with a time step of: "+str(simulation_time_step))
#          print("Going to attempt to identify a smaller time step that allows simulation for this model and its current settings...")
          time_step_list = [(10.0 * (0.5 ** i)) * unit.femtosecond for i in range(0,14)]
          if all(simulation_time_step.__lt__(time_step) for time_step in time_step_list):
            print("Error: couldn't identify a suitable simulation time step for this model.")
            print("Check the model settings, consider changing the input time step,")
            print("and if this doesn't fix the problem, try changing the default list of time steps")
            print("that are sampled in 'src.build.cg_build.build_mm_simulation.py'")
            exit()
          for time_step in time_step_list:
            if time_step < simulation_time_step:
              simulation = build_mm_simulation(topology,system,positions,temperature=temperature,simulation_time_step=time_step,total_simulation_time=total_simulation_time,output_pdb=output_pdb,output_data=output_data,print_frequency=print_frequency)
              try:
                simulation_temp.step(100)
                return(simulation)
              except:
                continue
        return(simulation)
