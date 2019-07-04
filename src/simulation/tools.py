import numpy as np
from simtk import openmm as mm
from simtk.openmm import *
from simtk import unit
import simtk.openmm.app.element as elem
from simtk.openmm.app import *

def get_simulation_time_step(topology,system,positions,temperature,total_simulation_time,time_step_list=None):
        """
        Determine a valid simulation time step for our coarse grained model.

        Parameters
        ----------

        simulation: OpenMM simulation object

        time_step_list: List of time steps for which to attempt a simulation in OpenMM.

        Returns
        -------

        time_step: A time step that was successful for our simulation object.
        """
        tolerance = 10.0
        success = False

        if time_step_list == None:
          time_step_list = [(10.0 * (0.5 ** i)) * unit.femtosecond for i in range(0,14)]

        if type(time_step_list) != list:
          time_step_list = [time_step_list]
        for time_step in time_step_list:
          integrator = LangevinIntegrator(temperature._value, total_simulation_time.in_units_of(unit.picosecond)._value, time_step.in_units_of(unit.picosecond)._value)

          simulation = Simulation(topology, system, integrator)
          simulation.context.setPositions(positions)
#          simulation.context.setVelocitiesToTemperature(temperature)
          simulation.reporters.append(PDBReporter('test.pdb',1))
          simulation.reporters.append(StateDataReporter('test.dat',1, \
          step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True))

          total_steps = round(total_simulation_time.__div__(time_step))
#          print("Running test simulation with time step of "+str(time_step)+" for "+str(total_steps)+" steps.")
#          simulation.minimizeEnergy()
#          exit()
          try:
#            simulation.context.applyConstraints(1.0e-8)
            simulation.minimizeEnergy()
#            for step in range(0,total_steps):
#              simulation.step(1)
#              simulation.context.applyConstraints(1.0e-8)
            positions = simulation.context.getState(getPositions=True).getPositions()
#              confirm_bond_constraints(cgmodel,positions)
#            print("Simulation successful with a time step of: "+str(simulation_time_step))
            success = True
            break
          except:
            continue
        if not success:
#          print("ERROR: unable to find a suitable simulation time step for this coarse grained model.")
#          print("Attempting to identify a suitable time step by adjusting the force tolerance.")
#          print("with a constant time step of: "+str(time_step))
          for tolerance in [10 ** exponent for exponent in range(2,10)]:
#            print("Running test simulation with a force tolerance of: "+str(tolerance))
            try:
#            simulation.context.applyConstraints(1.0e-8)
             integrator = LangevinIntegrator(temperature, total_simulation_time, time_step)
             simulation = Simulation(topology, system, integrator)
             simulation.context.setPositions(positions)
             simulation.minimizeEnergy(tolerance=tolerance)
#             print("Simulation successful with a tolerance of: "+str(tolerance))
             success = True
             break
            except:
             continue
        if not success:
#          print("ERROR: unable to find a suitable simulation time step for this coarse grained model.")
#          print("Check the model parameters, and the range of time step options,")
#          print(str(time_step_list))
#          print("to see if either of these settings are the source of the problem.")
          tolerance = None
#          exit()
#        print(time_step)
#        print("Simulation successful with a time step of: "+str(time_step))
        return(time_step,tolerance)

def minimize_structure(topology,system,positions,temperature=0.0 * unit.kelvin,simulation_time_step=None,total_simulation_time=1.0 * unit.picosecond,output_pdb='minimum.pdb',output_data='minimization.dat',print_frequency=10):
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

        output_data: Name of output file where we will write the data from this
        simulation ( string )

        print_frequency: Number of simulation steps to skip when writing data
        to 'output_data' ( integer )

        """
        if simulation_time_step == None:
          simulation_time_step_list = [(10.0 * (0.5 ** i)) * unit.femtosecond for i in range(0,14)]
          time_step,tolerance = get_simulation_time_step(topology,system,positions,temperature,total_simulation_time,simulation_time_step_list)
          if tolerance == None:
#            print("This set of positions is not a reasonable initial configuration.")
            energy = "NaN"
            return(positions,energy)
        else:
          time_step = simulation_time_step
        integrator = LangevinIntegrator(temperature._value, total_simulation_time.in_units_of(unit.picosecond)._value, time_step.in_units_of(unit.picosecond)._value)


        simulation = Simulation(topology, system, integrator)
        simulation.context.setPositions(positions)
#        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.reporters.append(PDBReporter(output_pdb,print_frequency))
        simulation.reporters.append(StateDataReporter(output_data,print_frequency, \
        step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True))


        total_steps = round(total_simulation_time.__div__(time_step))
        try:
#          simulation.minimizeEnergy() # Set the simulation type to energy minimization
          step_length = 10
          for step in range(round(total_steps/step_length)):
            simulation.step(step_length)
#            simulation.context.applyConstraints(1.0e8)
          positions = simulation.context.getState(getPositions=True).getPositions()
          potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
#          confirm_bond_constraints(cgmodel,positions)
        except:
          print("Minimization attempt failed with a time step of: "+str(time_step))
          if time_step.__gt__(0.01 * unit.femtosecond):
            time_step = time_step / 2.0
            print("Attempting minimization with a smaller time step.")
            positions,potential_energy,time_step = minimize_structure(topology,system,positions,temperature=temperature,simulation_time_step=time_step,total_simulation_time=total_simulation_time,output_pdb=output_pdb,output_data=output_data,print_frequency=print_frequency)
          else:
            print("Try using the 'get_simulation_time_step()' function,")
            print("or changing the 'simulation_time_step',")
            print("to see if one of these changes solves the problem.")
            exit()

        return(positions,potential_energy,time_step)

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

def build_mm_simulation(topology,system,positions,temperature=300.0 * unit.kelvin,simulation_time_step=None,total_simulation_time=1.0 * unit.picosecond,output_pdb='output.pdb',output_data='output.dat',print_frequency=100,test_time_step=True):
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
        friction = 0.1

        integrator = LangevinIntegrator(temperature._value,friction,simulation_time_step.in_units_of(unit.picosecond)._value)
        
        simulation = Simulation(topology, system, integrator)

        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature)

        simulation.reporters.append(PDBReporter(output_pdb,print_frequency))
        simulation.reporters.append(StateDataReporter(output_data,print_frequency, \
        step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True))

#        simulation.minimizeEnergy() # Set the simulation type to energy minimization

        if test_time_step:
          try:
            simulation_temp = simulation.__deepcopy__(memo={})
            simulation_temp.step(100)
          except:
#            print("Simulation attempt failed with a time step of: "+str(simulation_time_step))
#            print("Going to attempt to identify a smaller time step that allows simulation for this model and its current settings...")
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


def run_simulation(cgmodel,output_directory,total_simulation_time,simulation_time_step,temperature,print_frequency):

  total_steps = round(total_simulation_time.__div__(simulation_time_step))
  if not os.path.exists(output_directory): os.mkdir(output_directory)
  output_pdb = str(str(output_directory)+'/simulation.pdb')
  output_data = str(str(output_directory)+'/simulation.dat')

  simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions,total_simulation_time=total_simulation_time,simulation_time_step=simulation_time_step,temperature=temperature,output_pdb=output_pdb,output_data=output_data,print_frequency=print_frequency)

  potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
  positions = simulation.context.getState(getPositions=True).getPositions()
  kinetic_energy = simulation.context.getState(getEnergy=True).getKineticEnergy()
  forces = simulation.context.getState(getForces=True).getForces()

  output_kinetic_energy = str(str(output_directory)+'/kinetic_energy.dat')
  kinetic_dat = open(output_kinetic_energy,"w")
  kinetic_dat.write(str(str(kinetic_energy)+"\n"))
  kinetic_dat.close()

  output_forces = str(str(output_directory)+'/forces.dat')
  forces_dat = open(output_forces,"w")
  forces_dat.write(str(str(forces)+"\n"))
  forces_dat.close()
  for step in range(total_steps):
      try:
        simulation.step(1)
      except:
        break
      kinetic_energy = simulation.context.getState(getEnergy=True).getKineticEnergy()
      output_kinetic_energy = str(str(output_directory)+'/kinetic_energy.dat')
      kinetic_dat = open(output_kinetic_energy,"a")
      kinetic_dat.write(str(str(kinetic_energy)+"\n"))
      kinetic_dat.close()

      forces = simulation.context.getState(getForces=True).getForces()
      output_forces = str(str(output_directory)+'/forces.dat')
      forces_dat = open(output_forces,"a")
      forces_dat.write(str(str(forces)+"\n"))
      forces_dat.close()

  return

