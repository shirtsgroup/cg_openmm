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

        for time_step in time_step_list:
          integrator = LangevinIntegrator(temperature, total_simulation_time, time_step)

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
          time_step,tolerance = get_simulation_time_step(topology,system,positions,temperature,simulation_time_step_list,total_simulation_time)
          if tolerance == None:
#            print("This set of positions is not a reasonable initial configuration.")
            energy = "NaN"
            return(positions,energy)
        else:
          time_step = simulation_time_step
        integrator = LangevinIntegrator(temperature, total_simulation_time, time_step)

        simulation = Simulation(topology, system, integrator)
        simulation.context.setPositions(positions)
#        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.reporters.append(PDBReporter(output_pdb,print_frequency))
        simulation.reporters.append(StateDataReporter(output_data,print_frequency, \
        step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True))


        total_steps = round(total_simulation_time.__div__(time_step))
#        print("Running minimization with a time step of "+str(time_step))
#        print("for "+str(total_steps)+" steps.")
        try:
          simulation.minimizeEnergy(tolerance=tolerance) # Set the simulation type to energy minimization
#          for step in range(0,total_steps):
#            simulation.step(1)
#            simulation.context.applyConstraints(1.0e8)
          positions = simulation.context.getState(getPositions=True).getPositions()
          potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
#          confirm_bond_constraints(cgmodel,positions)
        except:
          print("Minimization attempt failed with a time step of: "+str(time_step))
          print("Try using the 'get_simulation_time_step()' function,")
          print("or changing the 'simulation_time_step',")
          print("to see if one of these changes solves the problem.")
          exit()

        return(positions,potential_energy)

