import numpy as np
from simtk import openmm as mm
from simtk.openmm import *
from simtk.openmm.vec3 import Vec3
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
import mdtraj
import simtk.openmm.app.element as elem
from simtk.openmm.app import *
import matplotlib.pyplot as pyplot
import csv
from cg_openmm.utilities.iotools import write_bonds
import sys

def get_simulation_time_step(
    topology, system, positions, temperature, total_simulation_time, friction=1.0/unit.picosecond, time_step_list=None
):
    """
    Determine a suitable simulation time step.

    :param topology: OpenMM Topology
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_

    :param system: OpenMM System()
    :type system: `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_

    :param positions: Positions array for the model we would like to test
    :type positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit )

    :param temperature: Simulation temperature
    :type temperature: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param total_simulation_time: Total run time for individual simulations
    :type total_simulation_time: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param friction: Langevin thermostat friction coefficient, default = 1 / ps
    :type friction: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param time_step_list: List of time steps for which to attempt a simulation in OpenMM.
    :type time_step_list: List, default = None

    :returns:
         - time_step ( `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_ ) - A successfully-tested simulation time-step for the provided coarse grained model
         - tolerance ( `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_ ) - The maximum change in forces that will be tolerated when testing the time step.

    :Example:

    >>> from simtk import unit
    >>> from foldamers.cg_model.cgmodel import CGModel
    >>> cgmodel = CGModel()
    >>> topology = cgmodel.topology
    >>> system = cgmodel.system
    >>> positions = cgmodel.positions
    >>> temperature = 300.0 * unit.kelvin
    >>> friction = 1.0 / unit.picosecond
    >>> total_simulation_time = 1.0 * unit.picosecond
    >>> time_step_list = [1.0 * unit.femtosecond, 2.0 * unit.femtosecond, 5.0 * unit.femtosecond]
    >>> best_time_step,max_force_tolerance = get_simulation_time_step(topology,system,positions,temperature,total_simulation_time,friction=friction,time_step_list=time_step_list)

    """
    tolerance = 10.0
    success = False

    if time_step_list is None:
        time_step_list = [10.0 - i * unit.femtosecond for i in [5.0, 7.5, 9.0, 9.5, 9.9, 9.99]]

    if not isinstance(time_step_list, list):
        time_step_list = [time_step_list]
    for time_step in time_step_list:
        integrator = LangevinIntegrator(
            temperature._value,
            friction,
            time_step.in_units_of(unit.picosecond)._value,
        )

        simulation = Simulation(topology, system, integrator)
        simulation.context.setPositions(positions.in_units_of(unit.nanometer))
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.reporters.append(PDBReporter("test.pdb", 1))
        simulation.reporters.append(
            StateDataReporter(
                "test.dat",
                1,
                step=True,
                totalEnergy=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
            )
        )
        total_steps = round(total_simulation_time.__div__(time_step))
        try:
            simulation.minimizeEnergy()
            positions = simulation.context.getState(getPositions=True).getPositions()
            success = True
            break
        except BaseException:
            continue
    if not success:
        for tolerance in [10 ** exponent for exponent in range(2, 10)]:
            try:
                #            simulation.context.applyConstraints(1.0e-8)
                integrator = LangevinIntegrator(
                    temperature._value,
                    friction,
                    time_step.in_units_of(unit.picosecond)._value,
                )
                simulation = Simulation(topology, system, integrator)
                simulation.context.setPositions(positions)
                simulation.minimizeEnergy(tolerance=tolerance)
                success = True
                break
            except BaseException:
                continue
    if not success:
        tolerance = None
    return (time_step, tolerance)


def minimize_structure(
    topology,
    system,
    positions,
    output_pdb=None,
    print_frequency=1,
    expand=None
):
    """
    Minimize the potential energy

    :param topology: OpenMM topology
    :type topology: Topology()

    :param system: OpenMM system
    :type system: System()

    :param positions: Positions array for the model we would like to test
    :type positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit )

    :param output_pdb: Output destinaton for PDB-formatted coordinates during the simulation
    :type output_pdb: str

    :returns:
         - positions ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit ) ) - Minimized positions
         - potential_energy ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ - Potential energy for the minimized structure.

    :Example:

    >>> from simtk import unit
    >>> from foldamers.cg_model.cgmodel import CGModel
    >>> cgmodel = CGModel()
    >>> topology = cgmodel.topology
    >>> system = cgmodel.system
    >>> positions = cgmodel.positions
    >>> output_pdb = "output.pdb"
    >>> minimum_energy_structure,potential_energy,openmm_simulation_object = minimize_structure(topology,system,positions,output_pdb=output_pdb,output_data=output_data,print_frequency=print_frequency)

    """
    integrator = LangevinIntegrator(500,1,0.0005) # parameters for expanding the structure.

    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions.in_units_of(unit.nanometer))
    if output_pdb is not None:
        simulation.reporters.append(PDBReporter(output_pdb, print_frequency))
    try:
        simulation.minimizeEnergy(tolerance=10,maxIterations=500)  # Set the simulation type to energy minimization
    except Exception:
        print(f"Minimization attempt failed.") 
        return (positions, simulation, np.nan)
    if expand:
        simulation.step(expand)
    positions_vec = simulation.context.getState(getPositions=True).getPositions()

    # turn it into a numpy array of quantity
    positions = unit.Quantity(
        np.array(
            [[float(positions_vec[i]._value[j]) for j in range(3)] for i in range(len(positions))]
        ),
        positions_vec.unit
    )

    potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()

    return (positions, potential_energy, simulation)


def get_mm_energy(topology, system, positions):
    """
    Get the OpenMM potential energy for a system, given a topology and set of positions.

    :param topology: OpenMM Topology()
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_

    :param system: OpenMM System()
    :type system: `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_

    :param positions: Positions array for the model we would like to test
    :type positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit )

    :returns:
         - potential_energy ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The potential energy for the model with the provided positions.

    :Example:

    >>> from foldamers.cg_model.cgmodel import CGModel
    >>> cgmodel = CGModel()
    >>> topology = cgmodel.topology
    >>> system = cgmodel.system
    >>> positions = cgmodel.positions
    >>> openmm_potential_energy = get_mm_energy(topology,system,positions)

    """
    simulation_time_step = 5.0 * unit.femtosecond
    friction = 0.0 / unit.picosecond
    integrator = LangevinIntegrator(
        0.0 * unit.kelvin, friction, simulation_time_step.in_units_of(unit.picosecond)
    )
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()

    return potential_energy


def build_mm_simulation(
    topology,
    system,
    positions,
    temperature=300.0 * unit.kelvin,
    friction=1.0 / unit.picosecond,
    simulation_time_step=None,
    total_simulation_time=1.0 * unit.picosecond,
    output_pdb=None,
    output_data=None,
    print_frequency=100,
    test_time_step=False,
):
    """
    Build an OpenMM Simulation()

    :param topology: OpenMM Topology()
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_

    :param system: OpenMM System()
    :type system: `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_

    :param positions: Positions array for the model we would like to test
    :type positions: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit )

    :param temperature: Simulation temperature, default = 300.0 K
    :type temperature: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param friction: Langevin thermostat friction coefficient, default = 1 / ps
    :type friction: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param simulation_time_step: Simulation integration time step
    :type simulation_time_step: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param total_simulation_time: Total run time for individual simulations
    :type total_simulation_time: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param output_pdb: Output destination for PDB coordinates, Default = None
    :type output_pdb: str

    :param output_data: Output destination for non-coordinate simulation data, Default = None
    :type output_data: str

    :param print_frequency: Number of simulation steps to skip when writing to output, Default = 100
    :type print_frequence: int

    :param test_time_step: Logical variable determining if a test of the time step will be performed, Default = False
    :type test_time_step: Logical

    :returns:
         - simulation ( `Simulation() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1simulation_1_1Simulation.html>`_ ) - OpenMM Simulation() object

    :Example:

    >>> from simtk import unit
    >>> from foldamers.cg_model.cgmodel import CGModel
    >>> cgmodel = CGModel()
    >>> topology = cgmodel.topology
    >>> system = cgmodel.system
    >>> positions = cgmodel.positions
    >>> temperature = 300.0 * unit.kelvin
    >>> friction = 1.0 / unit.picosecond
    >>> simulation_time_step = 5.0 * unit.femtosecond
    >>> total_simulation_time= 1.0 * unit.picosecond
    >>> output_pdb = "output.pdb"
    >>> output_data = "output.dat"
    >>> print_frequency = 20
    >>> openmm_simulation = build_mm_simulation(topology,system,positions,temperature=temperature,friction=friction,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,output_pdb=output_pdb,output_data=output_data,print_frequency=print_frequency,test_time_step=False)

    """

    integrator = LangevinIntegrator(
        temperature._value, friction, simulation_time_step.in_units_of(unit.picosecond)._value
    )


    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    if output_pdb is not None:
        simulation.reporters.append(PDBReporter(output_pdb, print_frequency))
    if output_data is not None:
        simulation.reporters.append(
            StateDataReporter(
                output_data,
                print_frequency,
                step=True,
                totalEnergy=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
            )
        )
    simulation.reporters.append(StateDataReporter(sys.stdout, print_frequency, step=True,
                                                  potentialEnergy=True, temperature=True))

    # minimization
    init_positions = positions
    try:
        simulation.minimizeEnergy(tolerance=100,maxIterations=1000) # Set the simulation type to energy
    except Exception:
        print("Minimization failed on building model")
        print(f"potential energy was {simulation.context.getState(getEnergy=True).getPotentialEnergy()}")
        print("initial positions were:")
        print(positions)

    return simulation


def run_simulation(
    cgmodel,
    output_directory,
    total_simulation_time,
    simulation_time_step,
    temperature,
	friction,
    print_frequency,
    minimize=True,
    output_pdb=None,
    output_data=None,
):
    """

    Run OpenMM() simulation

    :param cgmodel: CGModel() object
    :type cgmodel: class

    :param output_directory: Output directory for simulation data
    :type output_directory: str

    :param total_simulation_time: Total run time for individual simulations
    :type total_simulation_time: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param simulation_time_step: Simulation integration time step
    :type simulation_time_step: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param temperature: Simulation temperature, default = 300.0 K
    :type temperature: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_
	
    :param friction: Langevin thermostat friction coefficient, default = 1 / ps
    :type friction: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param print_frequency: Number of simulation steps to skip when writing to output, Default = 100
    :type print_frequence: int

    :Example:

    >>> import os
    >>> from simtk import unit
    >>> from foldamers.cg_model.cgmodel import CGModel
    >>> cgmodel = CGModel()
    >>> topology = cgmodel.topology
    >>> system = cgmodel.system
    >>> positions = cgmodel.positions
    >>> temperature = 300.0 * unit.kelvin
	>>> friction = 1.0 / unit.picosecond
    >>> simulation_time_step = 5.0 * unit.femtosecond
    >>> total_simulation_time= 1.0 * unit.picosecond
    >>> output_directory = os.getcwd()
    >>> output_pdb = "output.pdb"
    >>> output_data = "output.dat"
    >>> print_frequency = 20
    >>> run_simulation(cgmodel,output_directory,total_simulation_time,simulation_time_step,temperature,friction,print_frequency,output_pdb=output_pdb,output_data=output_data)

    .. warning:: When run with default options this subroutine is capable of producing a large number of output files.  For example, by default this subroutine will plot the simulation data that is written to an output file.

    """
    total_steps = round(total_simulation_time.__div__(simulation_time_step))
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    if output_pdb is None:
        output_pdb = str(str(output_directory) + "/simulation.pdb")
    else:
        output_pdb = str(str(output_directory) + "/" + str(output_pdb))
    if output_data is None:
        output_data = str(str(output_directory) + "/simulation.dat")
    else:
        output_data = str(str(output_directory) + "/" + str(output_data))

    simulation = build_mm_simulation(
        cgmodel.topology,
        cgmodel.system,
        cgmodel.positions,
        total_simulation_time=total_simulation_time,
        simulation_time_step=simulation_time_step,
        temperature=temperature,
		friction=friction,
        output_pdb=output_pdb,
        output_data=output_data,
        print_frequency=print_frequency,
    )

    print(f"Will run {total_steps} simulation steps")
    try:
        simulation.step(total_steps)
    except BaseException:
        plot_simulation_results(
            output_data, output_directory, simulation_time_step, total_simulation_time
            )
        print("Error: simulation attempt failed.")
        print("We suggest trying the following changes to see if they fix the problem:")
        print("1) Reduce the simulation time step")
        print("2) Make sure that the values for the model parameters are reasonable,")
        print("   particularly in comparison with the requested simulation")
        print(str("   temperature: " + str(temperature)))
        print("3) Make sure that the initial/input structure is reasonable for the")
        print("   input set of model parameters.")
        exit()

    if not cgmodel.include_bond_forces and cgmodel.constrain_bonds:
        file = open(output_pdb, "r")
        lines = file.readlines()
        file.close()
        os.remove(output_pdb)
        file = open(output_pdb, "w")
        for line in lines[:-1]:
            file.write(line)
        write_bonds(cgmodel, file)
        file.close()

    plot_simulation_results(
        output_data, output_directory, simulation_time_step, total_simulation_time
    )
    return


def read_simulation_data(simulation_data_file, simulation_time_step):
    """
    Read OpenMM simulation data

    :param simulation_data_file: Path to file that will be read
    :type simulation_data_file: str

    :param simulation_time_step: Time step to apply for the simulation data
    :type simulation_time_step: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :returns:
      - data ( dict( "Simulation Time": list,"Potential Energy": list,"Kinetic Energy": list,"Total Energy": list,"Temperature": list ) ) - A dictionary containing the simulation times, potential energies, kinetic energies, and total energies from an OpenMM simulation trajectory.

    :Example:

    >>> from simtk import unit
    >>> simulation_data_file = "output.dat"
    >>> simulation_time_step = 5.0 * unit.femtosecond
    >>> data = read_simulation_data(simulation_data_file,simulation_time_step)

    """
    data = {
        "Simulation Time": [],
        "Potential Energy": [],
        "Kinetic Energy": [],
        "Total Energy": [],
        "Temperature": [],
    }
    with open(simulation_data_file, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for row in reader:
            data["Simulation Time"].append(
                float(simulation_time_step.in_units_of(unit.picosecond)._value) * float(row[0])
            )
            data["Potential Energy"].append(float(row[1]))
            data["Kinetic Energy"].append(float(row[2]))
            data["Total Energy"].append(float(row[3]))
            data["Temperature"].append(float(row[4]))

    return data


def plot_simulation_data(simulation_times, y_data, plot_type=None, output_directory=None):
    """
    Plot simulation data.

    :param simulation_times: List of simulation times (x data)
    :type simulation_times: List

    :param y_data: List of simulation data
    :type y_data: List

    :param plot_type: Form of data to plot, Default = None, Valid options include: "Potential Energy", "Kinetic Energy", "Total Energy", "Temperature"
    :type plot_type: str

    :Example:

    >>> import os
    >>> from simtk import unit
    >>> simulation_data_file = "output.pdb"
    >>> simulation_time_step = 5.0 * unit.femtosecond
    >>> simulation_data = read_simulation_data(simulation_data_file,simulation_time_step)
    >>> simulation_times = simulation_data["Simulation Time"]
    >>> y_data = simulation_data["Potential Energy"]
    >>> plot_type = "Potential Energy"
    >>> plot_simulation_data(simulation_times,y_data,plot_type="Potential Energy")

    """
    figure = pyplot.figure(1)
    pyplot.xlabel("Simulation Time (Picoseconds)")
    if plot_type == "Potential Energy":
        file_name = "Potential_Energy.png"
        pyplot.ylabel("Potential Energy (kJ/mole)")
        pyplot.title("Simulation Potential Energy")
    pyplot.xlabel("Simulation Time (Picoseconds)")
    if plot_type == "Kinetic Energy":
        file_name = "Kinetic_Energy.png"
        pyplot.ylabel("Kinetic Energy (kJ/mole)")
        pyplot.title("Simulation Kinetic Energy")
    pyplot.xlabel("Simulation Time (Picoseconds)")
    if plot_type == "Total Energy":
        file_name = "Total_Energy.png"
        pyplot.ylabel("Total Energy (kJ/mole)")
        pyplot.title("Simulation Total Energy")
    pyplot.xlabel("Simulation Time (Picoseconds)")
    if plot_type == "Temperature":
        file_name = "Temperature.png"
        pyplot.ylabel("Temperature (Kelvin)")
        pyplot.title("Simulation Temperature")

    pyplot.plot(simulation_times, y_data)
    if output_directory is None:
        pyplot.savefig(file_name)
    else:
        output_file = str(str(output_directory) + "/" + str(file_name))
        pyplot.savefig(output_file)
    pyplot.close()
    return


def plot_simulation_results(
    simulation_data_file, plot_output_directory, simulation_time_step, total_simulation_time
):
    """
    Plot all data from an OpenMM output file

    :param simulation_data_file: Path to file containing simulation data
    :type simulation_data_file: str

    :param plot_output_directory: Path to folder where plotting results will be written.
    :type plot_output_directory: str

    :param simulation_time_step: Simulation integration time step
    :type simulation_time_step: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :Example:

    >>> import os
    >>> from simtk import unit
    >>> simulation_data_file = "output.pdb"
    >>> plot_output_directory = os.getcwd()
    >>> simulation_time_step = 5.0 * unit.femtosecond
    >>> plot_simulation_results(simulation_data_file,plot_output_directory,simulation_time_step)

    """
    data = read_simulation_data(simulation_data_file, simulation_time_step)

    plot_simulation_data(
        data["Simulation Time"],
        data["Potential Energy"],
        plot_type="Potential Energy",
        output_directory=plot_output_directory,
    )
    plot_simulation_data(
        data["Simulation Time"],
        data["Kinetic Energy"],
        plot_type="Kinetic Energy",
        output_directory=plot_output_directory,
    )
    plot_simulation_data(
        data["Simulation Time"],
        data["Total Energy"],
        plot_type="Total Energy",
        output_directory=plot_output_directory,
    )
    plot_simulation_data(
        data["Simulation Time"],
        data["Temperature"],
        plot_type="Temperature",
        output_directory=plot_output_directory,
    )
    return
