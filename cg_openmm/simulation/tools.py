import csv
import sys

import matplotlib.pyplot as pyplot
import mdtraj as md
import numpy as np
import openmm as mm
import openmm.app.element as elem
from cg_openmm.utilities.iotools import (write_bonds,
                                         write_pdbfile_without_topology)
from openmm import *
from openmm import unit
from openmm.app import *
from openmm.app.pdbfile import PDBFile
from openmm.vec3 import Vec3


def minimize_structure(
    cgmodel, positions, tol=0.1, max_iter=1000, timestep=5*unit.femtosecond, output_file='minimized_structure.pdb',
):
    """
    Minimize the potential energy

    :param cgmodel: CGModel() object containing openmm system and topology objects 
    :type cgmodel: class

    :param positions: Positions array for the structure to be minimized
    :type positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit )

    :param output_file: Output destination for minimized structure file (including extension - 'dcd' and 'pdb' supported)
    :type output_file: str

    :returns:
         - positions ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit ) ) - Minimized positions
         - initial_potential_energy ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ - Potential energy for the initial structure.
         - final_potential_energy ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ - Potential energy for the minimized structure.
         - simulation (Openmm simulation object for minimization run)

    """
    # Integrator parameters shouldn't matter?
    integrator = LangevinIntegrator(300, 1, timestep)

    simulation = Simulation(cgmodel.topology, cgmodel.system, integrator)
    simulation.context.setPositions(positions.in_units_of(unit.nanometer))
    
    initial_potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    
    # ***Reporters do not do anything during minimization

    try:
        simulation.minimizeEnergy(
            tolerance=tol, maxIterations=max_iter,
        )
        
    except Exception:
        print(f"Minimization attempt failed.")

    positions_vec = simulation.context.getState(getPositions=True).getPositions()

    # turn it into a numpy array of quantity
    positions = unit.Quantity(
        np.array(
            [[float(positions_vec[i]._value[j]) for j in range(3)] for i in range(len(positions))]
        ),
        positions_vec.unit,
    )
    
    # Write positions to file:
    if output_file[-3:].lower() == 'dcd':
        dcdtraj = md.Trajectory(
            xyz=positions.value_in_unit(unit.nanometer),
            topology=md.Topology.from_openmm(cgmodel.topology),
        )
        md.Trajectory.save_dcd(dcdtraj,output_file)
    elif output_file[-3:].lower() == 'pdb':
        cgmodel.positions = positions
        write_pdbfile_without_topology(cgmodel, output_file)

    final_potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()

    return (positions, initial_potential_energy, final_potential_energy, simulation)


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
    output_traj=None,
    output_data=None,
    print_frequency=100,
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

    :param output_traj: Output destination for trajectory coordinates (.pdb or .dcd), Default = None
    :type output_traj: str

    :param output_data: Output destination for non-coordinate simulation data, Default = None
    :type output_data: str

    :param print_frequency: Number of simulation steps to skip when writing to output, Default = 100
    :type print_frequence: int

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
    >>> output_traj = "output.pdb"
    >>> output_data = "output.dat"
    >>> print_frequency = 20
    >>> openmm_simulation = build_mm_simulation(topology,system,positions,temperature=temperature,friction=friction,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,output_traj=output_traj,output_data=output_data,print_frequency=print_frequency)

    """

    integrator = LangevinIntegrator(
        temperature._value, friction, simulation_time_step.in_units_of(unit.picosecond)._value
    )

    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    if output_traj is not None:
        if output_traj[-3:] == 'pdb':
            simulation.reporters.append(PDBReporter(output_traj, print_frequency))
        elif output_traj[-3:] == 'dcd':
            simulation.reporters.append(DCDReporter(output_traj, print_frequency))
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
    simulation.reporters.append(
        StateDataReporter(
            sys.stdout, print_frequency, step=True, potentialEnergy=True, temperature=True
        )
    )

    # minimization
    init_positions = positions
    try:
        simulation.minimizeEnergy(
            tolerance=100, maxIterations=1000  # probably shouldn't be hard-coded.
        )  # Set the simulation type to energy
    except Exception:
        print("Minimization failed on building model")
        print(
            f"potential energy was {simulation.context.getState(getEnergy=True).getPotentialEnergy()}"
        )
        print("initial positions were:")
        print(positions)

    return simulation


def run_simulation(
    cgmodel,
    total_simulation_time,
    simulation_time_step,
    temperature,
    friction=1.0 / unit.picosecond,
    print_frequency=1000,
    minimize=True,
    output_directory="output",
    output_traj="simulation.pdb",
    output_data="simulation.dat",
):
    """

    Run OpenMM() simulation

    :param cgmodel: CGModel() object
    :type cgmodel: class

    :param total_simulation_time: Total run time for individual simulations
    :type total_simulation_time: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param simulation_time_step: Simulation integration time step
    :type simulation_time_step: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param temperature: Simulation temperature, default = 300.0 K
    :type temperature: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_
	
    :param friction: Langevin thermostat friction coefficient, default = 1 / ps
    :type friction: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param minimize: Whether or not the structure is energy-minimized before simulating.
    :type minimize: bool

    :param print_frequency: Number of simulation steps to skip when writing to output, Default = 1000
    :type print_frequence: int

    :param output_directory: Output directory for simulation data
    :type output_directory: str

    :param output_traj: file to write the trajectory to (with .pdb or .dcd extension)
    :type output_traj: str

    :param output_data: file to write the output data as a function of time.
    :type output_data: str

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
    >>> output_traj = "output.pdb"
    >>> output_data = "output.dat"
    >>> print_frequency = 20
    >>> run_simulation(cgmodel,total_simulation_time,simulation_time_step,temperature,friction,print_frequency,output_directory=output_directory,minimize=True,output_traj=output_traj,output_data=output_data)

    .. warning:: When run with default options this subroutine is capable of producing a large number of output files.  For example, by default this subroutine will plot the simulation data that is written to an output file.

    """
    total_steps = int(np.floor(total_simulation_time / simulation_time_step))
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    output_traj = os.path.join(output_directory, output_traj)
    output_data = os.path.join(output_directory, output_data)

    simulation = build_mm_simulation(
        cgmodel.topology,
        cgmodel.system,
        cgmodel.positions,
        total_simulation_time=total_simulation_time,
        simulation_time_step=simulation_time_step,
        temperature=temperature,
        friction=friction,
        output_traj=output_traj,
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
        print(f"   temperature: {temperature}")
        print("3) Make sure that the initial/input structure is reasonable for the")
        print("   input set of model parameters.")
        exit()

    if not cgmodel.include_bond_forces and cgmodel.constrain_bonds and output_traj[-3:] == 'pdb':
        file = open(output_traj, "r")
        lines = file.readlines()
        file.close()
        os.remove(output_traj)
        file = open(output_traj, "w")
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
    pyplot.xlabel("Simulation Time (ps)")
    if plot_type == "Potential Energy":
        file_name = "Potential_Energy.pdf"
        pyplot.ylabel("Potential Energy (kJ/mol)")
        pyplot.title("Simulation Potential Energy")

    if plot_type == "Kinetic Energy":
        file_name = "Kinetic_Energy.pdf"
        pyplot.ylabel("Kinetic Energy (kJ/mol)")
        pyplot.title("Simulation Kinetic Energy")

    if plot_type == "Total Energy":
        file_name = "Total_Energy.pdf"
        pyplot.ylabel("Total Energy (kJ/mol)")
        pyplot.title("Simulation Total Energy")

    if plot_type == "Temperature":
        file_name = "Temperature.pdf"
        pyplot.ylabel("Temperature (K)")
        pyplot.title("Simulation Temperature")

    pyplot.plot(simulation_times, y_data)
    output_file = os.path.join(output_directory, file_name)
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
