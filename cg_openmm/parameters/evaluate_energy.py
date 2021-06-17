import os
import numpy as np
import simtk.unit as unit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cg_openmm.simulation.tools import get_mm_energy
from simtk.openmm.app import *
from simtk.openmm import *
import pymbar
import mdtraj as md

kB = unit.MOLAR_GAS_CONSTANT_R # Boltzmann constant

def eval_energy(cgmodel, file_list, temperature_list, param_dict,
    frame_begin=0, frame_end=-1, frame_stride=1, verbose=False):
    """
    Given a cgmodel with a topology and system, evaluate the energy at all structures in each
    trajectory files specified.

    :param cgmodel: CGModel() class object to evaluate energy with
    :type cgmodel: class

    :param file_list: List of replica trajectory files to evaluate the energies of
    :type file_list: list or str

    :param temperature_list: List of temperatures associated with file_list
    :type temperature_list: List( float * simtk.unit.temperature )

    :param param_dict: dictionary containing parameter scanning instructions
    :type param_dict: dict{'param_name': new_value * simtk.unit}

    :param frame_begin: analyze starting from this frame, discarding all prior as equilibration period (default=0)
    :type frame_begin: int

    :param frame_end: analyze up to this frame only, discarding the rest (default=-1).
    :type frame_end: int

    :param frame_stride: advance by this many frames between each evaluation (default=1)
    :type frame_stride: int

    :param verbose: option to print out detailed per-particle parameter changes (default=False)
    :type verbose: Boolean

    :returns:
        - U_eval - A numpy array of energies evaluated with the given cgmodel [n_replicas x n_states x frames]
        - simulation - OpenMM Simulation object containing the updated force field parameters
    """

    # Check input
    if len(file_list) != len(temperature_list):
        print('Error: mismatch between number of files and number of temperatures given.')
        exit()

    Tunit = temperature_list.unit

    # Determine which types of force field parameters are being varied:
    nonbond_sigma = False
    nonbond_epsilon = False

    bond_k = False
    bond_eq = False

    angle_k = False
    angle_eq = False

    torsion_k = False
    torsion_eq = False
    torsion_per = False

    for param_name, val in param_dict.items():
        # Search which parameters are being changed:
        if param_name.find('sigma'):
            nonbond_sigma = True

        if param_name.find('epsilon'):
            nonbond_epsilon = True

        if param_name.find('bond_force_constant'):
            bond_k = True

        if param_name.find('bond_length'):
            bond_eq = True

        if param_name.find('bond_angle_force_constant'):
            angle_k = True

        if param_name.find('equil_bond_angle'):
            angle_eq = True

        if param_name.find('torsion_force_constant'):
            torsion_k = True

        if param_name.find('torsion_phase_angle'):
            torsion_eq = True

        if param_name.find('torsion_periodicity'):
            torsion_per = True

    # Set up Simulation object beforehand:
    simulation_time_step = 5.0 * unit.femtosecond
    friction = 0.0 / unit.picosecond
    integrator = LangevinIntegrator(
        0.0 * unit.kelvin, friction, simulation_time_step.in_units_of(unit.picosecond)
    )
    simulation = Simulation(cgmodel.topology, cgmodel.system, integrator)

    # Get particle list:
    particle_list = cgmodel.create_particle_list()
    particle_type_list = []

    # Use particle indices rather than the full particle dictionary:
    for par in range(len(particle_list)):
        particle_type_list.append(cgmodel.get_particle_type_name(par))

    # Get bond list:
    bond_list = cgmodel.get_bond_list()

    # Get angle list:
    angle_list = cgmodel.get_bond_angle_list()

    # Get torsion list:
    torsion_list = cgmodel.get_torsion_list()

    for force_index, force in enumerate(simulation.system.getForces()):
        # These are the overall classes of forces, not the particle-specific forces
        force_name = force.__class__.__name__

        if force_name == 'NonbondedForce':
            if nonbond_epsilon or nonbond_sigma:
                # Update the nonbonded parameters:
                # 'sigma' and 'epsilon' are contained in the particle dictionaries

                for particle_index in range(len(particle_type_list)):
                    name = particle_type_list[particle_index]
                    sigma_str = f'{name}_sigma'
                    epsilon_str = f'{name}_epsilon'
                    if sigma_str in param_dict:
                        # Update the sigma parameter for this particle:
                        (q,sigma_old,eps) = force.getParticleParameters(particle_index)
                        force.setParticleParameters(
                            particle_index, q, param_dict[sigma_str].value_in_unit(unit.nanometer), eps
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {sigma_str}:')
                            print(f'cgmodel particle: {particle_index}')
                            print(f'Old value: {sigma_old}')
                            print(f'New value: {param_dict[sigma_str]}\n')

                    if epsilon_str in param_dict:
                        # Update the epsilon parameter for this particle:
                        (q,sigma,eps_old) = force.getParticleParameters(particle_index)
                        force.setParticleParameters(
                            particle_index,q,sigma,param_dict[epsilon_str].value_in_unit(unit.kilojoule_per_mole)
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {epsilon_str}:')
                            print(f'cgmodel particle: {particle_index}')
                            print(f'Old value: {eps_old}')
                            print(f'New value: {param_dict[epsilon_str]}\n')

        elif force_name == 'HarmonicBondForce':
            if bond_eq or bond_k:
                # Update the harmonic bond parameters:
                bond_index = 0
                for bond in bond_list:
                    # These are indices of the 2 particles
                    # Force field parameters are of the form:
                    # 'bb_bb_bond_force_constant'
                    # 'bb_bb_bond_length'

                    #--------------------------------------------------#
                    # Check for updated bond_force_constant parameters #
                    #--------------------------------------------------#
                    suffix = 'bond_force_constant'
                    name_1 = particle_type_list[bond[0]]
                    name_2 = particle_type_list[bond[1]]

                    str_name = f'{name_1}_{name_2}_{suffix}'
                    rev_str_name = f'{name_2}_{name_1}_{suffix}'

                    if str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, length, k_old) = force.getBondParameters(bond_index)
                        force.setBondParameters(
                            bond_index, par1, par2,
                            length, param_dict[str_name].value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2),
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {str_name}:')
                            print(f'cgmodel particles: {par1} {par2}')
                            print(f'Old value: {k_old}')
                            print(f'New value: {param_dict[str_name]}\n')

                    elif rev_str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, length, k_old) = force.getBondParameters(bond_index)
                        force.setBondParameters(
                            bond_index, par1, par2,
                            length, param_dict[rev_str_name].value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2),
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {rev_str_name}:')
                            print(f'cgmodel particles: {par1} {par2}')
                            print(f'Old value: {k_old}')
                            print(f'New value: {param_dict[rev_str_name]}\n')

                    #------------------------------------------#
                    # Check for updated bond_length parameters #
                    #------------------------------------------#
                    suffix = 'bond_length'

                    str_name = f'{name_1}_{name_2}_{suffix}'
                    rev_str_name = f'{name_2}_{name_1}_{suffix}'

                    if str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, length_old, k) = force.getBondParameters(bond_index)
                        force.setBondParameters(
                            bond_index, par1, par2,
                            param_dict[str_name].value_in_unit(unit.nanometer), k,
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {str_name}:')
                            print(f'cgmodel particles: {par1} {par2}')
                            print(f'Old value: {length_old}')
                            print(f'New value: {param_dict[str_name]}\n')

                    elif rev_str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, length_old, k) = force.getBondParameters(bond_index)
                        force.setBondParameters(
                            bond_index, par1, par2,
                            param_dict[rev_str_name].value_in_unit(unit.nanometer), k,
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {rev_str_name}:')
                            print(f'cgmodel particles: {par1} {par2}')
                            print(f'Old value: {length_old}')
                            print(f'New value: {param_dict[rev_str_name]}\n')

                    bond_index += 1

        elif force_name == 'HarmonicAngleForce':
            # Update the harmonic angle parameters:
            if angle_eq or angle_k:
                angle_index = 0
                for angle in angle_list:
                    # These are indices of the 3 particles
                    # Force field parameter names are of the form:
                    # 'bb_bb_bb_equil_bond_angle'
                    # 'bb_bb_bb_bond_angle_force_constant'

                    #--------------------------------------------------------#
                    # Check for updated bond_angle_force_constant parameters #
                    #--------------------------------------------------------#
                    suffix = 'bond_angle_force_constant'
                    name_1 = particle_type_list[angle[0]]
                    name_2 = particle_type_list[angle[1]]
                    name_3 = particle_type_list[angle[2]]

                    str_name = f'{name_1}_{name_2}_{name_3}_{suffix}'
                    rev_str_name = f'{name_3}_{name_2}_{name_1}_{suffix}'

                    if str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, par3, theta0, k_old) = force.getAngleParameters(angle_index)
                        force.setAngleParameters(
                            angle_index, par1, par2, par3,
                            theta0, param_dict[str_name].value_in_unit(unit.kilojoule_per_mole / unit.radian**2),
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {str_name}:')
                            print(f'cgmodel particles: {par1} {par2} {par3}')
                            print(f'Old value: {k_old}')
                            print(f'New value: {param_dict[str_name]}\n')

                    elif rev_str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, par3, theta0, k_old) = force.getAngleParameters(angle_index)
                        force.setAngleParameters(
                            angle_index, par1, par2, par3,
                            theta0, param_dict[rev_str_name].value_in_unit(unit.kilojoule_per_mole / unit.radian**2),
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {rev_str_name}:')
                            print(f'cgmodel particles: {par1} {par2} {par3}')
                            print(f'Old value: {k_old}')
                            print(f'New value: {param_dict[rev_str_name]}\n')

                    #-----------------------------------------------#
                    # Check for updated equil_bond_angle parameters #
                    #-----------------------------------------------#
                    suffix = 'equil_bond_angle'

                    str_name = f'{name_1}_{name_2}_{name_3}_{suffix}'
                    rev_str_name = f'{name_3}_{name_2}_{name_1}_{suffix}'

                    if str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, par3, theta0_old, k) = force.getAngleParameters(angle_index)
                        force.setAngleParameters(
                            angle_index, par1, par2, par3,
                            param_dict[str_name].value_in_unit(unit.radian), k,
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {str_name}:')
                            print(f'cgmodel particles: {par1} {par2} {par3}')
                            print(f'Old value: {theta0_old}')
                            print(f'New value: {param_dict[str_name]}\n')

                    elif rev_str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, par3, theta0_old, k) = force.getAngleParameters(angle_index)
                        force.setAngleParameters(
                            angle_index, par1, par2, par3,
                            param_dict[rev_str_name].value_in_unit(unit.radian), k,
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {rev_str_name}:')
                            print(f'cgmodel particles: {par1} {par2} {par3}')
                            print(f'Old value: {theta0_old}')
                            print(f'New value: {param_dict[rev_str_name]}\n')

                    angle_index += 1

        elif force_name == 'PeriodicTorsionForce':
            # Update the periodic torsion parameters:
            if torsion_eq or torsion_k or torsion_per:

                # OpenMM seems to index the torsions differently than in the cgmodel.
                # Need to create a new list of torsions with the OpenMM indexing.

                torsion_list_openmm = []
                for i_tor in range(len(torsion_list)):
                    (par1, par2, par3, par4, periodicity, phase, k) = \
                        force.getTorsionParameters(i_tor)
                    torsion_list_openmm.append([par1, par2, par3, par4])

                torsion_index = 0
                for torsion in torsion_list_openmm:
                    # These are indices of the 4 particles
                    # Force field parameter names are of the form:
                    # 'bb_bb_bb_bb_torsion_phase_angle'
                    # 'bb_bb_bb_bb_torsion_force_constant'
                    # 'bb_bb_bb_bb_torsion_periodicity'

                    #-----------------------------------------------------#
                    # Check for updated torsion_force_constant parameters #
                    #-----------------------------------------------------#
                    suffix = 'torsion_force_constant'
                    name_1 = particle_type_list[torsion[0]]
                    name_2 = particle_type_list[torsion[1]]
                    name_3 = particle_type_list[torsion[2]]
                    name_4 = particle_type_list[torsion[3]]

                    str_name = f'{name_1}_{name_2}_{name_3}_{name_4}_{suffix}'
                    rev_str_name = f'{name_4}_{name_3}_{name_2}_{name_1}_{suffix}'

                    if str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, par3, par4, periodicity, phase, k_old) = \
                            force.getTorsionParameters(torsion_index)

                        force.setTorsionParameters(
                            torsion_index, par1, par2, par3, par4,
                            periodicity, phase, param_dict[str_name].value_in_unit(unit.kilojoule_per_mole),
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {str_name}:')
                            print(f'OpenMM Particles: {par1} {par2} {par3} {par4}')
                            print(f'cgmodel particles: {torsion_list[torsion_index]}')
                            print(f'Old value: {k_old}')
                            print(f'New value: {param_dict[str_name]}\n')

                    elif rev_str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, par3, par4, periodicity, phase, k_old) = \
                            force.getTorsionParameters(torsion_index)

                        force.setTorsionParameters(
                            torsion_index, par1, par2, par3, par4,
                            periodicity, phase, param_dict[rev_str_name].value_in_unit(unit.kilojoule_per_mole),
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {rev_str_name}:')
                            print(f'OpenMM Particles: {par1} {par2} {par3} {par4}')
                            print(f'cgmodel particles: {torsion_list[torsion_index]}')
                            print(f'Old value: {k_old}')
                            print(f'New value: {param_dict[rev_str_name]}\n')


                    #--------------------------------------------------#
                    # Check for updated torsion_phase_angle parameters #
                    #--------------------------------------------------#
                    suffix = 'torsion_phase_angle'

                    str_name = f'{name_1}_{name_2}_{name_3}_{name_4}_{suffix}'
                    rev_str_name = f'{name_4}_{name_3}_{name_2}_{name_1}_{suffix}'

                    if str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, par3, par4, periodicity, phase_old, k) = \
                            force.getTorsionParameters(torsion_index)

                        force.setTorsionParameters(
                            torsion_index, par1, par2, par3, par4,
                            periodicity, param_dict[str_name].value_in_unit(unit.radian), k,
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {str_name}:')
                            print(f'OpenMM Particles: {par1} {par2} {par3} {par4}')
                            print(f'cgmodel particles: {torsion_list[torsion_index]}')
                            print(f'Old value: {phase_old}')
                            print(f'New value: {param_dict[str_name]}\n')

                    elif rev_str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, par3, par4, periodicity, phase_old, k) = \
                            force.getTorsionParameters(torsion_index)

                        force.setTorsionParameters(
                            torsion_index, par1, par2, par3, par4,
                            periodicity, param_dict[rev_str_name].value_in_unit(unit.radian), k,
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {rev_str_name}:')
                            print(f'OpenMM Particles: {par1} {par2} {par3} {par4}')
                            print(f'cgmodel particles: {torsion_list[torsion_index]}')
                            print(f'Old value: {phase_old}')
                            print(f'New value: {param_dict[rev_str_name]}\n')

                    #--------------------------------------------------#
                    # Check for updated torsion_periodicity parameters #
                    #--------------------------------------------------#
                    
                    # TODO: make this work for sums of periodic torsions
                    
                    suffix = 'torsion_periodicity'

                    str_name = f'{name_1}_{name_2}_{name_3}_{name_4}_{suffix}'
                    rev_str_name = f'{name_4}_{name_3}_{name_2}_{name_1}_{suffix}'

                    if str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, par3, par4, periodicity_old, phase, k) = \
                            force.getTorsionParameters(torsion_index)

                        force.setTorsionParameters(
                            torsion_index, par1, par2, par3, par4,
                            param_dict[str_name], phase, k,
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {str_name}:')
                            print(f'OpenMM Particles: {par1} {par2} {par3} {par4}')
                            print(f'cgmodel particles: {torsion_list[torsion_index]}')
                            print(f'Old value: {periodicity_old}')
                            print(f'New value: {param_dict[str_name]}\n')

                    elif rev_str_name in param_dict:
                        # Update this parameter:
                        (par1, par2, par3, par4, periodicity_old, phase, k) = \
                            force.getTorsionParameters(torsion_index)

                        force.setTorsionParameters(
                            torsion_index, par1, par2, par3, par4,
                            param_dict[rev_str_name], phase, k,
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {rev_str_name}:')
                            print(f'OpenMM Particles: {par1} {par2} {par3} {par4}')
                            print(f'cgmodel particles: {torsion_list[torsion_index]}')
                            print(f'Old value: {periodicity_old}')
                            print(f'New value: {param_dict[rev_str_name]}\n')

                    torsion_index += 1

    # Update the positions:
    for i in range(len(file_list)):
        # Load in the coordinates as mdtraj object:
        if file_list[i][-3:] == 'dcd':
            traj = md.load(file_list[i],top=md.Topology.from_openmm(cgmodel.topology))
        else:
            traj = md.load(file_list[i])

        # Select frames to analyze:
        if frame_end < 0:
            traj = traj[frame_begin::frame_stride]
        else:
            traj = traj[frame_begin:frame_end:frame_stride]

        if i == 0:
            nframes = traj.n_frames
            print(f'n_frames: {nframes}')
            U_eval = np.zeros((len(file_list),len(file_list),nframes))

        for k in range(nframes):
            positions = traj[k].xyz[0]*unit.nanometer
            # Compute potential energy for current frame, evaluating at all states
            simulation.context.setPositions(positions)
            potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()

            # Boltzmann factor for reduce the potential energy:
            beta_all = 1/(kB*temperature_list)

            # Evaluate this reduced energy at all thermodynamic states:
            for j in range(len(temperature_list)):
                U_eval[i,j,k] = (potential_energy*beta_all[j])

            # This can be converted to the 2d array for MBAR with kln_to_kn utility
            # The 3d array is organized in the same way as replica_energies extracted from
            # the .nc file.

    return U_eval, simulation

