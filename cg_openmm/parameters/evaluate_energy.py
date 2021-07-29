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
import multiprocessing as mp
import pickle


kB = unit.MOLAR_GAS_CONSTANT_R # Boltzmann constant

def eval_energy(cgmodel, file_list, temperature_list, param_dict,
    frame_begin=0, frame_end=-1, frame_stride=1, verbose=False, n_cpu=1):
    """
    Given a cgmodel with a topology and system, evaluate the energy at all structures in each
    trajectory files specified with updated force field parameters specified in param_dict.

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
    
    :param n_cpu: number of cpus for running parallel energy evaluations (default=1)
    :type n_cpu: int

    :returns:
        - U_eval - A numpy array of energies evaluated with the updated force field parameters [n_replicas x n_states x frames]
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
    
    # Is a single periodicity being turned into sums of multiple periodicities:
    sums_per_torsion = False
    # Multiple torsion parameters can be set as either a list of quantities, or a
    # quantity with a list as its value.
    # Multiple periodicities must be defined as a list however.

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
                            print(f'Particle: {particle_index}')
                            print(f'Old value: {sigma_old}')
                            print(f'New value: {param_dict[sigma_str].in_units_of(sigma_old.unit)}\n')

                    if epsilon_str in param_dict:
                        # Update the epsilon parameter for this particle:
                        (q,sigma,eps_old) = force.getParticleParameters(particle_index)
                        force.setParticleParameters(
                            particle_index,q,sigma,param_dict[epsilon_str].value_in_unit(unit.kilojoule_per_mole)
                        )
                        force.updateParametersInContext(simulation.context)
                        if verbose:
                            print(f'Updating parameter {epsilon_str}:')
                            print(f'Particle: {particle_index}')
                            print(f'Old value: {eps_old}')
                            print(f'New value: {param_dict[epsilon_str].in_units_of(eps_old.unit)}\n')

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
                            print(f'Particles: {par1} {par2}')
                            print(f'Old value: {k_old}')
                            print(f'New value: {param_dict[str_name].in_units_of(k_old.unit)}\n')

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
                            print(f'Particles: {par1} {par2}')
                            print(f'Old value: {k_old}')
                            print(f'New value: {param_dict[rev_str_name].in_units_of(k_old.unit)}\n')

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
                            print(f'Particles: {par1} {par2}')
                            print(f'Old value: {length_old}')
                            print(f'New value: {param_dict[str_name].in_units_of(length_old.unit)}\n')

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
                            print(f'Particles: {par1} {par2}')
                            print(f'Old value: {length_old}')
                            print(f'New value: {param_dict[rev_str_name].in_units_of(length_old.unit)}\n')

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
                            print(f'Particles: {par1} {par2} {par3}')
                            print(f'Old value: {k_old}')
                            print(f'New value: {param_dict[str_name].in_units_of(k_old.unit)}\n')

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
                            print(f'Particles: {par1} {par2} {par3}')
                            print(f'Old value: {k_old}')
                            print(f'New value: {param_dict[rev_str_name].in_units_of(k_old.unit)}\n')

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
                            print(f'Particles: {par1} {par2} {par3}')
                            print(f'Old value: {theta0_old}')
                            print(f'New value: {param_dict[str_name].in_units_of(theta0_old.unit)}\n')

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
                            print(f'Particles: {par1} {par2} {par3}')
                            print(f'Old value: {theta0_old}')
                            print(f'New value: {param_dict[rev_str_name].in_units_of(theta0_old.unit)}\n')

                    angle_index += 1

        elif force_name == 'PeriodicTorsionForce':
            # Update the periodic torsion parameters:
            if torsion_eq or torsion_k or torsion_per:                  
                # cgmodel torsion_list is based only on the topology, not the forces.
                # So for each torsion in torsion_list, we can loop over the different
                # periodicities.
                
                # There are three possible scenarios:
                # 1) Going from m --> n periodicities, where n > m
                # 2) No change in number of periodicities, but new parameters (n = m)
                # 3) Going from m --> n periodicities, where n < m.
                
                # We cannot add new torsions when doing updateParametersInContext
                # Hence, we need to create a new torsion force object
                
                # For now, only option 2 is supported. A fairly easy workaround
                # for n != m is to create a new cgmodel with the same topology,
                # but with the desired amount of periodic torsion terms.
                
                # Get the number of torsion forces in the original model:
                n_torsion_forces = force.getNumTorsions()
                
                # Store the original periodicities and openmm indices found for each particle sequence:
                periodicity_map = {}
                torsion_index_map = {}
                for torsion in torsion_list:
                    periodicity_map[f'{torsion}'] = []
                    torsion_index_map[f'{torsion}'] = []
                
                # Get the torsion list as ordered in OpenMM:
                # The order of particles within each torsion can be reversed in OpenMM vs cgmodel
                torsion_list_openmm = []
                for i_tor in range(n_torsion_forces):
                    (par1, par2, par3, par4, periodicity, phase, k) = \
                        force.getTorsionParameters(i_tor)
                    if [par1, par2, par3, par4] in torsion_list:
                        torsion_list_openmm.append([par1, par2, par3, par4])
                    else:
                        torsion_list_openmm.append([par4, par3, par2, par1])
                        
                    periodicity_map[f'{torsion_list_openmm[-1]}'].append(periodicity)
                    torsion_index_map[f'{torsion_list_openmm[-1]}'].append(i_tor)
            
                torsion_index = 0
                
                # Here we loop over the particle sequences, not the torsion forces:
                for torsion in torsion_list:
                    # These are indices of the 4 particles
                    # Force field parameter names are of the form:
                    # 'bb_bb_bb_bb_torsion_phase_angle'
                    # 'bb_bb_bb_bb_torsion_force_constant'
                    # 'bb_bb_bb_bb_torsion_periodicity'

                    #--------------------------------------------------#
                    # Check for updated torsion_periodicity parameters #
                    #--------------------------------------------------#
                    
                    name_1 = particle_type_list[torsion[0]]
                    name_2 = particle_type_list[torsion[1]]
                    name_3 = particle_type_list[torsion[2]]
                    name_4 = particle_type_list[torsion[3]]

                    suffix = 'torsion_periodicity'
                    str_name_per = f'{name_1}_{name_2}_{name_3}_{name_4}_{suffix}'
                    rev_str_name_per = f'{name_4}_{name_3}_{name_2}_{name_1}_{suffix}'
                    
                    suffix = 'torsion_force_constant'
                    str_name_kt = f'{name_1}_{name_2}_{name_3}_{name_4}_{suffix}'
                    rev_str_name_kt = f'{name_4}_{name_3}_{name_2}_{name_1}_{suffix}'
                    
                    suffix = 'torsion_phase_angle'
                    str_name_phi = f'{name_1}_{name_2}_{name_3}_{name_4}_{suffix}'
                    rev_str_name_phi = f'{name_4}_{name_3}_{name_2}_{name_1}_{suffix}'

                    # Check if this torsion needs updating:
                    if ((str_name_per in param_dict) or (rev_str_name_per in param_dict) or
                        (str_name_phi in param_dict) or (rev_str_name_phi in param_dict) or
                        (str_name_kt in param_dict) or (rev_str_name_kt in param_dict)):
                    
                        # Check number of new periodic torsion terms:
                        n_per_old = len(periodicity_map[f'{torsion}'])
                        if str_name_per in param_dict:
                            if type(param_dict[str_name_per]) == list:
                                n_per_new = len(param_dict[str_name_per]) # Multiple periodicity always given as list
                            else:
                                n_per_new = 1
                        elif rev_str_name_per in param_dict:
                            if type(param_dict[rev_str_name_per]) == list:
                                n_per_new = len(param_dict[rev_str_name_per])
                            else:
                                n_per_new = 1
                        else:
                            n_per_new = n_per_old
                            
                        if n_per_new > n_per_old:
                            # Increasing the number of periodic torsion terms, which is not yet supported:
                            print('Invalid sums of periodic torsion parameter update:')
                            print('Increasing the number of periodic torsion terms is not yet supported')
                            print(f'Original number of terms: {n_per_old}')
                            print(f'New number of terms: {n_per_new}')
                            exit()
                            
                        elif n_per_new < n_per_old:
                            # Reducing the number of periodic torsion terms, with non-supported input:
                            print('Invalid sums of periodic torsion parameter update:')
                            print('When reducing number of periodic torsion terms, set the deleted term')
                            print('to have zero force constant.')
                            print(f'Original number of terms: {n_per_old}')
                            print(f'New number of terms: {n_per_new}')
                            exit()
                            
                        elif n_per_new == n_per_old:
                            # Update each of the existing periodic torsion forces:
                            for i_per in range(n_per_old):
                                (par1, par2, par3, par4, periodicity_old, phase_old, k_old) = \
                                    force.getTorsionParameters(torsion_index_map[f'{torsion}'][i_per])

                                # Check the input style of phase angle and force constant:
                                # Each can independently be a list of quantities,
                                # or a quantity with list value.
                                
                                #--------------------------------------------------#
                                # Check for updated torsion_phase_angle parameters #
                                #--------------------------------------------------#
                                if str_name_phi in param_dict:
                                    if type(param_dict[str_name_phi]) == list:
                                        param_phi_curr = param_dict[str_name_phi][i_per]
                                        
                                    elif type(param_dict[str_name_phi]) == unit.quantity.Quantity:
                                        phi_unit = param_dict[str_name_phi].unit
                                        if type(param_dict[str_name_phi].value_in_unit(phi_unit)) == list:
                                            param_phi_curr = param_dict[str_name_phi].value_in_unit(phi_unit)[i_per] * phi_unit
                                        else:
                                            param_phi_curr = param_dict[str_name_phi]
                                
                                elif rev_str_name_phi in param_dict:
                                    if type(param_dict[rev_str_name_phi]) == list:
                                        param_phi_curr = param_dict[rev_str_name_phi][i_per]
                                        
                                    elif type(param_dict[rev_str_name_phi]) == unit.quantity.Quantity:
                                        phi_unit = param_dict[rev_str_name_phi].unit
                                        if type(param_dict[rev_str_name_phi].value_in_unit(phi_unit)) == list:
                                            param_phi_curr = param_dict[rev_str_name_phi].value_in_unit(phi_unit)[i_per] * phi_unit
                                        else:
                                            param_phi_curr = param_dict[rev_str_name_phi]

                                else:
                                    # Not updating this parameter, since it was not specified in parameter dictionary:
                                    param_phi_curr = phase_old
                                 
                                #-----------------------------------------------------#
                                # Check for updated torsion_force_constant parameters #
                                #-----------------------------------------------------#
                                if str_name_kt in param_dict:
                                    if type(param_dict[str_name_kt]) == list:
                                        param_kt_curr = param_dict[str_name_kt][i_per]
                                        
                                    elif type(param_dict[str_name_kt]) == unit.quantity.Quantity:
                                        # The value of this can be a list, or a single float
                                        kt_unit = param_dict[str_name_kt].unit
                                        if type(param_dict[str_name_kt].value_in_unit(kt_unit)) == list:
                                            param_kt_curr = param_dict[str_name_kt].value_in_unit(kt_unit)[i_per] * kt_unit
                                        else:
                                            param_kt_curr = param_dict[str_name_kt]
                                        
                                elif rev_str_name_kt in param_dict:
                                    if type(param_dict[rev_str_name_kt]) == list:
                                        param_kt_curr = param_dict[rev_str_name_kt][i_per]
                                        
                                    elif type(param_dict[rev_str_name_kt]) == unit.quantity.Quantity:
                                        # The value of this can be a list, or a single float
                                        kt_unit = param_dict[rev_str_name_kt].unit
                                        if type(param_dict[rev_str_name_kt].value_in_unit(kt_unit)) == list:
                                            param_kt_curr = param_dict[rev_str_name_kt].value_in_unit(kt_unit)[i_per]
                                        else:
                                            param_kt_curr = param_dict[rev_str_name_kt]
                                else:
                                    # Not updating this parameter, since it was not specified in parameter dictionary:
                                    param_kt_curr = k_old
                                  
                                #--------------------------------------------------#
                                # Check for updated torsion_periodicity parameters #
                                #--------------------------------------------------# 
                                if str_name_per in param_dict:
                                    if type(param_dict[str_name_per]) == list:
                                        param_per_curr = param_dict[str_name_per][i_per]
                                    else:
                                        param_per_curr = param_dict[str_name_per]
                                
                                elif rev_str_name_per in param_dict:
                                    if type(param_dict[rev_str_name_per]) == list:
                                        param_per_curr = param_dict[rev_str_name_per][i_per]
                                    else:
                                        param_per_curr = param_dict[rev_str_name_per]
                                
                                else:
                                    # Not updating this parameter, since it was not specified in parameter dictionary:
                                    param_per_curr = periodicity_old
                                
                                # Update the periodic torsion parameters:
                                force.setTorsionParameters(
                                    torsion_index_map[f'{torsion}'][i_per],
                                    par1, par2, par3, par4,
                                    param_per_curr,
                                    param_phi_curr,
                                    param_kt_curr,
                                ) 
                                
                                force.updateParametersInContext(simulation.context)
                                if verbose:
                                    print(f'\nUpdating sums of periodic torsions, type {name_1}_{name_2}_{name_3}_{name_4}:')
                                    print(f'Particles: {par1} {par2} {par3} {par4}')
                                    print(f'Periodicity: {periodicity_old} --> {param_per_curr}')
                                    print(f'Phase angle: {phase_old} --> {param_phi_curr.in_units_of(phase_old.unit)}')
                                    print(f'Force constant: {k_old} --> {param_kt_curr.in_units_of(k_old.unit)}')

                    torsion_index += 1
                    
                n_torsion_forces_new = force.getNumTorsions()
                if verbose:
                    print(f'\nOld total number of torsion force terms: {n_torsion_forces}')
                    print(f'New total number of torsion force terms: {n_torsion_forces_new}')

    # Update the positions and evaluate all specified frames:
    # Run with multiple processors and gather data from all replicas:        
    pool = mp.Pool(n_cpu)
    print(f'Using {n_cpu} CPU out of total available {mp.cpu_count()}')
    
    results = pool.starmap(get_replica_reeval_energies, 
        [(replica, temperature_list, file_list, cgmodel.topology, simulation.system,
        frame_begin, frame_stride, frame_end) for replica in range(len(file_list))])
    pool.close()
    
    # results is a list of tuples, each containing (U_kln_replica, replica_ID)
    # Actually, the replicas are ordered correctly within results regardless of the order in which they
    # are executed, but we can add a check to be sure.
    
    # Overall energy matrix (all replicas)                 
    U_eval = np.zeros((len(file_list),len(file_list),results[0][0].shape[1]))
             
    # This can be converted to the 2d array for MBAR with kln_to_kn utility
    # The 3d array is organized in the same way as replica_energies extracted from
    # the .nc file.          
    
    # Assign replica energies:
    for i in range(len(file_list)):
        rep_id = results[i][1]
        U_eval[rep_id,:,:] = results[i][0]
    
    return U_eval, simulation   


def get_replica_reeval_energies(replica, temperature_list, file_list, topology, system,
    frame_begin, frame_stride, frame_end):
    """
    Internal function for evaluating energies for all specified frames in a replica trajectory.
    We can't use an inner nested function to do multiprocessor parallelization, since it needs to be pickled.
    """
    
    # Need to recreate Simulation object here:
    simulation_time_step = 5.0 * unit.femtosecond
    friction = 0.0 / unit.picosecond
    integrator = LangevinIntegrator(
        0.0 * unit.kelvin, friction, simulation_time_step.in_units_of(unit.picosecond)
    )
    simulation = Simulation(topology, system, integrator)
    
    # Load in the coordinates as mdtraj object:
    if file_list[replica][-3:] == 'dcd':
        traj = md.load(file_list[replica],top=md.Topology.from_openmm(topology))
    else:
        traj = md.load(file_list[replica])
        
    # Select frames to analyze:
    if frame_end < 0:
        traj = traj[frame_begin::frame_stride]
    else:
        traj = traj[frame_begin:frame_end:frame_stride]
        
    nframes = traj.n_frames
    print(f'Evaluating {nframes} frames (replica {replica})')

    # Local variable for replica energies:
    U_eval_rep = np.zeros((len(file_list),nframes))
    
    for k in range(nframes):
        positions = traj[k].xyz[0]*unit.nanometer
        # Compute potential energy for current frame, evaluating at all states
        simulation.context.setPositions(positions)
        potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()

        # Boltzmann factor for reduce the potential energy:
        beta_all = 1/(kB*temperature_list)

        # Evaluate this reduced energy at all thermodynamic states:
        for j in range(len(temperature_list)):
            U_eval_rep[j,k] = (potential_energy*beta_all[j])
    
    return U_eval_rep, replica 