import os
import numpy as np
import simtk.unit as unit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cg_openmm.simulation.tools import get_mm_energy
from cg_openmm.thermo.calc import *
from simtk.openmm.app import *
from simtk.openmm import *
import pymbar
import mdtraj as md
import multiprocessing as mp
import pickle
from itertools import combinations, combinations_with_replacement, product
import time
#from memory_profiler import profile

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
    
    # 1-4 interactions are nonbonded exceptions, and must be updated separately
    torsion_ends_list = []
    for torsion in torsion_list:
        torsion_ends_list.append([torsion[0],torsion[3]])

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

                # Update the sigma and epsilon parameters in relevant exceptions:
                n_nonbond_exceptions = force.getNumExceptions()
                n_exceptions_removed = 0
                for x in range(n_nonbond_exceptions):
                    (par1, par2, chargeProd, sigma_old, epsilon_old) = force.getExceptionParameters(x)
                    
                    # Only update the 1-4 exceptions. The 1-2 and 1-3 should retain a 0 epsilon value.
                    if [par1, par2] in torsion_ends_list or [par2, par1] in torsion_ends_list:
                        
                        name1 = particle_type_list[par1]
                        sigma_str1 = f'{name1}_sigma'
                        epsilon_str1 = f'{name1}_epsilon'
                        
                        name2 = particle_type_list[par2]
                        sigma_str2 = f'{name2}_sigma'
                        epsilon_str2 = f'{name2}_epsilon'
                        
                        if (sigma_str1 in param_dict or sigma_str2 in param_dict or 
                            epsilon_str1 in param_dict or epsilon_str2 in param_dict):
                            # Update this exception:
                            (q1,sigma1,eps1) = force.getParticleParameters(par1)
                            (q2,sigma2,eps2) = force.getParticleParameters(par2)
                        
                            sigma_ij = (sigma1+sigma2)/2
                            epsilon_ij = np.sqrt(eps1*eps2)
                            
                            # ***TODO: also update the charge product here
                            force.setExceptionParameters(x,par1,par2,0,sigma_ij,epsilon_ij)
                            if verbose:
                                print(f'Updating nonbonded exception {x}:')
                                print(f'Particles: {par1}, {par2}')
                                print(f'Old epsilon_ij: {epsilon_old}')
                                print(f'New epsilon_ij: {epsilon_ij}')
                                print(f'Old sigma_ij: {sigma_old}')
                                print(f'New sigma_ij: {sigma_ij}\n')
                                
                            # If epsilon_ij = 0, this will fail to update context since the interaction will be omitted
                            if epsilon_ij.value_in_unit(unit.kilojoule_per_mole) != 0:
                                force.updateParametersInContext(simulation.context)
                            else:
                                n_exceptions_removed += 1
                                
                if n_exceptions_removed > 0:
                    # Reinitialize context to include updates to the number of exceptions:
                    simulation.context.reinitialize()
                    if verbose:
                        print(f'Reinitializing context ({n_exceptions_removed} exceptions removed)')

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

    # Update the positions and evaluate all specified frames:
    # Run with multiple processors and gather data from all replicas:       

    if n_cpu > 1:
        pool = mp.Pool(n_cpu)
        print(f'Using {n_cpu} CPU out of total available {mp.cpu_count()}')
        
        results = pool.starmap(get_replica_reeval_energies, 
            [(replica, temperature_list, file_list, cgmodel.topology, simulation.system,
            frame_begin, frame_stride, frame_end) for replica in range(len(file_list))])
        pool.close()
            
        # results is a list of tuples, each containing (U_kln_replica, replica_ID)
        # Actually, the replicas are ordered correctly within results regardless of the order in which they
        # are executed, but we can add a check to be sure.    
             
        # This can be converted to the 2d array for MBAR with kln_to_kn utility
        # The 3d array is organized in the same way as replica_energies extracted from
        # the .nc file.    

        # Overall energy matrix (all replicas)                 
        U_eval = np.zeros((len(file_list),len(file_list),results[0][0].shape[1]))

        # Assign replica energies:
        for i in range(len(file_list)):
            rep_id = results[i][1]
            U_eval[rep_id,:,:] = results[i][0]        
            
    else:
        # Use the non-multiprocessor version to avoid potential issues:
        for replica in range(len(file_list)):
            U_eval_rep, rep_id = get_replica_reeval_energies(
                replica, temperature_list, file_list, cgmodel.topology, simulation.system,
                frame_begin, frame_stride, frame_end
                )
              
            if replica == 0:              
                # Overall energy matrix (all replicas)                 
                U_eval = np.zeros((len(file_list),len(file_list),U_eval_rep.shape[1]))
            
            U_eval[replica,:,:] = U_eval_rep          
    
    return U_eval, simulation   


def eval_energy_sequences(cgmodel, file_list, temperature_list, monomer_list, sequence=None,
    output_data='output/output.nc', num_intermediate_states=3, n_trial_boot=200, plot_dir='',
    frame_begin=0, frame_end=-1, sample_spacing=1, sparsify_stride=1, verbose=False, n_cpu=1):
    """
    Given a cgmodel with a topology and system, evaluate the energy at all structures in each
    trajectory file specified, with new monomer parameters and oligomer sequence(s) specified in monomer_list
    and sequence, respectively. Then, the new heat capacity curve and FWHM are calculated
    using MBAR reweighting.

    :param cgmodel: CGModel() class object to evaluate energy with
    :type cgmodel: class

    :param file_list: List of replica trajectory files to evaluate the energies of
    :type file_list: list or str

    :param temperature_list: List of temperatures associated with file_list
    :type temperature_list: List( float * simtk.unit.temperature )

    :param monomer_list: list of monomer type dicts (for formatting, see CGModel input)
    :type monomer_list: list ( dict {} )

    :param sequence: list of sequences to evaluate. Can be integer or monomer dict (0=monomer_list[0], 1=monomer_list[1],...) If None, all possible combinations will be attempted. (default=None)
    :type sequence: list(int) or list(list(int), list(int)...) or list(dict) or list(list(dict), list(dict)...)

    :param output_data: Path to NetCDF-formatted file containing the reference simulation data (default = "output/output.nc")                                                                                          
    :type output_data: str  

    :param num_intermediate_states: The number of states to insert between existing states in 'temperature_list' (default=3)
    :type num_intermediate_states: int

    :param n_trial_boot: number of trials to run for generating bootstrapping uncertainties. If None, a single heat capacity calculation will be performed. (default=200)
    :type n_trial_boot: int
    
    :param plot_dir: path to directory to which plot files will be saved (default='')
    :type plot_dir: str

    :param frame_begin: analyze starting from this frame, discarding all prior as equilibration period (default=0)
    :type frame_begin: int

    :param frame_end: analyze up to this frame only, discarding the rest (default=-1).
    :type frame_end: int

    :param sample_spacing: number of frames between decorrelated energies, before applying any sparsify_stride (default=1)
    :type frame_stride: int
    
    :param sparsify_stride: apply this stride to reduce the number of energies evaluated (default=1)
    :type sparsify_stride: int

    :param verbose: option to print out details on parameter changes and timing (default=False)
    :type verbose: Boolean
    
    :param n_cpu: number of cpus for running parallel energy evaluations (default=1)
    :type n_cpu: int

    :returns:
        - seq_FWHM - dictionary mapping sequences to their Cv full-width half-maximum
        - seq_FWHM_uncertainty - dictionary mapping sequences to uncertainty in FWHM (1 standard deviation)
        - seq_Cv - dictionary mapping sequences to their Cv vs. temperature curve
        - seq_Cv_uncertainty - dictionary mapping sequences to uncertainty in Cv
        - seq_N_eff - dictionary mapping sequences to MBAR number of effective samples for all states
    """

    # Check that the new monomer topologies are compatible with the reference cgmodel:
    cgmodel_mono0 = cgmodel.monomer_types[0]
    n_particle_per_mono_ref = len(cgmodel_mono0["particle_sequence"])
    bond_list_per_mono_ref = cgmodel_mono0["bond_list"]
    bond_start_mono_ref = cgmodel_mono0["start"]
    bond_end_mono_ref = cgmodel_mono0["end"]
    
    # Check topological consistency within the cgmodel:
    if len(cgmodel.monomer_types) > 1:
        for mono in cgmodel.monomer_types[1:]:
            if ((len(mono["particle_sequence"]) != n_particle_per_mono_ref) or
                (mono["bond_list"] != bond_list_per_mono_ref) or
                (mono["start"] != bond_start_mono_ref) or
                (mono["end"] != bond_end_mono_ref)):
                
                print('Error: residue types in the cgmodel must have the same topology for sequence scan')
                exit()
            
    # Check topological consistency in the new monomer types:
    for mono in monomer_list:
        if ((len(mono["particle_sequence"]) != n_particle_per_mono_ref) or
            (mono["bond_list"] != bond_list_per_mono_ref) or
            (mono["start"] != bond_start_mono_ref) or
            (mono["end"] != bond_end_mono_ref)):
            
            print('Error: new residue types must have the same topology as those in the reference cgmodel')
            exit()    
    
    # Check that sparsify_stride is not greater than sample_spacing
    if sparsify_stride > sample_spacing:
        print(f'Error: sparsify_stride ({sparsify_stride}) cannot be greater than sample_spacing ({sample_spacing})')
        exit()

    # Compute distance matrix for all nonbonded pairs with MDTraj
    nonbonded_list = cgmodel.get_nonbonded_interaction_list()
    
    # Get the nonbonded exclusion list
    nonbonded_exclusion_list = cgmodel.get_nonbonded_exclusion_list()
    
    # Determine the true nonbonded inclusion list
    nonbonded_inclusion_list = []
    
    for i in range(len(nonbonded_list)):
        par1 = nonbonded_list[i][0]
        par2 = nonbonded_list[i][1]
        
        if [par1,par2] not in nonbonded_exclusion_list and [par2,par1] not in nonbonded_exclusion_list:
            if par1 < par2:
                nonbonded_inclusion_list.append([par1,par2])
            else:
                nonbonded_inclusion_list.append([par2,par1])
    
    nonbonded_arr = np.asarray(nonbonded_inclusion_list) # Rows are each pair
    
    nonbond_power12_array = {} # r_ij^12
    nonbond_power6_array = {}  # r_ij^6   

    # These may be massive files - do each one separately and delete after computing distances
    distance_time_start = time.perf_counter()
    
    for i in range(len(file_list)):
        if file_list[0][-3:] == 'dcd':
            rep_traj = md.load(file_list[i],top=md.Topology.from_openmm(cgmodel.topology))
        else:
            rep_traj = md.load(file_list[i])
            
        # Select frames:
        if n_trial_boot is None:
            # Evaluate only decorrelated frames, with optional sparsifying stride:
            if frame_end > 0:
                rep_traj = rep_traj[frame_begin:frame_end:int(sample_spacing*sparsify_stride)]
            else:
                rep_traj = rep_traj[frame_begin::sample_spacing*sparsify_stride]
        else:
            # Evaluate all production frames with optional sparsifying stride:
            if frame_end > 0:
                rep_traj = rep_traj[frame_begin:frame_end:sparsify_stride]
            else:
                rep_traj = rep_traj[frame_begin::sparsify_stride]
            
        # Get the number of frames remaining:    
        if i == 0:
            nframes = rep_traj.n_frames      
       
        # This array assigned to dict is [n_frames x n_pairs]
        nonbond_dist_array = md.compute_distances(rep_traj,nonbonded_inclusion_list)
        
        # Now do element wise powers for the 2 LJ terms:
        nonbond_power12_array[i] = np.power(nonbond_dist_array,12)
        nonbond_power6_array[i] = np.power(nonbond_dist_array,6)
        
        # Delete trajectories and large distance arrays:
        del nonbond_dist_array
        del rep_traj
        
    distance_time_end = time.perf_counter()
    if verbose:
        print(f'distance calculations done ({distance_time_end-distance_time_start:.4f} s)')
         
    # Get the per-particle nonbonded parameters:
    sigma_list = []
    epsilon_list = []
    res_type_list = []
    param_dict_ref = {} # Use this to evaluate energies without nonbonded terms
    
    for mono in monomer_list:
        for particle in mono["particle_sequence"]:
            res_type_list.append(mono["monomer_name"])
            
            # These units must be nm, kJ/mol
            sigma_list.append(particle["sigma"].value_in_unit(unit.nanometer))
            epsilon_list.append(particle["epsilon"].value_in_unit(unit.kilojoules_per_mole))
    
    # Set all epsilon to zero using the original cgmodel, not the new monomer definitions:
    for particle in cgmodel.particle_type_list:
        param_dict_ref[f'{particle["particle_type_name"]}_epsilon'] = 0.0 * unit.kilojoule_per_mole
    
    bonded_eval_start = time.perf_counter()
    
    # Get the bonded energies:
    if n_trial_boot is None:
        U_bonded, simulation = eval_energy(cgmodel, file_list, temperature_list, param_dict_ref,
            frame_begin=frame_begin, frame_end=frame_end, frame_stride=int(sample_spacing*sparsify_stride), verbose=verbose, n_cpu=n_cpu)
    else:
        U_bonded, simulation = eval_energy(cgmodel, file_list, temperature_list, param_dict_ref,
            frame_begin=frame_begin, frame_end=frame_end, frame_stride=sparsify_stride, verbose=verbose, n_cpu=n_cpu)
        
    bonded_eval_end = time.perf_counter()
    if verbose:
        print(f'bonded energies done ({bonded_eval_end-bonded_eval_start:.4f} s)')
    
    seq_unique = []
    num_monomers = int(cgmodel.get_num_beads()/n_particle_per_mono_ref)
    
    if sequence is None:
        # Attempt to scan all possible combinations.
        # This assumes binary sequences, with all possible compositions
        # To enforce equal amounts, extra filtering step would go here
        # This can easily run out of memory - use only for small systems
        if len(monomer_list) > 2:
            print('Error: full sequence scan only supported for binary sequences')
            exit()
        
        elif len(monomer_list) == 2:
            for seq in list(product([0,1],repeat=num_monomers)):
                if seq not in seq_unique and tuple(reversed(seq)) not in seq_unique: # No reverse duplicates
                    # Compute new nonbonded energies
                    seq_unique.append(seq)
        else:
            # Homopolymer:
            seq = list(np.zeros(num_monomers))
            seq_unique.append(seq)
                    
    else:
        # Use a user-specified sequence or list of sequences
        if type(sequence[0]) == list:
            # Multiple sequences specified as list of lists
            for seq in sequence:
                if type(seq[0]) == dict:
                    seq_int = []
                    # Convert monomer dict to integers
                    for elem in seq:
                        i = 0
                        for mono in monomer_list:
                            if elem == mono:
                                seq_int.append(i)
                                break
                            else:
                                i += 1
                    seq_unique.append(seq_int)
                else:
                    # Use integer sequence
                    seq_unique.append(seq)
        elif type(sequence[0]) == dict:
            seq_int = []
            # Convert monomer dict to integers
            for elem in sequence:
                i = 0
                for mono in monomer_list:
                    if elem == mono:
                        seq_int.append(i)
                        break
                    else:
                        i += 1
            seq_unique.append(seq_int)
        
        elif type(sequence[0]) == int:
            # Sequence is list of integers,
            seq_unique.append(sequence)
            
        else:
            print(f'Invalid sequence input of type {type(sequence)} ({type(sequence[0])})')
            exit()
        
    if verbose:
        print(f'Evaluating {len(seq_unique)} sequences')
        
    # Determine all 4*epsilon_ij*(sigma_ij)^12
    # Determine all -4*epsilon_ij*(sigma_ij)^6
    
    def get_epsilon_ij(epsilon_ii, epsilon_jj):
        # Mixing rules for epsilon
        return np.sqrt(epsilon_ii*epsilon_jj)
        
    def get_sigma_ij(sigma_ii, sigma_jj):
        # Mixing rules for sigma
        return (sigma_ii+sigma_jj)/2

    LJ_12_term = np.zeros((len(sigma_list),len(sigma_list)))
    LJ_6_term = np.zeros_like(LJ_12_term)
    
    for i in range(len(sigma_list)):
        for j in range(len(sigma_list)):
            epsilon_ij = get_epsilon_ij(epsilon_list[i],epsilon_list[j])
            sigma_ij = get_sigma_ij(sigma_list[i],sigma_list[j])
            
            LJ_12_term[i,j] = 4*epsilon_ij*np.power(sigma_ij,12)
            LJ_6_term[i,j] = -4*epsilon_ij*np.power(sigma_ij,6)
            
    # Now evaluate energies at each sequence:
    # We need to apply the new interaction types to the original nonbonded pair list
    
    # Get the residue index and intra-residue particle index for all nonbonded pairs (constant for all sequences): 
    res_id_pairs, particle_type_pairs = np.divmod(nonbonded_arr,n_particle_per_mono_ref)
    
    # Boltzmann factor for reduce the potential energy:
    beta_all_kJ_mol = 1/(kB.in_units_of(unit.kilojoule_per_mole/unit.kelvin)*temperature_list)    
    
    # Set up results dicts:
    seq_Cv = {}
    seq_Cv_uncertainty = {}

    seq_Tm = {}
    seq_Tm_uncertainty = {}
    
    seq_Cv_height = {}
    seq_Cv_height_uncertainty = {}
    
    seq_FWHM = {}
    seq_FWHM_uncertainty = {}    
    
    seq_N_eff = {}
    
    for seq in seq_unique:
        seq_time_start = time.perf_counter()
        
        # Format the sequence for printing:
        seq_print = str(seq).replace(',','')
        seq_print = seq_print.replace(' ','')
        seq_print = seq_print.replace('[','')
        seq_print = seq_print.replace(']','')
        for s in range(len(monomer_list)):
            seq_print = seq_print.replace(str(s),monomer_list[s]["monomer_name"])
            
        if verbose:
            print(f'Evaluating sequence: {seq_print}')  
            
        # Set the heat capacity plot path:
        if plot_dir == None:
            # Don't create plot:
            plot_file_reeval = None
        elif plot_dir == '' or plot_dir == ' ':
            # Use the current working directory:
            plot_file_reeval = f"heat_capacity_{seq_print}.pdf"
        else:
            # Use the specified plot directory:
            if os.path.isdir(plot_dir):
                pass
            else:
                os.mkdir(plot_dir)
            plot_file_reeval = f"{plot_dir}/heat_capacity_{seq_print}.pdf"
        
        # Get residue types of nonbonded pairs:
        res_types = np.zeros((len(res_id_pairs),2))
        i = 0
        for pair in res_id_pairs:
            res_types[i,:] = np.array([seq[pair[0]],seq[pair[1]]])
            i += 1
            
        nonbond_types = particle_type_pairs + n_particle_per_mono_ref*res_types

        LJ_12_vec = np.zeros(len(nonbond_types))
        LJ_6_vec = np.zeros_like(LJ_12_vec)
        
        j = 0
        for pair in nonbond_types:
            LJ_12_vec[j] = LJ_12_term[int(pair[0]),int(pair[1])]
            LJ_6_vec[j] = LJ_6_term[int(pair[0]),int(pair[1])]
            j += 1
        
        # Do the vector math to get total nonbonded energy in each frame:
        U_eval_rep = {}
        for rep in range(len(file_list)):
            # Broadcast LJ vectors across all frames:
            nonbond_energy = LJ_12_vec/nonbond_power12_array[rep] + LJ_6_vec/nonbond_power6_array[rep]
            nonbond_energy = nonbond_energy.sum(axis=1)
            
            U_eval_rep[rep] = np.zeros((len(temperature_list),nframes))
            
            # Evaluate reduced energies at all states:
            for k in range(len(temperature_list)):
                U_eval_rep[rep][k,:] = (nonbond_energy*beta_all_kJ_mol[k])
              
        # Assign replica energies:
        U_eval = np.zeros((len(temperature_list),len(temperature_list),nframes))
        for rep in range(len(file_list)):
            U_eval[rep,:,:] = U_eval_rep[rep]
        
        # Finally, add the bonded energies:
        U_eval += U_bonded
        
        seq_time_energies_done = time.perf_counter()
        if verbose:
            print(f'nonbonded eval done ({seq_time_energies_done-seq_time_start:.4f} s)')
        
        # Now, evaluate the FWHM
        if n_trial_boot is None:
            # Single heat capacity calculation
            (Cv_sim, dCv_sim,
            C_v_values, C_v_uncertainty,
            T_list, FWHM_value,
            Tm_value, Cv_height_value,
            N_eff_values) = get_heat_capacity_reeval(
                U_eval,
                output_data=output_data,
                frame_begin=frame_begin,
                frame_end=frame_end,
                sample_spacing=int(sample_spacing*sparsify_stride),
                num_intermediate_states=num_intermediate_states,
                plot_file_reeval=plot_file_reeval,
                plot_file_sim=None,
            )
            
            Tm_uncertainty = None
            Cv_height_uncertainty = None
            FWHM_uncertainty = None
            
        else:
            # Use bootstrapping version of heat capacity calculation
            (T_list, C_v_values, C_v_uncertainty,
            Tm_value, Tm_uncertainty,
            Cv_height_value, Cv_height_uncertainty,
            FWHM_value, FWHM_uncertainty,
            N_eff_values) = bootstrap_heat_capacity(
                U_kln=U_eval,
                output_data=output_data,
                frame_begin=frame_begin,
                frame_end=frame_end,
                sample_spacing=sample_spacing,
                sparsify_stride=sparsify_stride,
                n_trial_boot=n_trial_boot,
                num_intermediate_states=num_intermediate_states,
                plot_file=plot_file_reeval,
            )
        
        seq_time_cv_done = time.perf_counter()
        if verbose:
            print(f'Cv eval done ({seq_time_cv_done-seq_time_energies_done:.4f} s)\n')
        
        seq_Cv[seq_print] = C_v_values
        seq_Cv_uncertainty[seq_print] = C_v_uncertainty
        
        seq_Tm[seq_print] = Tm_value
        seq_Tm_uncertainty[seq_print] = Tm_uncertainty
        
        seq_Cv_height[seq_print] = Cv_height_value
        seq_Cv_height_uncertainty[seq_print] = Cv_height_uncertainty        
        
        seq_FWHM[seq_print] = FWHM_value
        seq_FWHM_uncertainty[seq_print] = FWHM_uncertainty
        
        seq_N_eff[seq_print] = N_eff_values
    
    return seq_Cv, seq_Cv_uncertainty, seq_Tm, seq_Tm_uncertainty, seq_Cv_height, seq_Cv_height_uncertainty, seq_FWHM, seq_FWHM_uncertainty, seq_N_eff


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
            # This reducing step gets rid of the simtk units
            U_eval_rep[j,k] = (potential_energy*beta_all[j])
    
    return U_eval_rep, replica 