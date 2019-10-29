import sys
import numpy as np
import math, random, statistics
import simtk.unit as unit
from cg_openmm.build.cg_build import *
from cg_openmm.simulation.tools import *
from cg_openmm.utilities.iotools import *
from foldamers.cg_model.cgmodel import *

def random_sign(number):
        """
        Returns the provided 'number' with a random sign.

        :param number: The number to add a random sign (positive or negative) to
        :type number: float

        :returns:
          - number (float) - The provided number with a random sign added

        """

        # 0 = negative, 1 = positive
        random_int = random.randint(0,1)
        if random_int == 0: number = -1.0 * number

        return(number)

def first_bead(positions):
        """
        Determine if the provided 'positions' contain any particles (are the coordinates non-zero).

        :param positions: Positions for all beads in the coarse-grained model.
        :type positions: np.array( float * unit ( shape = num_beads x 3 ) )
        
        :returns:
         - first_bead (Logical) - Variable stating if the positions are all non-zero.

        """

        first_bead = True
    
        if str(positions.shape) == '(2,)':
          return(first_bead)
        else:
          for value in positions._value:
            if any(i != 0. for i in value):
              first_bead = False

        return(first_bead)


def get_move(trial_coordinates,move_direction,distance,bond_length,finish_bond=False):
        """
        Used to build random structures.  Given a set of input coordinates, this function attempts to add a new particle.

        :param trial_coordinates: Positions for a particle
        :type trial_coordinates: np.array( float * unit.angstrom ( length = 3 ) )

        :param move_direction: Cartesian direction in which we will attempt a particle placement, where: x=0, y=1, z=2. 
        :type move_direction: int

        :param distance: Current distance between the trial coordinates for the particle this function is positioning and the particle that it is branched from (bonded to).
        :type distance: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :param bond_length: The distance to step before placing a new particle.
        :type bond_length: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :param finish_bond: Logical variable determining how we will update the coordinates for this particle, default = False.  If set to "True", the "move" length will be the difference between "distance" and "bond_length".
        :type finish_bond: 

        :returns:
         - trial_coordinates (np.array( float * unit.angstrom (length=3) )) - Updated positions for the particle.

        """

        if distance.__gt__(bond_length):
          print("ERROR: The particle distance is larger than the bond length.")
          exit()

        # Determine the 'max_step_size' as the square root of the difference
        # between 'bond_length' and 'distance'
#        print("Bond length is "+str(bond_length))
#        print("Distance is "+str(distance))
        max_step_size = bond_length.__pow__(2.0).__sub__(distance.__pow__(2.0)).sqrt()

        # Add a random sign to 'max_step_size', to randomize our particle placement.
        sign_index = random.randint(0,1)
        if sign_index == 0:
          max_step_size = max_step_size.__neg__()


        # If we are "finishing the bond", then the "step" size is
        # the length (in the direction 'move_direction') required
        # so that 'distance' == 'bond_length'
        if finish_bond:

          # Calculate the step size as 'step' = sqrt('difference')
          step = max_step_size._value

        # If we aren't "finishing the bond", then the "step" size
        # is a random float in the range 0.0
        if not finish_bond:

          step = random.uniform(0.0,max_step_size._value)

        # Add this 'step' to the existing coordinates
        # print("The trial coordinates are: "+str(trial_coordinates))
        # print("The step size is: "+str(step))
        trial_coordinates[move_direction] = trial_coordinates[move_direction].__add__(unit.Quantity(step,trial_coordinates.unit))
        # print("The trial coordinates are: "+str(trial_coordinates))

        return(trial_coordinates)

def attempt_lattice_move(parent_coordinates,bond_length,move_direction_list):
        """
        Given a set of cartesian coordinates this function positions a new particle a distance of 'bond_length' away in a random direction.

        :param parent_coordinates: Positions for a single particle, away from which we will place a new particle a distance of 'bond_length' away.
        :type parent_coordinates: np.array( float * unit.angstrom ( length = 3 ) )

        :param bond_length: Bond length for all beads that are bonded.
        :type bond_length: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :param move_direction_list: A list of cartesian directions (denoted by integers) that tracks the directions in which a particle placement has been attempted.
        :type move_direction_list: List( int )

        :returns:
          - trial_coordinates ( np.array( float * unit.angstrom ( length = 3 ) ) ) - The coordinates for a new, trial particle.
          - move_direction_list ( List(int) ) - A list of cartesian directions (denoted by integers) that tracks the directions in which a particle placement has been attempted.

        """

        units = bond_length.unit
        dist = unit.Quantity(0.0,units)

        # Assign the parent coordinates as the initial coordinates for a trial particle
        trial_coordinates = np.zeros([3]) * units
        try:
          parent_coordinates.unit
        except:
          parent_coordinates = parent_coordinates * units

        move_direction = random.randint(1,3)
        move_sign = random_sign(move_direction)
        while move_sign in move_direction_list:
         move_direction = random.randint(1,3)
         move_sign = random_sign(move_direction)
        move_direction_list.append(move_sign)
        if move_sign < 0:
         move_sign = -bond_length
        else:
         move_sign = bond_length
        #print("The trial coordinates are: "+str(trial_coordinates))
        #print("The parent coordinates are: "+str(parent_coordinates))
        trial_coordinates[:] = parent_coordinates[:]
        trial_coordinates[move_direction-1] = parent_coordinates[move_direction-1].__add__(move_sign)
        dist = distance(parent_coordinates,trial_coordinates)
        if round(dist._value,4) < round(bond_length._value,4):

           print("Error: particles are being placed at a distance different from the bond length")
           print("Bond length is: "+str(bond_length))
           print("The particle distance is: "+str(dist))
           print(ref)
           print(trial_coordinates)
           exit()

        return(trial_coordinates,move_direction_list)

def attempt_move(parent_coordinates,bond_length):
        """
        Given a set of cartesian coordinates, assign a new particle
        a distance of 'bond_length' away in a random direction.

        :param parent_coordinates: Positions for a single particle, away from which we will place a new particle a distance of 'bond_length' away.
        :type parent_coordinates: np.array( float * unit.angstrom ( length = 3 ) )

        :param bond_length: Bond length for all beads that are bonded.
        :type bond_length: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :returns:
          - trial_coordinates ( np.array( float * unit.angstrom ( length = 3 ) ) ) - The coordinates for a new, trial
particle.

        """

        units = bond_length.unit
        dist = unit.Quantity(0.0,units)

        move_direction_list = []

        # Assign the parent coordinates as the initial coordinates for a trial particle

        trial_coordinates = parent_coordinates.__deepcopy__(memo={})
        ref = parent_coordinates.__deepcopy__(memo={})

        for direction in range(3):
            move_direction = random.randint(0,2)
            while move_direction in move_direction_list:
              move_direction = random.randint(0,2)

            if float(round(bond_length._value**2.0,4)-round(dist._value**2.0,4)) < 0.0:

              print("The bond length is: "+str(round(bond_length._value**2.0,4)))
              print("The distance is: "+str(round(dist._value**2.0,4)))
              print("The parent coordinates are: "+str(ref))
              print("The trial coordinates are: "+str(trial_coordinates))
              print("Error: new particles are not being assigned correctly.")
              exit()

            if direction == 2:
              trial_coordinates = get_move(trial_coordinates,move_direction,dist,bond_length,finish_bond=True)

            else:
              trial_coordinates = get_move(trial_coordinates,move_direction,dist,bond_length)

            move_direction_list.append(move_direction)

#            print("Parent coordinates are: "+str(ref))
#            print("Trial coordinates are: "+str(trial_coordinates))
            dist = distance(ref,trial_coordinates)
#            print(direction)
#            print(dist)

        if round(dist._value,4) < round(bond_length._value,4):

           print("Error: particles are being placed at a distance different from the bond length")
           print("Bond length is: "+str(bond_length))
           print("The particle distance is: "+str(dist))
           print(ref)
           print(trial_coordinates)
           exit()

        return(trial_coordinates)

def distances(interaction_list,positions):
        """
        Calculate the distances between all non-bonded particles in a model, given a list of particle interactions and particle positions.

        :param interaction_list: A list of non-bonded particle interactions
        :type interaction_list: List( [ int, int ] )

        :param positions: Positions for the particles in a coarse grained model.
        :type positions: np.array( float * unit.angstrom ( num_particles x 3 ) )

        :returns:
           - distance_list ( List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) ) - A list of distances for the non-bonded interactions in the coarse grained model.

        """

        distance_list = []

#        print("In 'distances()' the positions are:"+str(positions))

        for interaction in interaction_list:
            if interaction[0] < len(positions) and interaction[1] < len(positions):
#             print("The distance between particles: "+str(interaction[0])+" and "+str(interaction[1])+" is "+str(distance(positions[interaction[0]],positions[interaction[1]])))
             distance_list.append(distance(positions[interaction[0]],positions[interaction[1]]))

        return(distance_list)

def collisions(positions,distance_list,distance_cutoff):
        """
        Determine if there are any collisions between non-bonded
        particles, where a "collision" is defined as a distance shorter than 'distance_cutoff'.

        :param positions: Positions for the particles in a coarse grained model.
        :type positions: np.array( float * unit.angstrom ( num_particles x 3 ) )

        :param distance_list: A list of distances.
        :type distance_list: List( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )

        :param distance_cutoff: The distance below which particles will be considered to have "collisions".
        :type distance_cutoff: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :returns:
          - collision (Logical) - A variable indicating whether or not the model contains particle collisions.

        """

        collision = False

        #print("The nonbonded cutoff distance is: "+str(distance_cutoff._value))
        if len(distance_list) > 0:
          #print(distance_list)
          for distance in distance_list:

            if round(distance._value,4) < round(distance_cutoff._value,4):

              collision = True

        if positions.shape != '(3,)':
         for position_1_index in range(len(positions)):
          for position_2_index in range(len(positions)):
           if position_1_index != position_2_index:
            if all([positions[position_1_index][i] == positions[position_2_index][i] for i in range(len(positions[position_1_index]))]):
             #print("Detected a collision between particles "+str(position_1_index)+" and "+str(position_2_index))
             collision = True

        return(collision)

def assign_position_lattice_style(cgmodel,positions,distance_cutoff,parent_bead_index,bead_index):
        """
        Assign random position for a particle

        :param cgmodel: CGModel() class object.
        :type cgmodel: class

        :param positions: Positions for the particles in a coarse grained model.
        :type positions: np.array( float * unit.angstrom ( num_particles x 3 ) )

        :param distance_cutoff: The distance below which particles will be considered to have "collisions".
        :type distance_cutoff: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :param parent_bead_index: The index of the particle from which we will bond a new particle, when assigning positions.
        :type parent_bead_index: int

        :param bead_index: The index of the particle for which the function will assign positions.
        :type bead_index: int

        :returns:
          - test_positions ( np.array( float * unit.angstrom ( num_particles x 3 ) ) ) - A set of positions for the updated model, including the particle that was just added.
          - success ( Logical ) - Indicates whether or not a particle was placed successfully.

        """
        #print("The first line positions are: "+str(positions))
        #print("The index for the bead we are trying to place is: "+str(bead_index))
        saved_positions = positions

        bond_list = cgmodel.get_bond_list()
        nonbonded_list = cgmodel.nonbonded_interaction_list

        #if str(positions.shape) == '(3,)':
          #parent_coordinates = positions
        #else:
          #print(parent_bead_index)
          #print(positions)
          #parent_coordinates = positions[parent_bead_index].__deepcopy__(memo={})

        success = False
        move_direction_list = []
        while len(move_direction_list) < 6 and not success:
           #print(len(move_direction_list))
           bond_length = cgmodel.get_bond_length(parent_bead_index,bead_index)
           #print("Calling attempt lattice move.")
           #print(move_direction_list)
           if str(positions.shape) == '(3,)':
            new_coordinates,move_direction_list = attempt_lattice_move(positions,bond_length,move_direction_list)
           else:
            #print(positions)
            #print(len(positions))
            #print(parent_bead_index)
            new_coordinates,move_direction_list = attempt_lattice_move(positions[parent_bead_index],bond_length,move_direction_list)
           if str(positions.shape) == '(3,)':
             test_positions = np.zeros([2,3]) * bond_length.unit
             test_positions[0] = positions
           else:
             test_positions = np.zeros([len(positions)+1,3]) * bond_length.unit
             test_positions[0:len(positions)] = positions[0:bead_index+2].__deepcopy__(memo={})
           #print("The provided positions were: "+str(positions))
           #print("The test positions before adding the trial coordinates are: "+str(test_positions))
           #print("The new coordinates are: "+str(new_coordinates))
           #print("The index for the bead we are trying to place is: "+str(bead_index))
           test_positions[bead_index] = new_coordinates
           #print("The test positions after adding the trial coordinates are: "+str(test_positions))
           #if len(test_positions) > 4: exit()
           nonbonded_distance_list = distances(nonbonded_list,test_positions)
           bonded_distance_list = distances(bond_list,test_positions)
           #print(nonbonded_list)
           #print(distance_list)
           if len(nonbonded_distance_list) == 0:
             success = True
             break
           if len(nonbonded_distance_list) > 0: 
            if not collisions(test_positions,nonbonded_distance_list,distance_cutoff) and not collisions(test_positions,bonded_distance_list,cgmodel.bond_lengths['bb_bb_bond_length']):
             success = True
             break 
        #print("Exited while loop in assign position lattice style.")       
        if success:
         if str(saved_positions.shape) != '(3,)':
          if all([test_positions[atom][index] == saved_positions[atom][index] for index in range(3) for atom in range(len(saved_positions))]) and len(test_positions) == len(saved_positions):
            print("ERROR: the input coordinates are the same as the output coordinates in 'assign_position_lattice_style()'.")
            print("The positions at the beginning of this function call were:"+str(saved_positions))
            print("The positions at the end of this function call are:"+str(positions))
            exit()
          else:
            positions = test_positions
            #print("Returing this set of positions: "+str(positions))
        #print(test_positions)
        return(test_positions,success)

def assign_position(positions,bond_length,distance_cutoff,parent_index,bead_index):
        """
        Assign random position for a bead

        :param positions: Positions for the particles in a coarse grained model.
        :type positions: np.array( float * unit.angstrom ( num_particles x 3 ) )

        :param bond_length: The distance to step when placing new particles.
        :type bond_length: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :param distance_cutoff: The distance below which particles will be considered to have "collisions".
        :type distance_cutoff: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :param parent_bead_index: The index of the particle from which we will bond a new particle, when assigning pos
itions.
        :type parent_bead_index: int

        :param bead_index: The index of the particle for which the function will assign positions.
        :type bead_index: int

        :returns:
          - positions ( np.array( float * unit.angstrom ( num_particles x 3 ) ) ) - A set of positions for the updated model, including the particle that was just added.
          - success ( Logical ) - Indicates whether or not a particle was placed successfully.

        """
        if bead_index == 1:
           success = True
           return(positions,success)

        units = bond_length.unit       
        if parent_index == -1:
               parent_index = bead_index - 1

        parent_coordinates = positions[parent_index-1]

        new_coordinates = unit.Quantity(np.zeros(3), units)
        success = False
        attempts = 0

        while not success:

           new_coordinates = attempt_move(parent_coordinates,bond_length)

           distance_list = distances(new_coordinates,positions)

           if not collisions(new_coordinates,distance_list,sigma): success = True

           if attempts > 100:
#            print([distance._value for distance in distance_list])
            return(positions,success)

           attempts = attempts + 1

        positions[bead_index-1] = new_coordinates

        return(positions,success)


def get_structure_from_library( cgmodel, high_energy=False, low_energy=False ):
        """
        Given a coarse grained model class object, this function retrieves
        a set of positions for the model from the 'foldamers' ensemble library, in:
        'foldamers/ensembles/${backbone_length}_${sidechain_length}_${sidechain_positions}'
        If this coarse grained model does not have an ensemble library, an 
        error message will be returned and positions at random with 'random_positions()'.

        cgmodel: CGModel() class object.

        :param high_energy: If set to 'True', this function will generate an ensemble of high-energy structures, default = False
        :type high_energy: Logical

        :param low_energy: If set to 'True', this function will generate an ensemble of low-energy structures, default = False
        :type low_energy: Logical

        :returns:
           - positions ( np.array( float * unit.angstrom ( num_particles x 3 ) ) ) - A set of coarse grained model positions.

        """
        ensemble_size = 10
        # In its current form this subroutine does not save the positions for random configurations we generate from heteropolymers.  It only saves the positions for homopolymers.

        if len(cgmodel.monomer_types) > 1:

          print("The foldamers ensemble library does not currently store conformations for polymers composed of more than one unique monomer.\n")
          print("The 'random_positions()' subroutine will be called instead, with 'use_library'=False.")
          positions = random_positions(cgmodel,use_library=False)
          return(positions)

        else:

          monomer_type = cgmodel.monomer_types[0]
          ensembles_directory = str(str(__file__.split('foldamers/utilities/util.py')[0])+"ensembles")
          if not os.path.exists(ensembles_directory):
            os.mkdir(ensembles_directory)
          model_directory = str(str(ensembles_directory)+"/"+str(cgmodel.polymer_length)+"_"+str(monomer_type['backbone_length'])+"_"+str(monomer_type['sidechain_length'])+"_"+str(monomer_type['sidechain_positions']))
          if not os.path.exists(model_directory):
            os.mkdir(model_directory)

          # We determine a suitable name for the ensemble directory by combining the 'bb_bb_bond_length', 'bb_sc_bond_length', and 'sc_sc_bond_length' into a single string:
          ens_str = [monomer_type['bond_lengths']['bb_bb_bond_length']._value,monomer_type['bond_lengths']['bb_sc_bond_length']._value,monomer_type['bond_lengths']['sc_sc_bond_length']._value]
          if not high_energy and not low_energy:
            ensemble_directory = str(str(model_directory)+"/bonds_"+str(ens_str[0])+"_"+str(ens_str[1])+"_"+str(ens_str[2]))
          if high_energy:
            ensemble_directory = str(str(model_directory)+"/bonds_"+str(ens_str[0])+"_"+str(ens_str[1])+"_"+str(ens_str[2])+"_high_energy")
          if low_energy:
            ensemble_directory = str(str(model_directory)+"/bonds_"+str(ens_str[0])+"_"+str(ens_str[1])+"_"+str(ens_str[2])+"_low_energy")

          generate_ensemble = False
          if not os.path.exists(ensemble_directory):
            os.mkdir(ensemble_directory)
            generate_ensemble = True
#            positions = random_positions(cgmodel,use_library=False)
          pdb_list = []
          energy_list = {'energy': [],'file_index': []}
          for file in os.listdir(ensemble_directory):
            if file.endswith('.pdb'):
              pdb_list.append(str(str(ensemble_directory)+"/"+str(file)))
              file_obj = open(str(str(ensemble_directory)+"/"+file),"r")
              #print(file_obj.readlines())
              if str('The OpenMM potential energy for this structure is:') in str(file_obj.readlines()):
                try:
                  energy = float(file_obj.readlines()[0].split(": ")[1].split(' ')[0])
                except:
                  energy = 9.9e9
              else:
                energy = 9.9e9
              energy_list['energy'].append(float(energy))
              file_index = int(file.split('cg')[1].split('.pdb')[0])
              energy_list['file_index'].append(file_index)
          if len(pdb_list) < ensemble_size:
            generate_ensemble = True

          if generate_ensemble:
#            print("The foldamers ensemble library only contains "+str(len(pdb_list))+" structures with these settings.\n")
#            print("The 'random_positions()' subroutine will be called instead, with 'use_library'=False,")
#            print("in order to generate a total of "+str(100)+" configurations for the database,")
#            print("before a specific configuration is chosen to assign random positions for this model.")
            if low_energy:
              total_iterations = 1000
            else:
              total_iterations = ensemble_size
            index = 1
            current_size = 1
            while index <= total_iterations and current_size < ensemble_size:
              file_name = str(ensemble_directory+"/cg"+str(current_size)+".pdb")
              if not os.path.exists(file_name):
                cgmodel.positions,cgmodel.simulation = random_positions(cgmodel,use_library=False)
                write_pdbfile_without_topology(cgmodel,file_name)
                cgmodel.topology = get_topology_from_pdbfile(file_name)
                cgmodel.system = build_system(cgmodel)
                simulation_time_step = 5.0 * unit.femtosecond
                positions_after,energy,simulation = minimize_structure(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature=300.0 * unit.kelvin,simulation_time_step=5.0 * unit.femtosecond,total_simulation_time=0.1 * unit.picosecond,print_frequency=10)
                cgmodel.positions = positions_after
                write_pdbfile_without_topology(cgmodel,file_name)

                if low_energy and any(energy._value < energy_list['energy'][i] for i in range(len(energy_list['energy']))) and len(energy_list['energy']) <= ensemble_size:
                    highest_energy_index = energy_list['energy'].index(max(energy_list['energy']))
                    print("Replacing the structure in "+str(str(ensemble_directory+"/cg"+str(energy_list['file_index'][highest_energy_index])+".pdb")))
                    print("Which had an energy of: "+str(energy_list['energy'][highest_energy_index]))
                    print("with a new structure that has an energy of: "+str(energy._value)) 
                    energy_list['energy'][highest_energy_index] = energy._value
                    write_pdbfile_without_topology(cgmodel,str(ensemble_directory+"/cg"+str(energy_list['file_index'][highest_energy_index])+".pdb"),energy=energy)

                else:

                    if len(energy_list['energy']) <= ensemble_size:
                      for step in range(len(energy_list['energy'])):
                        file_name = str(ensemble_directory+"/cg"+str(step+1)+".pdb")
                        if not os.path.exists(file_name):
                          energy_list['energy'].append(float(energy._value))
                          energy_list['file_index'].append(step+1)
                      write_pdbfile_without_topology(cgmodel,file_name,energy=energy)
                if len(energy_list['energy']) > 1:
                    print("The average energy for the new ensemble is: "+str(statistics.mean(energy_list['energy'])))

                index = index + 1
                current_size = current_size + 1

              else:
                
                index = index + 1
                current_size = current_size + 1

        pdb_list = []
        for file in os.listdir(ensemble_directory):
          if file.endswith('.pdb'):
            pdb_list.append(str(str(ensemble_directory)+"/"+str(file)))
        if len(pdb_list) > 0:
          random_file = pdb_list[random.randint(0,len(pdb_list)-1)]
        cgmodel.positions = get_positions_from_pdbfile(random_file)
        try:
         cgmodel.simulation = simulation
        except:
         cgmodel.system = build_system(cgmodel)
         positions,energy,simulation = minimize_structure(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature=300.0 * unit.kelvin,simulation_time_step=5.0 * unit.femtosecond,total_simulation_time=0.1 * unit.picosecond,print_frequency=10)

        return(positions)

def get_random_positions( cgmodel,max_attempts=1000,use_library=False,high_energy=False,low_energy=False,generate_library=False):
        """
        Assign random positions for all beads in a coarse-grained polymer.

        cgmodel: CGModel() class object.

        :param max_attempts: The maximum number of attempts to generate random positions a coarse grained model with the current parameters, default = 1000
        :type max_attempts: int
       
        :param use_library: A logical variable determining if a new random structure will be generated, or if an ensemble will be read from the 'foldamers' database, default = False
        :param use_library: Logical

        :param high_energy: If set to 'True', this function will generate an ensemble of high-energy structures, default = False
        :type high_energy: Logical

        :param low_energy: If set to 'True', this function will generate an ensemble of low-energy structures, default = False
        :type low_energy: Logical

        :param generate_library: If set to 'True', this function will save the poses that are generated to the 'foldamers' ensemble database.
        :type generate_libary: False

        :returns:
           - positions ( np.array( float * unit.angstrom ( num_particles x 3 ) ) ) - A set of coarse grained model positions.

        """

        if high_energy and low_energy:
            print("ERROR: Both 'high_energy' and 'low_energy' ensembles were requested in 'get_ensemble()'.  Please set only one of these variables to 'True', and call the function again.")
            exit()


        total_attempts = 0
        if use_library:
#            print("Attempting to find a suitable random starting configuration in the foldamers structural database.\n")
            positions,simulation = get_structure_from_library(cgmodel,high_energy=high_energy,low_energy=low_energy)
            return(positions)

        units = cgmodel.bond_lengths['bb_bb_bond_length'].unit
        positions = np.array([[0.0,0.0,0.0] for bead in range(cgmodel.num_beads)]) * units
        bond_list = cgmodel.get_bond_list()
        total_attempts = 0
        distance_cutoff =  0.99 * cgmodel.bond_lengths['bb_bb_bond_length']
        nonbonded_list = cgmodel.nonbonded_interaction_list
        lattice_style = True
        stored_positions = positions[0].__deepcopy__(memo={})
        while total_attempts < max_attempts and len(stored_positions) != len(positions):
         #print("Beginning build attempt #"+str(total_attempts+1)+".",flush=True)
         if str(positions.shape) == '(3,)':
           stored_positions = positions
         else:
           stored_positions = positions[0].__deepcopy__(memo={})

         bead_index = 0
         for monomer_index in range(cgmodel.polymer_length): 
          monomer_type = cgmodel.monomer_types[monomer_index]
          num_beads_in_monomer = monomer_type['num_beads']
          monomer_bead_list = [i for i in range(bead_index,bead_index+num_beads_in_monomer)]
          completed_list = []
          print(monomer_bead_list)
          while completed_list.sort() != monomer_bead_list:
           
           if bead_index == 0:
            completed_list.append(bead_index)

           if bead_index != 0:

            for bond_index in range(len(bond_list)):
             bond = bond_list[bond_index]
           
             if (bond[0] in monomer_bead_list and bond[1] not in monomer_bead_list) or (bond[1] in monomer_bead_list and bond[0] not in monomer_bead_list):

              if bond[0] not in completed_list and bond[0] == bead_index: completed_list.append(bond[0])
              if bond[1] not in completed_list and bond[1] == bead_index: completed_list.append(bond[1])

              if lattice_style:

               if bond[1] > bond[0]:
                positions,placement = assign_position_lattice_style(cgmodel,stored_positions,distance_cutoff,bond[0],bond[1])
               if bond[0] > bond[1]:
                positions,placement = assign_position_lattice_style(cgmodel,stored_positions,distance_cutoff,bond[1],bond[0])

              else:

               positions,placement = assign_position(stored_positions,cgmodel.bond_length,distance_cutoff,bond[1],bond[0])
 
              if not placement:

               if len(positions) < 3 and cgmodel.num_beads > 6:
                print("ERROR: having difficulty assigning random coordinates for this model,")
                print("even with a small number of particles ( <= 4 ).")
                print("The current positions are: "+str(positions))
                print("Please check the parameter values and other settings for your coarse grained model.")
                print("Then check the 'random_positions()' subroutine for errors.")
                exit()
               if bond_index > 4: start = bond_index - 2
               else: start = 2
               #print("Restarting random configuration build from bond #"+str(start)) 
               #print("This attempt produced a model with length= "+str(len(positions)))
               total_attempts = total_attempts + 1
               stored_positions = positions[0:min(bond_list[start])].__deepcopy__(memo={})
               #print("The new positions are:"+str(stored_positions)+"\n")
               break
              if placement:
               if str(stored_positions.shape) != '(3,)':
                if int(len(stored_positions)) != int(len(positions)-1):
                 print("ERROR: coordinates are not being added correctly in the 'random_positions()' subroutine.")
                 print("The positions before appension are:"+str(stored_positions))
                 print("The positions after appension are:"+str(positions))
                 exit()
                stored_positions = positions[0:max(bond_list[bond_index])+1].__deepcopy__(memo={})
               else:
                stored_positions = positions
          distance_cutoff =  0.99 * cgmodel.bond_lengths['bb_bb_bond_length']
          polymer_length = monomer_index+1
          test_cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=cgmodel.backbone_lengths,sidechain_lengths=cgmodel.sidechain_lengths,sidechain_positions=cgmodel.sidechain_positions,masses=cgmodel.masses,sigmas=cgmodel.sigmas,epsilons=cgmodel.epsilons,bond_lengths=cgmodel.bond_lengths,bond_force_constants=cgmodel.bond_force_constants,bond_angle_force_constants=cgmodel.bond_angle_force_constants,torsion_force_constants=cgmodel.torsion_force_constants,equil_bond_angles=cgmodel.equil_bond_angles,equil_torsion_angles=cgmodel.equil_torsion_angles,torsion_periodicities=cgmodel.torsion_periodicities,include_nonbonded_forces=cgmodel.include_nonbonded_forces,include_bond_forces=cgmodel.include_bond_forces,include_bond_angle_forces=cgmodel.include_bond_angle_forces,include_torsion_forces=cgmodel.include_torsion_forces,constrain_bonds=cgmodel.constrain_bonds)
          nonbonded_list = test_cgmodel.nonbonded_interaction_list

          distance_list = distances(nonbonded_list,positions)
          if len(distance_list) == 0 or not collisions(positions,distance_list,distance_cutoff):
            test_cgmodel.positions = positions
            test_cgmodel.topology = build_topology(cgmodel,use_pdbfile=True)
            test_cgmodel.system = build_system(cgmodel)
            positions,energy,simulation = minimize_structure(test_cgmodel.topology,test_cgmodel.system,positions,temperature=1.0 * unit.kelvin,simulation_time_step=5.0 * unit.femtosecond,total_simulation_time=0.1 * unit.picosecond,print_frequency=10)
        


        nonbonded_list = cgmodel.nonbonded_interaction_list
        distance_list = distances(nonbonded_list,positions)
        if len(distance_list) == 0 or not collisions(positions,distance_list,distance_cutoff):
          cgmodel.positions = positions
          cgmodel.topology = build_topology(cgmodel,use_pdbfile=True)
          cgmodel.system = build_system(cgmodel)
          positions,energy,simulation = minimize_structure(cgmodel.topology,cgmodel.system,positions,temperature=1.0 * unit.kelvin,simulation_time_step=5.0 * unit.femtosecond,total_simulation_time=0.1 * unit.picosecond,print_frequency=10)

          return(positions)

        else:

          print("Error: A model was successfully built, however,")
          print("particle collisions were detected.\n")
          print("The shortest nonbonded particle distance is:")
          print(str(min(distance_list)))
          #collision = nonbonded_list[distance_list.index(min(distance_list))]
          print("The nonbonded particle cutoff distance used for")
          print("random structure generation is set to:"+str(distance_cutoff))
          print("the positions are: "+str(positions))
          simulation = None
#          print("This will continue until the user issues a disruption command with the keyboard. (Ctrl + c)")
        return(positions)

def distance(positions_1,positions_2):
        """
        Calculate the distance between two particles.

        :param positions_1: Positions for a particle
        :type positions_1: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :param positions_2: Positions for a particle
        :type positions_2: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :returns:
           - distance ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The distance between the provided particles.

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


def distance_matrix(positions):
        """
        Construct a matrix of the distances between an input array of particles.

        :param positions: Positions for an array of particles.
        :type positions: np.array( float * unit.angstrom ( num_particles x 3 ) )

        :returns:
         - distance_matrix (np.array(num_particles x num_particles)) - Matrix containing the distances between all beads.
        """

        distance_matrix = np.array([[0.0 for index in range(0,len(positions))] for index in range(0,len(positions))])

        for index_1 in range(0,len(positions)):
          for index_2 in range(0,len(positions)):
            distance_matrix[index_1][index_2] = get_distance(positions[index_1],positions[index_2])

        return(distance_matrix)   
