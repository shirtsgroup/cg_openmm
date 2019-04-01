## No default python environment

# This script generates random coordinates for a CG polymer

# =============================================================================================
# 1) PYTHON PACKAGE IMPORTS
# =============================================================================================

# System packages
import numpy as np
import math, random
from include.build_cg_model import distance
from simtk import unit
# =============================================================================================
# 2) ENVIRONMENT/JOB SETTINGS
# =============================================================================================

def random_sign(number):
# 0 = negative, 1 = positive
 random_int = random.randint(0,1)
 if random_int == 0: number = -1.0 * number
 return(number)

def mm_sqrt(argument):
 value = math.sqrt(argument._value)
 units = str(argument.unit).replace("**2.0","")
 sqrt=unit.Quantity(value,units)
 return(sqrt)

def append_position(positions,new_coordinate):
 positions_values = positions._value
 units = new_coordinate.unit
 new_coordinate = np.array([float(coord) for coord in new_coordinate._value])
 positions_values = np.vstack([positions_values,new_coordinate])
 new_positions = unit.Quantity(positions_values,units)
 return(new_positions)

def update_trial_coordinates(new_coordinate,direction,trial_coordinates=None):
 units = new_coordinate.unit
 if trial_coordinates == None:
  trial_coordinates = unit.Quantity(np.zeros([3]), units)
 else:
  value = (unit.Quantity(value=trial_coordinates._value[direction],unit=units).__add__(new_coordinate))._value
  values = [trial_coordinates._value[index] for index in range(0,3)]
  values[direction] = value
  trial_coordinates = unit.Quantity(values,units)
 return(trial_coordinates)

def first_atom(positions):
 first_atom = True
 for value in positions._value:
  if type(value) == np.ndarray:
   first_atom = False
 return(first_atom)

def get_move(move,direction,step):
     if direction != 2:
      new_jump = random_sign(random.uniform(0.0,step._value))
     if direction == 2:
      new_jump = random_sign(step._value)
     values = [value for value in move._value]
     values[direction] = new_jump
     move = unit.Quantity(values,move.unit)
     return(move)

def attempt_move(parent_coordinates,bond_length):
    units = parent_coordinates.unit
    move = unit.Quantity(np.zeros([3]), units)
    move_direction_list = []
    dist = unit.Quantity(0.0,units)
    if first_atom(parent_coordinates):
     trial_coordinates = parent_coordinates
    if not(first_atom(parent_coordinates)):
     trial_coordinates = unit.Quantity(parent_coordinates._value[len(parent_coordinates._value)-1],parent_coordinates.unit)
    ref = trial_coordinates
    for direction in range(0,3):
     move_direction = random.randint(0,2)
     while move_direction in move_direction_list:
      move_direction = random.randint(0,2)
     value = float(round(bond_length._value**2.0,4)-round(dist._value**2.0,4))
     if value < 0.0:
      print("Error: new particles are not being assigned correctly.")
      exit()
     units = dist.unit.__pow__(2.0)
     step_arg = unit.Quantity(value,units)
     step = mm_sqrt(step_arg)
     move = get_move(move,move_direction,step)
     move_direction_list.append(move_direction)
     trial_coordinates = update_trial_coordinates(move[move_direction],move_direction,trial_coordinates)
     dist = distance(ref,trial_coordinates)
    if round(distance(trial_coordinates,ref)._value,4) < round(bond_length._value,4):
     print("Error: particles are being placed at a distance different from the bond length")
     print("Bond length is: "+str(bond_length))
     print("The particle distance is: "+str(distance(trial_coordinates,ref)))
     print(ref)
     print(trial_coordinates)
     exit()
    return(trial_coordinates)

def non_bonded_distances(new_coordinates,existing_coordinates):
   distances = []
   if first_atom(existing_coordinates):
    return(distances)
   else:
    for particle in range(0,len(existing_coordinates)):
     test_position = unit.Quantity(existing_coordinates._value[particle],existing_coordinates.unit)
     distances.append(distance(new_coordinates,test_position))
   return(distances)

def collisions(distances,bond_length):
   collision = False
   if len(distances) > 0:
    for distance in distances:
     if round(distance._value,4) < round(bond_length._value,4):
      collision = True
   return(collision)

def assign_position(positions,bond_length,parent_index=-1):
  units = bond_length.unit
  if len(positions) == 0:
   new_coordinates = unit.Quantity(np.zeros([3]), units)
   return
  else:
   if parent_index == -1:
    parent_coordinates = positions
   if parent_index == 0:
    parent_coordinates = positions
   if parent_index > 0:
    parent_coordinates = positions[parent_index-1]
   new_coordinates = unit.Quantity(np.zeros([3]), units)
   success = False
   attempts = 0
   while not success:
    new_coordinates = attempt_move(parent_coordinates,bond_length)
    distances = non_bonded_distances(new_coordinates,positions)
    if not collisions(distances,bond_length): success = True
    if not success and attempts > 1000000:
     print("Error: maximum number of bead placement attempts exceeded")
     exit()
    attempts = attempts + 1
  positions = append_position(positions,new_coordinates)
  return(positions)

def assign_sidechain_beads(positions,model_settings,bond_length):
     sidechain_length = model_settings[3]
     for sidechain in range(0,sidechain_length):
#       print("Assigning coordinates for sidechain bead: bead "+str(len(positions)+1))
       positions = assign_position(positions,bond_length)
     return(positions)

def assign_backbone_beads(positions,monomer_start,model_settings,bond_length):
    backbone_length = model_settings[2]
    sidechain_positions = model_settings[4]
    units = bond_length.unit
    for backbone_bead_index in range(0,backbone_length):
#     print("Assigning coordinates for backbone bead: bead "+str(len(positions)+1))
     if backbone_bead_index == 0:
      if not first_atom(positions):
       positions = assign_position(positions,bond_length,parent_index=monomer_start)
     else:
      positions = assign_position(positions,bond_length)
     # Assign side-chain beads if appropriate
     if backbone_bead_index in sidechain_positions:
       positions = assign_sidechain_beads(positions,model_settings,bond_length)
    return(positions)

def assign_random_initial_coordinates(input_array):
# Define array for initial Cartesian coordinates
 model_settings,particle_properties = input_array[0],input_array[1]
 box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings[:]
 mass,q,sigma,epsilon,bond_length = particle_properties[:] 
 positions = unit.Quantity(np.zeros([3]), unit.angstrom)
 for monomer in range(0,polymer_length):
   if monomer == 0:
    monomer_start = 0
   if monomer != 0:
    monomer_start = len(positions) - sidechain_length - 1
# Assign backbone bead positions
   positions = assign_backbone_beads(positions,monomer_start,model_settings,bond_length)
 return(positions)

def write_positions_to_xyzfile(coordinates,filename,model_settings):
 box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings[:]
 monomer_size = backbone_length + sidechain_length
 xyz_object = open(filename,"w")
 xyz_object.write(str(polymer_length * monomer_size )+"\n")
 xyz_object.write("\n")
 polymer_index = 1
 bead_index = 1
 while polymer_index <= polymer_length:
  monomer_index = 1
  while monomer_index <= monomer_size:
   xyz_object.write(str("C "+str("{:10.5f}".format(coordinates[bead_index-1][0]))+" "+str("{:10.5f}".format(coordinates[bead_index-1][1]))+" "+str("{:10.5f}".format(coordinates[bead_index-1][2]))+"\n")
)
   bead_index = bead_index + 1
   monomer_index = monomer_index + 1
#   if bead_index <= 9:
  polymer_index = polymer_index + 1
 xyz_object.close()
 return

def write_positions_to_pdbfile(coordinates,filename,model_settings):
 box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings[:]
 monomer_size = backbone_length + sidechain_length
 pdb_object = open(filename,"w")
 bead_index = 1
 coordinates = coordinates._value
 for monomer_index in range(0,polymer_length):
  for backbone_bead in range(0,backbone_length):
   pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+"  X   CG  A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(round(coordinates[bead_index-1][0],3)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][1],3)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][2],3)))+"  1.00  0.00\n"))
   bead_index = bead_index + 1
  for sidechain_bead in range(0,sidechain_length):
   pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+"  Q   CG  A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(round(coordinates[bead_index-1][0],3)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][1],3)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][2],3)))+"  1.00  0.00\n"))
   bead_index = bead_index + 1
 pdb_object.write(str("END"))
 pdb_object.close()
 return

def calculate_distance_matrix(positions):
    distance_matrix = np.array([[0.0 for index in range(0,len(positions))] for index in range(0,len(positions))])
    for index_1 in range(0,len(positions)):
     for index_2 in range(0,len(positions)):
      distance_matrix[index_1][index_2] = get_distance(positions[index_1],positions[index_2])
#      if index_1 != index_2:
#       if round(distance_matrix[index_1][index_2],2) == round(sigma,2):
#        print("Atoms "+str(index_1)+" and "+str(index_2)+" are "+str(round(distance_matrix[index_1][index_2],2))+" angstroms apart")
#       if round(distance_matrix[index_1][index_2],2) < round(sigma,2):
#        print("Atoms "+str(index_1)+" and "+str(index_2)+" are "+str(round(distance_matrix[index_1][index_2],2))+" angstroms apart")
    return(distance_matrix)   
