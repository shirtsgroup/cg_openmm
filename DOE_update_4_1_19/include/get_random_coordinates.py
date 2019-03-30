## No default python environment

# This script generates random coordinates for a 1-1 CG polymer

# =============================================================================================
# 1) PYTHON PACKAGE IMPORTS
# =============================================================================================

# System packages
import os, sys, timeit, socket
from io import StringIO
import numpy as np
import math, random
from simtk import unit
# =============================================================================================
# 2) ENVIRONMENT/JOB SETTINGS
# =============================================================================================

sigma = 1.0

def get_distance(bead_1,bead_2):
 sqrt_arg = 0.0
 for direction in range(0,len(bead_1)):
  sqrt_arg = sqrt_arg + ( bead_2[direction] - bead_1[direction] ) ** 2.0
 distance = math.sqrt(sqrt_arg)
 return(distance)

def random_sign(number):
# 0 = negative, 1 = positive
 random_int = random.randint(0,1)
 if random_int == 0: number = -1.0 * number
 return(number)

def attempt_move(parent_coordinates):
#    print(parent_coordinates)
    new_coordinates = np.zeros([3])
    move = np.zeros([3])
    move_direction_list = []
    distance = 0.0
    for direction in range(0,3):
     move_direction = random.randint(0,2)
     while move_direction in move_direction_list:
      move_direction = random.randint(0,2)
     if direction != 2:
      move[move_direction] = random_sign(random.uniform(0.0,math.sqrt(sigma**2.0-distance**2.0)))
     if direction == 2:
      move[move_direction] = random_sign(math.sqrt(sigma**2.0-distance**2.0))
     
     distance = get_distance(parent_coordinates,np.array([parent_coordinates[index] + move[index] for index in range(0,3)]))
     
     move_direction_list.append(move_direction)
     new_coordinates[move_direction] = float("{:10.5f}".format(parent_coordinates[move_direction] + move[move_direction])) 
#    print(get_distance(parent_coordinates,new_coordinates))
    return(new_coordinates)

def non_bonded_distances(new_coordinates,existing_coordinates):
   distances = [float(round(get_distance(new_coordinates,existing_coordinates[monomer_index]),2)) for monomer_index in range(0,len(existing_coordinates))]
#   print(distances)
   if sigma in distances:
    distances.remove(sigma)
   return(distances)

def collisions(distances):
   collision = False
   if len(distances) > 0:
    for distance in distances:
     if distance < float(round(sigma,2)):
      collision = True
#   if collision: print(distances)
   return(collision)

def assign_position(positions,parent_index=-1):
  if len(positions) == 0:
   new_coordinates = np.zeros([3])
  else:
   if parent_index == -1:
    parent_index = len(positions)
   parent_coordinates = positions[parent_index-1]
   new_coordinates = np.zeros([3])
   success = False
   attempts = 0
   while not success: 
    new_coordinates = attempt_move(parent_coordinates)
    distances = non_bonded_distances(new_coordinates,positions)
    if not collisions(distances): success = True
    if not success and attempts > 1000000:
     print("Error: maximum number of bead placement attempts exceeded")
     exit()
    attempts = attempts + 1
   for index in range(0,3):
    new_coordinates[index] = float("{:7.2f}".format(new_coordinates[index]))
#    print(parent_atom_coordinates)
#    print(get_distance(new_coordinates,parent_atom_coordinates))
#  print(new_coordinates)
  positions.append(new_coordinates)
  return(positions)

def assign_sidechain_beads(positions,model_settings):
     sidechain_length = model_settings[3]
     for sidechain in range(0,sidechain_length):
#       print("Assigning coordinates for sidechain bead: bead "+str(len(positions)+1))
       positions = assign_position(positions)
     return(positions)

def assign_backbone_beads(positions,monomer_start,model_settings):
    backbone_length = model_settings[2]
    sidechain_positions = model_settings[4]
    for backbone_bead_index in range(0,backbone_length):
#     print("Assigning coordinates for backbone bead: bead "+str(len(positions)+1))
     if len(positions) == 0:
      positions = [np.zeros([3])]
     else:
      if backbone_bead_index == 0:
       positions = assign_position(positions,parent_index=monomer_start)
      else:
       positions = assign_position(positions)
     # Assign side-chain beads if appropriate
     if backbone_bead_index in sidechain_positions:
       positions = assign_sidechain_beads(positions,model_settings)
    return(positions)

def assign_random_initial_coordinates(model_settings):
# Define array for initial Cartesian coordinates
 box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings[:]
 positions = []
 for monomer in range(0,polymer_length):
   monomer_start = len(positions) - sidechain_length
#   if monomer > 0: print("monomer start is: "+str(monomer_start))
# Assign backbone bead positions
   positions = assign_backbone_beads(positions,monomer_start,model_settings)
# Assign all side-chain bead positions
 return(positions)

def add_position_units(positions):
 positions = unit.Quantity(positions, unit.angstroms)
 return(positions)

def remove_position_units(positions):
 new_positions = np.zeros([len(positions),3])
 for position in range(0,len(positions)):
  if str("nm") in str(positions[position]):
   coordinates = str(positions[position]).replace('(','').replace(')','').replace(' nm','').replace(' ','').split(',')
  if str("A") in str(positions[position]):
   coordinates = str(positions[position]).replace('[','').replace(']','').replace(' A','').replace('   ',' ').replace('  ',' ').strip().split(' ')
  for direction in range(0,3):
   new_positions[position][direction] = float(coordinates[direction])
 return(new_positions)

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

def write_positions_to_pdbfile(coordinates,filename,model_settings):
 box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions = model_settings[:]
 monomer_size = backbone_length + sidechain_length
 pdb_object = open(filename,"w")
 bead_index = 1
 for monomer_index in range(0,polymer_length):
  for backbone_bead in range(0,backbone_length):
   if monomer_index == 0 or monomer_index == int(polymer_length-1):
    pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+"  X   CG  A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(coordinates[bead_index-1][0]))+" "+str("{:>7}".format(coordinates[bead_index-1][1]))+" "+str("{:>7}".format(coordinates[bead_index-1][2]))+"  1.00  0.00\n"))
   else:
    pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+"  X   CG  A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(coordinates[bead_index-1][0]))+" "+str("{:>7}".format(coordinates[bead_index-1][1]))+" "+str("{:>7}".format(coordinates[bead_index-1][2]))+"  1.00  0.00\n"))
   bead_index = bead_index + 1
  for sidechain_bead in range(0,sidechain_length):
   if monomer_index == 0 or monomer_index == int(polymer_length-1):
    pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+"  Q   CG  A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(coordinates[bead_index-1][0]))+" "+str("{:>7}".format(coordinates[bead_index-1][1]))+" "+str("{:>7}".format(coordinates[bead_index-1][2]))+"  1.00  0.00\n"))
   else:
    pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+"  Q   CG  A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(coordinates[bead_index-1][0]))+" "+str("{:>7}".format(coordinates[bead_index-1][1]))+" "+str("{:>7}".format(coordinates[bead_index-1][2]))+"  1.00  0.00\n"))
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
