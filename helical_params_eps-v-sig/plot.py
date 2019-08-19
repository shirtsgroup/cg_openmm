import os
from simtk import unit
import numpy as np
import matplotlib.pyplot as pyplot

grid_size = 4
bond_length = 7.5 * unit.angstrom

sigma_list = [ (1.5 + i*0.2) * bond_length for i in range(grid_size)]
epsilon_list = [ unit.Quantity((0.5 + i*0.5),unit.kilocalorie_per_mole) for i in range(grid_size)]
pitch_list = []
radius_list = []
monomers_per_turn_list = []

x = np.unique([sigma._value for sigma in sigma_list])
y = np.unique([epsilon._value for epsilon in epsilon_list])
X,Y = np.meshgrid(x,y)

file = open("helical_data.dat","r")
data = file.readlines()
file.close()
line_index = 0
for line in data:
  if line_index != 0:
    pitch_list.append(line.split()[2])
    radius_list.append(line.split()[3])
    monomers_per_turn_list.append(line.split()[4])
  line_index = line_index + 1

pitch_list = np.array([float(pitch) for pitch in pitch_list])
radius_list = np.array([float(radius) for radius in radius_list])
monomers_per_turn_list = np.array([float(mpt) for mpt in monomers_per_turn_list])

Z = pitch_list.reshape(len(y),len(x))

pyplot.figure(1)
pyplot.xlabel("$\sigma$ ( nm )")
pyplot.ylabel("$\epsilon$ ( kcal/mol )")
pyplot.title("Helical Pitch (Angstroms)")
pyplot.pcolormesh(X,Y,Z)
pyplot.colorbar()
pyplot.savefig("pitch.png")
pyplot.show()
pyplot.close()

Z = radius_list.reshape(len(y),len(x))

pyplot.figure(2)
pyplot.xlabel("$\sigma$ ( nm )")
pyplot.ylabel("$\epsilon$ ( kcal/mol )")
pyplot.title("Helical Radius (Angstroms)")
pyplot.pcolormesh(X,Y,Z)
pyplot.colorbar()
pyplot.savefig("radius.png")
pyplot.show()
pyplot.close()

Z = monomers_per_turn_list.reshape(len(y),len(x))

pyplot.figure(3)
pyplot.xlabel("$\sigma$ ( nm )")
pyplot.ylabel("$\epsilon$ ( kcal/mol )")
pyplot.title("Monomers-per-turn")
pyplot.pcolormesh(X,Y,Z)
pyplot.colorbar()
pyplot.savefig("monomers-per-turn.png")
pyplot.show()
pyplot.close()

exit()
