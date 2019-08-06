#!/bin/bash

cat > input << EOF
inputhelix alpha_-57_-47.pdb
helixout_name KHelix.32.out
coord_type 1
num_grid 360
natoms 32
nframes 1
grid_phi_beg 75
grid_phi_end 95
grid_theta_beg 0
grid_theta_end 20
helix_atom_names CA
print_to_plot 1
EOF
/Users/k/Desktop/Manuscripts/kHelix/Code/kHelix.o input
