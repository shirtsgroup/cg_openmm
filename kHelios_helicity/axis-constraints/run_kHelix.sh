#!/bin/bash

for i in $(ls *pdb); do
#for j in $(echo 3 4 5 6 7 8 9 16 32); do
j=32
cat > input << EOF
inputhelix $i
helixout_name kHelix.$i.$j.out
coord_type 1
num_grid 360
natoms $j
nframes 1
grid_phi_beg 0
grid_phi_end 180
grid_theta_beg 0
grid_theta_end 180
helix_atom_names CA
print_to_plot 1
EOF
/Users/k/Desktop/Manuscripts/kHelix/Code/kHelix.o input
#done
done
