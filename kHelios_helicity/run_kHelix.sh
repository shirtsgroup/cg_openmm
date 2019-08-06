#!/bin/bash

for i in $(ls *pdb); do
cat > input << EOF
inputhelix $i
helixout_name kHelix.$i.out
coord_type 1
num_grid 360
natoms 8
nframes 1
grid_phi_beg 0
grid_phi_end 180
grid_theta_beg 0
grid_theta_end 180
helix_atom_names X1
print_to_plot 1
EOF
/home/gmeek/software/kHelios/bin/helios.o input
done
