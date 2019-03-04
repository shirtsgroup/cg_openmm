#!/bin/bash

#

gromacs_rtp="/home/gmeek/software/anaconda3/pkgs/gromacs-4.6.5-0/share/gromacs/top/gromos54a7.ff/aminoacids.rtp"

gromacs_atp="/home/gmeek/software/anaconda3/pkgs/gromacs-4.6.5-0/share/gromacs/top/gromos54a7.ff/atomtypes.atp"


# echo "" >> $gromacs_rtp
# echo "[ CG ]" >> $gromacs_rtp
# echo " [ atoms ]" >> $gromacs_rtp
# echo "   CG1  CH2     0.00000     0" >> $gromacs_rtp
# echo " [ bonds ]" >> $gromacs_rtp
# echo " [ angles ]" >> $gromacs_rtp
# echo ";  ai    aj    ak   gromos type" >> $gromacs_rtp
# echo " [ impropers ]" >> $gromacs_rtp
# echo ";  ai    aj    ak    al   gromos type" >> $gromacs_rtp
# echo " [ dihedrals ]" >> $gromacs_rtp
# echo ";  ai    aj    ak    al   gromos type" >> $gromacs_rtp

# echo "" >> $gromacs_atp
# echo "  CG1  12.001 ;     Coarse-grained particle type 1" >> $gromacs_atp
# echo "  CG2  12.001 ;     Coarse-grained particle type 2" >> $gromacs_atp
#fi

#exit
# Run the simulation
#echo "Running simulation..."
#yank script --yaml=CG_octamer.yaml

# Analyze the data
echo "Analyzing data..."
yank analyze --store=CG_octamer-output
