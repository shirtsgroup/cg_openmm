#!/bin/bash

#  NAB helicoidal parameters: rise=3.38, twist=36.0
#

# ................................................................
#          Phi     Theta  Residual    Radius    Pitch1    Sweep1
# ................................................................
#
#     179.8333  155.1667    0.1201    8.5983   33.8726  323.1431
##################################################################



#                                                             atom   &   radius  &   pitch   &   sweep   &  residual  & rise  & res/turn  
#grep -A3 Phi kHelix.*.out |tail -n 1|awk -v atom=$i '{printf "%i\t %s\t %.1f\t %s\t %.1f\t %s\t %.1f\t %s\t %.3f\t %s\t %.2f\t %s\t %.2f\t   %s%s \n",\

# atom   &   radius  &   pitch   &   sweep   &  residual  &   rise                 & res/turn
# atom, "&", $4,    "&", $5,    "&", $6/31, "&", $3,     "&", $5/($6/31), "&", 360/($6/31), "\\", "\\"  }'


for i in $(ls *pdb); do
grep -A3 Phi kHelix.$i.out |tail -n 1| \
awk -v atom=$i '{printf "%s\t %s\t %.3f\t %s\t %.3f\t %s\t %.3f\t %s\t %.3f\t %s\t %.3f\t %s\t %.3f\t   %s%s \n",\
 atom, "&", $4,    "&", $5,    "&", $6/31, "&", $3,     "&", $5/(360/($6/31)), "&", 360/($6/31), "\\", "\\"  }'
done
