#!/bin/bash



for i in $(ls *pdb); do
for j in $(echo 3 4 5 6 7 8 9 16 32); do
grep -A3 Phi kHelix.$i.$j.out |tail -n 1| \
awk -v atom=$j -v name=$i '{printf "%s\t %s\t %i\t %s\t %.3f\t %s\t %.3f\t %s\t %.3f\t %s\t %.3f\t %s\t %.3f\t %s\t %.3f\t   %s%s \n",\
 name, "&", atom, "&", $4, "&", $5, "&", $6/(atom-1), "&", $3, "&", $5/(360/($6/(atom-1))), "&", 360/($6/(atom-1)), "\\", "\\"  }'
done
done
