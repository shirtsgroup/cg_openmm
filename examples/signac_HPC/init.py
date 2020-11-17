# init.py

import signac
from simtk import unit

project = signac.init_project('signac_test')

# Possible parameters we can vary:

#---------------#
# LJ parameters #
#---------------#
# epsilon_bb - LJ12 epsilon parameter (unit.kilojoules_per_mole)
# epsilon_sc - LJ12 epsilon parameter (unit.kilojoules_per_mole)
# sigma_bb - LJ12 sigma parameter (unit.angstrom)
# sigma_sc - LJ12 sigma parameter (unit.angstrom)

epsilon = 2.25

sigma = 2.25

#--------------------------#
# Harmonic bond parameters #
#--------------------------#
# k_bond - bond force constant (unit.kilojoule_per_mole / unit.nanometer / unit.nanometer)
# equil_bond_length - equilibrium bond length (unit.nanometer)

k_bond = 10000
equil_bond_length = 0.2 

#---------------------------#
# Harmonic angle parameters #
#---------------------------#
# k_angle - bond angle force constant (unit.kilojoule_per_mole / unit.radian / unit.radian)
# equil_bond_angle - equilibrium bond angle (unit.degrees)

k_angle = 100
equil_bond_angle_bb_bb_bb = [100, 110, 120, 130, 140, 150, 160, 170]
equil_bond_angle_bb_bb_sc = 120

#-----------------------------#
# Periodic torsion parameters #
#-----------------------------#
# k_torsion - torsion force constant (unit.kilojoule_per_mole)
# equil_torsion_angle - equilibrium torsion angle (unit.degrees)
# torsion_periodicity - (unitless)

# non-backbone torsions will be turned off

k_torsion = 5
equil_torsion_angle_bb_bb_bb_bb = [10, 20, 30, 40, 50, 60, 70, 80, 90]
torsion_periodicity = 1

# For varying alpha and theta angles:
for theta in equil_bond_angle_bb_bb_bb:
    for alpha in equil_torsion_angle_bb_bb_bb_bb:
        sp = {
            'epsilon': epsilon,
            'sigma': sigma,
            'k_bond': k_bond,
            'equil_bond_length': equil_bond_length,
            'k_angle': k_angle,
            'equil_bond_angle_bb_bb_bb': theta,
            'equil_bond_angle_bb_bb_sc': equil_bond_angle_bb_bb_sc,
            'k_torsion': k_torsion,
            'equil_torsion_angle_bb_bb_bb_bb': alpha,
            'torsion_periodicity': torsion_periodicity,
        }
        job = project.open_job(sp)
        job.init()
            
            