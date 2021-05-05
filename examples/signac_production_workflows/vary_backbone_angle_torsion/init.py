# init.py

import signac
from simtk import unit

project = signac.init_project('signac_vary_theta_alpha')

# Possible parameters we can vary:

#---------------#
# LJ parameters #
#---------------#
# epsilon_bb - LJ12 epsilon parameter (unit.kilojoules_per_mole)
# epsilon_sc - LJ12 epsilon parameter (unit.kilojoules_per_mole)
# sigma_bb - LJ12 sigma parameter (unit.angstrom)
# sigma_sc - LJ12 sigma parameter (unit.angstrom)

epsilon_bb = 1.5
epsilon_sc = 5.0

sigma_bb = 2.25
sigma_sc = 3.5

#--------------------------#
# Harmonic bond parameters #
#--------------------------#
# k_bond - bond force constant (unit.kilojoule_per_mole / unit.nanometer / unit.nanometer)
# equil_bond_length - equilibrium bond length (unit.nanometer)

k_bond = 15000
equil_bond_length = 0.244

#---------------------------#
# Harmonic angle parameters #
#---------------------------#
# k_angle - bond angle force constant (unit.kilojoule_per_mole / unit.radian / unit.radian)
# equil_bond_angle - equilibrium bond angle (unit.degrees)

k_angle = 150
equil_bond_angle_bb_bb_bb = [100.0, 102.5, 105.0, 107.5, 110.0, 112.5, 115.0, 117.5, 120.0]
#equil_bond_angle_bb_bb_sc = (360-equil_bond_angle_bb_bb_bb)/2

#-----------------------------#
# Periodic torsion parameters #
#-----------------------------#
# k_torsion - torsion force constant (unit.kilojoule_per_mole)
# equil_torsion_angle - equilibrium torsion angle (unit.degrees)
# torsion_periodicity - (unitless)

# non-backbone torsions will be turned off

k_torsion = 5
equil_torsion_angle_bb_bb_bb_bb = [12, 14, 16, 18, 20]
torsion_periodicity = 1

# Run multiple trials:
trial = 1

# Number of replicas
n_replica = 12

# Exchange frequency (in time steps)
exch_freq = 200

# Collision frequency (1/ps)
coll_freq = 5

# For varying alpha and theta angles:
for theta in equil_bond_angle_bb_bb_bb:
    for alpha in equil_torsion_angle_bb_bb_bb_bb:
        sp = {
            'epsilon_bb': epsilon_bb,
            'epsilon_sc': epsilon_sc,
            'sigma_bb': sigma_bb,
            'sigma_sc': sigma_sc,
            'k_bond': k_bond,
            'equil_bond_length': equil_bond_length,
            'k_angle': k_angle,
            'equil_bond_angle_bb_bb_bb': theta,
            'k_torsion': k_torsion,
            'equil_torsion_angle_bb_bb_bb_bb': alpha,
            'torsion_periodicity': torsion_periodicity,
            'trial': trial,
            'n_replica': n_replica,
            'exch_freq': exch_freq,
            'coll_freq': coll_freq,
        }
        job = project.open_job(sp)
        job.init()
            
            