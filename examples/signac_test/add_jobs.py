# add_jobs.py

import signac
from simtk import unit

# Add signac statepoints to an existing project

project = signac.get_project()

# Possible parameters we can vary:

#---------------#
# LJ parameters #
#---------------#
# epsilon_bb - LJ12 epsilon parameter (unit.kilojoules_per_mole)
# epsilon_sc - LJ12 epsilon parameter (unit.kilojoules_per_mole)
# sigma_bb - LJ12 sigma parameter (unit.nanometer)
# sigma_sc - LJ12 sigma parameter (unit.nanometer)

epsilon_bb = 1.375
epsilon_sc = 1.375
sigma_bb = 0.3
sigma_sc = 0.3

#--------------------------#
# Harmonic bond parameters #
#--------------------------#
# k_bond - bond force constant (unit.kilojoule_per_mole / unit.nanometer / unit.nanometer)
# equil_bond_length - equilibrium bond length (unit.nanometer)

k_bond = 1000
equil_bond_length = 0.2 

#---------------------------#
# Harmonic angle parameters #
#---------------------------#
# k_angle - bond angle force constant (unit.kilojoule_per_mole / unit.radian / unit.radian)
# equil_bond_angle - equilibrium bond angle (unit.degrees)

k_angle = 25
# New points are 130, 150
equil_bond_angle = [120, 130, 140, 150, 160] 

#-----------------------------#
# Periodic torsion parameters #
#-----------------------------#
# k_torsion - torsion force constant (unit.kilojoule_per_mole)
# equil_torsion_angle - equilibrium torsion angle (unit.degrees)
# torsion_periodicity - (unitless)

k_torsion = 3 
# New points are 120, 140
equil_torsion_angle = [110, 120, 130, 140, 150]
torsion_periodicity = 1

# Create list of the existing statepoints:
existing_sp = []

for job in project:
    existing_sp.append(job.sp)

# For varying alpha and theta angles:
for theta in equil_bond_angle:
    for alpha in equil_torsion_angle:
        sp = {
            'epsilon_bb': epsilon_bb,
            'epsilon_sc': epsilon_sc,
            'sigma_bb': sigma_bb,
            'sigma_sc': sigma_sc,
            'k_bond': k_bond,
            'equil_bond_length': equil_bond_length,
            'k_angle': k_angle,
            'equil_bond_angle': theta,
            'k_torsion': k_torsion,
            'equil_torsion_angle': alpha,
            'torsion_periodicity': torsion_periodicity,
        }
        
        if sp not in existing_sp:
            job = project.open_job(sp)
            job.init()
            existing_sp.append(job.sp)
            