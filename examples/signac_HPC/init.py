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
# sigma_bb - LJ12 sigma parameter (unit.nanometer)
# sigma_sc - LJ12 sigma parameter (unit.nanometer)

#--------------------------#
# Harmonic bond parameters #
#--------------------------#
# k_bond - bond force constant (unit.kilojoule_per_mole / unit.nanometer / unit.nanometer)
# bond_length - equilibrium bond length (unit.nanometer)

#---------------------------#
# Harmonic angle parameters #
#---------------------------#
# k_angle - bond angle force constant (unit.kilojoule_per_mole / unit.radian / unit.radian)
# equil_bond_angle - equilibrium bond angle (unit.degrees)

#-----------------------------#
# Periodic torsion parameters #
#-----------------------------#
# k_torsion - torsion force constant (unit.kilojoule_per_mole)
# equil_torsion_angle - equilibrium torsion angle (unit.degrees)
# torsion_periodicity - (unitless)

equil_bond_angle = [120,140,160]
equil_torsion_angle = [110, 130, 150]

for theta in equil_bond_angle:
    for alpha in equil_torsion_angle:
        sp = {
            'alpha': alpha,
            'theta': theta,
        }
        job = project.open_job(sp)
        job.init()
            