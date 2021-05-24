Creating a cgmodel
==================

The CGModel object contains the topology, force field definitions, and initial particle positions of a coarse-grained model.
The following example creates a homo-oligomer model with 1 backbone bead and 1 sidechain bead per residue.
Particle positions are generated using the random builder included in cg_openmm. 

.. code-block:: python

    from simtk import unit
    from cg_openmm.cg_model.cgmodel import CGModel
    
    # Specify backbone (bb) and sidechain (sc) particle parameters:
    sigma = 0.3 * unit.nanometer
    epsilon = 2 * unit.kilojoule_per_mole
    mass = 100 * unit.amu
    
    bb = {"particle_type_name": "bb", "sigma": sigma, "epsilon": epsilon, "mass": mass}
    sc = {"particle_type_name": "sc", "sigma": sigma, "epsilon": epsilon, "mass": mass}
    
    # Define monomer (residue):
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0}
    
    # Specify bonded parameters:
    bond_lengths = {
        "default_bond_length": 0.35 * unit.nanometer,
        "bb_bb_bb_bond_length": 0.40 * unit.nanometer}
        
    bond_force_constants = {
        "default_bond_force_constant": 1000 * unit.kilojoule_per_mole / unit.nanometer**2}
    
    equil_bond_angles = {
        "default_equil_bond_angle": 120.0 * unit.degrees,
        "bb_bb_bb_equil_bond_angle": 150.0 * unit.degrees}        
    
    bond_angle_force_constants = {
        "default_bond_angle_force_constant": 100.0 * unit.kilojoule_per_mole / unit.radian**2}

    torsion_phase_angles = {
        "default_torsion_phase_angle": 150 * unit.degrees}        
    
    torsion_force_constants = {
        "default_torsion_force_constant": 2.0 * unit.kilojoule_per_mole,
        "bb_bb_bb_bb_torsion_force_constant": 5.0 * unit.kilojoule_per_mole}

    torsion_periodicities = {
        "default_torsion_periodicity": 1}

    # Define oligomer sequence:
    sequence = 12 * [A]
    
    # Initial particle positions determined from random builder
    
    cgmodel = CGModel(
        particle_type_list=[bb, sc],
        bond_lengths=bond_lengths,
        bond_force_constants=bond_force_constants,
        bond_angle_force_constants=bond_angle_force_constants,
        torsion_force_constants=torsion_force_constants,
        equil_bond_angles=equil_bond_angles,
        torsion_phase_angles=torsion_phase_angles,
        torsion_periodicities=torsion_periodicities,
        include_nonbonded_forces=True,
        include_bond_forces=True,
        include_bond_angle_forces=True,
        include_torsion_forces=True,
        constrain_bonds=False,
        sequence=sequence,
        monomer_types=[A],
    )

A CGModel may be saved in serialized form to use for further analysis:

.. code-block:: python
    
    cgmodel.export("stored_cgmodel.pkl")