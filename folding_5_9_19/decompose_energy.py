def build_cg_system(cgmodel):

        sigma = cgmodel.sigma.in_units_of(unit.nanometer)._value
        charge = cgmodel.charge._value
        epsilon = cgmodel.epsilon.in_units_of(unit.kilojoule_per_mole)._value
        bond_length = cgmodel.bond_length.in_units_of(unit.nanometer)._value

        # Create system
        system = mm.System()
        nonbonded_force = mm.NonbondedForce()
        bead_index = 0

        # Create nonbonded forces
        for monomer in range(cgmodel.polymer_length):
          for backbone_bead in range(cgmodel.backbone_length):
            system.addParticle(cgmodel.mass)
            nonbonded_force.addParticle(charge,sigma,epsilon)
            if backbone_bead in cgmodel.sidechain_positions:
              for sidechain_bead in range(cgmodel.sidechain_length):
                system.addParticle(cgmodel.mass)
                nonbonded_force.addParticle(charge,sigma,epsilon)
        system.addForce(nonbonded_force)
        return(system)


if decompose_energy:
 full_interaction_list = []
 for i in range(cgmodel.num_beads):
  for j in range(i+1,cgmodel.num_beads):
    if [i,j] not in cgmodel.bond_list:
      full_interaction_list.append([i,j])

 if cgmodel.system.getNumForces() != 0:
  nonbonded_force = cgmodel.system.getForce(cgmodel.system.getNumForces()-1)
 nonbonded_force.setNonbondedMethod(0)
 print(topology.bonds)
 print("There are "+str(nonbonded_force.getNumParticles())+" atoms.")
 print("There are "+str(len(full_interaction_list))+" total nonbonded interactions.")
 print("There are "+str(nonbonded_force.getNumExceptions())+" nonbonded exceptions.")
 print("There should be "+str(len(full_interaction_list)-nonbonded_force.getNumExceptions())+" nonbonded interactions after applying exceptions.")
 print("The OpenMM potential energy for this configuration is: "+str(simulation.context.getState(getEnergy=True).getPotentialEnergy()))
 print("The forces for this configuration are: "+str(simulation.context.getState(getForces=True).getForces()))
 nonbonded_energy = 0.0 * unit.kilojoule_per_mole
 for interaction in full_interaction_list:
  if interaction not in cgmodel.bond_list:
   energy = calculate_nonbonded_energy(cgmodel,particle1=interaction[0],particle2=interaction[1])
   nonbonded_energy = nonbonded_energy.__add__(energy)
#print(cgmodel.positions)
 print("The nonbonded energies are: "+str([calculate_nonbonded_energy(cgmodel,particle1=interaction[0],particle2=interaction[1]).in_units_of(unit.kilojoule_per_mole) for interaction in cgmodel.get_nonbonded_interaction_list()]))
 print("The manually-calculated potential energy for this configuration is: "+str(nonbonded_energy))
 exit()

