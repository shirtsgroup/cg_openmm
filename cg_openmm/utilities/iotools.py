import csv

def read_mm_energies(openmm_data_file):
          """
          Read the energies from an OpenMM data file.

          :param openmm_data_file: The path to an OpenMM data file (CSV format)
          :type openmm_data_file: str

          :returns: 
              - energies ( np.array( float * simtk.unit ) ) - An array containing all data in 'openmm_data_file'

          """

          energies = {'step': [],'potential_energy': [],'kinetic_energy': [], 'total_energy': [], 'temperature': []}

          with open(openmm_data_file) as csvfile:
            readCSV = csv.DictReader(csvfile,delimiter=',')
            for row in readCSV:
              try:
                energies['step'].append(row['#"Step"'])
              except:
                continue
              try:
                energies['potential_energy'].append(row['Potential Energy (kJ/mole)'])
              except:
                continue
              try:
                energies['kinetic_energy'].append(row['Kinetic Energy (kJ/mole)'])
              except:
                continue
              try:
                energies['total_energy'].append(row['Total Energy (kJ/mole)'])
              except:
                continue
              try:
                energies['temperature'].append(row['Temperature (K)'])
              except:
                continue

#              print(energies)
#              exit()

          return(energies)

def write_pdbfile_without_topology(CGModel,filename,energy=None):
        """
        Writes the positions from an input CGModel class object to the file 'filename'.

        Parameters
        ----------

        CGModel: Coarse grained model class object

        filename: Path to the file where we will write PDB coordinates.

        energy: Energy to write to the PDB file, default = None

        """

        pdb_object = open(filename,"w")
        if energy != None:
          pdb_object.write("## The OpenMM potential energy for this structure is: "+str(energy)+"\n")
          pdb_object.write("## with the following parameter settings:\n")
          pdb_object.write("## sigma = "+str(CGModel.sigmas['bb_bb_sigma'])+", epsilon = "+str(CGModel.epsilons['bb_bb_eps'])+"\n")

        coordinates = CGModel.positions
        bead_index = 1
        for monomer_index in range(CGModel.polymer_length):
          monomer_type = CGModel.sequence[monomer_index]
          element_index = 1
          for backbone_bead in range(monomer_type['backbone_length']):

            if monomer_index in list([0,CGModel.polymer_length-1]):
             pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+" X"+str(element_index)+str("{:>6}".format(str("MT")))+" A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(round(coordinates[bead_index-1][0].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][1].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][2].in_units_of(unit.angstrom)._value,2)))+"  1.00  0.00\n"))
            else:
             pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+" X"+str(element_index)+str("{:>6}".format(str("M")))+" A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(round(coordinates[bead_index-1][0].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][1].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][2].in_units_of(unit.angstrom)._value,2)))+"  1.00  0.00\n"))
            bead_index = bead_index + 1
            element_index = element_index + 1

            if backbone_bead in [monomer_type['sidechain_positions']]:
              for sidechain_bead in range(monomer_type['sidechain_length']):
                if monomer_index in list([0,CGModel.polymer_length-1]):
                 pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+" A"+str(element_index)+str("{:>6}".format(str("MT")))+" A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(round(coordinates[bead_index-1][0].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][1].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][2].in_units_of(unit.angstrom)._value,2)))+"  1.00  0.00\n"))
                else:
                 pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+" A"+str(element_index)+str("{:>6}".format(str("M")))+" A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(round(coordinates[bead_index-1][0].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][1].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][2].in_units_of(unit.angstrom)._value,2)))+"  1.00  0.00\n"))
                bead_index = bead_index + 1
                element_index = element_index + 1
        pdb_object.write(str("TER\n"))

        write_bonds(CGModel,pdb_object)
        pdb_object.close()
        return