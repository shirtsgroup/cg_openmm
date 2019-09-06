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
