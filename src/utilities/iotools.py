import csv

def read_mm_energies(openmm_data_file):
          """
          Read the energies from an OpenMM output file.

          Parameters
          ----------

          :param openmm_data_file: The path to an OpenMM data file (CSV format)
          :type openmm_data_file: string

          Returns
          -------

          energies: np.array( float * simtk.unit )
                    A numpy array containing all data in 'openmm_data_file'

          """

          with open(output_data) as csvfile:
            readCSV = csv.reader(csvfile,delimiter=',')
            next(readCSV)

          for row in readCSV:
