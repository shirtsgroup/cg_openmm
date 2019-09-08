import numpy as np
import os, statistics, random
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
#import mdtraj, msmbuilder
from foldamers.src.cg_model.cgmodel import basic_cgmodel
from cg_openmm.src.build.cg_build import build_topology, build_system
from cg_openmm.src.simulation.tools import get_mm_energy, build_mm_simulation
from foldamers.src.utilities.iotools import write_pdbfile_without_topology
from foldamers.src.utilities.util import random_positions
from foldamers.src.parameters.secondary_structure import fraction_native_contacts

def get_ensemble(cgmodel,ensemble_size=100,high_energy=False,low_energy=False):
        """
        Given a coarse grained model, this function generates an ensemble of high energy configurations and, by default, saves this ensemble to the foldamers/ensembles database for future reference/use, if a high-energy ensemble with these settings does not already exist.

        :param cgmodel: CGModel() class object.
        :type cgmodel: class

        :param ensemble_size: Number of structures to generate for this ensemble, default = 100
        :type ensemble_size: integer

        :param high_energy: If set to 'True', this function will generate an ensemble of high-energy structures, default = False
        :type high_energy: Logical

        :param low_energy: If set to 'True', this function will generate an ensemble of low-energy structures, default = False
        :type low_energy: Logical

        :returns:
           - ensemble (List(positions(np.array(float*simtk.unit (shape = num_beads x 3))))) - A list of the positions for all members in the ensemble.

        """
        if high_energy and low_energy:
          print("ERROR: Both 'high_energy' and 'low_energy' ensembles were requested in 'get_ensemble()'.  Please set only one of these variables to 'True', and call the function again.")
          exit()
        if low_energy:
          print("Generating an ensemble of "+str(ensemble_size)+" low energy configurations.")
        if high_energy:
          print("Generating an ensemble of "+str(ensemble_size)+" high energy configurations.")
        if not high_energy and not low_energy:
          print("Generating an ensemble of "+str(ensemble_size)+" configurations.")

        ensemble = []
        for member in range(ensemble_size):

          if high_energy:
            positions = random_positions(cgmodel,high_energy=True)
            
          if low_energy:
            positions = random_positions(cgmodel,low_energy=True)

          if not high_energy and not low_energy:
            positions = random_positions(cgmodel)

          ensemble.append(positions)
        
        return(ensemble)

def get_pdb_list(ensemble_directory):
        """
        Given an 'ensemble_directory', this function retrieves a list of the PDB files within it.

        :param ensemble_directory: Path to a folder containing PDB files
        :type ensemble_directory: str

        :returns:
         - pdb_list ( List(str) ) - A list of the PDB files in the provided 'ensemble_directory'.

        """
        pdb_list = []
        for file in os.listdir(ensemble_directory):
           if file.endswith('.pdb'):
              pdb_list.append(str(str(ensemble_directory)+"/"+str(file)))
        return(pdb_list)

def write_ensemble_pdb(cgmodel,ensemble_directory=None):
        """
        Given a CGModel() class object that contains positions, this function writes a PDB file for the coarse grained model, using those positions.

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :param ensemble_directory: Path to a folder containing PDB files, default = None
        :type ensemble_directory: str

        ..warning:: If no 'ensemble_directory' is provided, the  
        
        """
        if ensemble_directory == None:
          ensemble_directory = get_ensemble_directory(cgmodel)
        index = 1
        pdb_list = get_pdb_list(ensemble_directory)
        while pdb_file_name in pdb_list:
           pdb_file_name = str(ensemble_directory+"/cg"+str(index)+".pdb")
           index = index + 1
        write_pdbfile_without_topology(cgmodel,pdb_file_name,energy=energy)
 
        return

def get_ensemble_directory(cgmodel,ensemble_type=None):
        """
        Given a CGModel() class object, this function uses its attributes to assign an ensemble directory name.

        For example, the directory name for a model with 20 monomers, all of which contain one backbone bead and one sidechain bead, and whose bond lengths are all 7.5 Angstroms, would be: "foldamers/ensembles/20_1_1_0_7.5_7.5_7.5".

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :param ensemble_type: Designates the type of ensemble for which we will assign a directory name.  default = None.  Valid options include: "native" and "nonnative"
        :type ensemble_type: str

        :returns:
          - ensemble_directory ( str ) - The path/name for the ensemble directory.

        """
        monomer_type = cgmodel.monomer_types[0]
        ensembles_directory = str(str(__file__.split('src/ensembles/ens_build.py')[0])+"ensembles")
        if not os.path.exists(ensembles_directory):
            os.mkdir(ensembles_directory)
        model_directory = str(str(ensembles_directory)+"/"+str(cgmodel.polymer_length)+"_"+str(monomer_type['backbone_length'])+"_"+str(monomer_type['sidechain_length'])+"_"+str(monomer_type['sidechain_positions']))
        if not os.path.exists(model_directory):
            os.mkdir(model_directory)

        # We determine a suitable name for the ensemble directory by combining the 'bb_bb_bond_length', 'bb_sc_bond_length', and 'sc_sc_bond_length' into a single string:
        ens_str = [monomer_type['bond_lengths']['bb_bb_bond_length']._value,monomer_type['bond_lengths']['bb_sc_bond_length']._value,monomer_type['bond_lengths']['sc_sc_bond_length']._value]
        if ensemble_type == None:
          ensemble_directory = str(str(model_directory)+"/bonds_"+str(ens_str[0])+"_"+str(ens_str[1])+"_"+str(ens_str[2]))
        if ensemble_type == "nonnative":
          ensemble_directory = str(str(model_directory)+"/bonds_"+str(ens_str[0])+"_"+str(ens_str[1])+"_"+str(ens_str[2])+"_nonnative")
        if ensemble_type == "native":
          ensemble_directory = str(str(model_directory)+"/bonds_"+str(ens_str[0])+"_"+str(ens_str[1])+"_"+str(ens_str[2])+"_native")
        
        return(ensemble_directory)

def get_ensemble_data(cgmodel,ensemble_directory):
        """
        Given a CGModel() class object and an 'ensemble_directory', this function reads the PDB files within that directory, as well as any energy data those files contain.

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :param ensemble_directory: The path/name of the directory where PDB files for this ensemble are stored
        :type ensemble_directory: str

        :returns:
           - ensemble (List(positions(np.array(float*simtk.unit (shape = num_beads x 3))))) - A list of the positions for all members in the ensemble.

           - ensemble_energies ( List(`Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )) - A list of the energies that were stored in the PDB files for the ensemble, if any.

        ..warning:: When energies are written to a PDB file, only the sigma and epsilon values for the model are written to the file with the positions.  Unless the user is confident about the model parameters that were used to generate the energies in the PDB files, it is probably best to re-calculate their energies.  This can be done with the 'cg_openmm' package.  More specifically, one can compute an updated energy for individual ensemble members, with the current coarse grained model parameters, with 'get_mm_energy', a function in 'cg_openmm/cg_openmm/simulation/tools.py'.

        """
        ensemble_energies = []
        ensemble = []
        pdb_list = get_pdb_list(ensemble_directory)
        random.shuffle(pdb_list)
        if len(pdb_list) > 0:
           print("Searching for suitable ensemble members in the 'foldamers' database.")
           for pdb_file in pdb_list:
              pdb_mm_obj = PDBFile(pdb_file)
              cgmodel.positions = pdb_mm_obj.getPositions()
              ensemble.append(cgmodel.positions)
              cgmodel.simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions)
              energy = cgmodel.simulation.context.getState(getEnergy=True).getPotentialEnergy()
              ensemble_energies.append(energy)

        return(ensemble,ensemble_energies)

def test_energy(energy):
        """
        Given an energy, this function determines if that energy is too large to be "physical".  This function is used to determine if the user-defined input parameters for a coarse grained model give a reasonable potential function.

        :param energy: The energy to test.
        :type energy: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ or float        

        :returns:
          - pass_energy_test ( Logical ) - A variable indicating if the energy passed ("True") or failed ("False") a "sanity" test for the model's energy.

        """
        try:
          pass_energy_test = energy.__lt__(9.9e5*unit.kilojoule_per_mole)
        except:
          if energy < 9.9e5:
            pass_energy_test = True
          else:
            pass_energy_test = False

        return(pass_energy_test)

def improve_ensemble(energy,positions,ensemble,ensemble_energies,unchanged_iterations):
        """

        Given an energy and positions for a single pose, as well as the same data for a reference ensemble, this function "improves" the quality of the ensemble by identifying poses with the lowest potential energy.

        :param energy: The energy for a pose.
        :type energy: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ 

        :param positions: Positions for coarse grained particles in the model, default = None
        :type positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit )

        :param ensemble: A group of similar poses.
        :type ensemble: List(positions(np.array(float*simtk.unit (shape = num_beads x 3))))

        :param ensemble_energies: A list of energies for a conformational ensemble.
        :type ensemble_energies: List(`Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )

        :param unchanged_iterations: The number of iterations for which the ensemble has gone unchanged.
        :type unchanged_iterations: int

        :returns:
           - ensemble (List(positions(np.array(float*simtk.unit (shape = num_beads x 3))))) - A list of the positions for all members in the ensemble.

           - ensemble_energies ( List(`Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ )) - A list of the energies that were stored in the PDB files for the ensemble, if any.

           - unchanged_iterations ( int ) - The number of iterations for which the ensemble has gone unchanged.

        """
        if any([energy < ensemble_energies[i] for i in range(len(ensemble_energies))]):
          ensemble_energies[ensemble_energies.index(max(ensemble_energies))] = energy
          ensemble[ensemble_energies.index(max(ensemble_energies))] = positions
          unchanged_iterations = 0
        else:
          unchanged_iterations = unchanged_iterations + 1
        return(ensemble,ensemble_energies,unchanged_iterations)

def get_nonnative_ensemble(cgmodel,native_structure,ensemble_size=100,native_fraction_cutoff=0.75,rmsd_cutoff=10.0,ensemble_build_method="native_contacts"):
        """
        """
        library_ensemble = []
        print("Building/retrieving nonnative ensemble.")
        ensemble_directory = get_ensemble_directory(cgmodel,ensemble_type="nonnative")
        if os.path.exists(ensemble_directory):
          library_ensemble,library_ensemble_energies = get_ensemble_data(cgmodel,ensemble_directory)
        else:
          os.mkdir(ensemble_directory)

        ensemble = []
        ensemble_energies = []
        unchanged_iterations = 0
        if len(library_ensemble) > 0:
          for index in range(len(library_ensemble)):

            positions = library_ensemble[index]
            energy = library_ensemble_energies[index]

            if ensemble_build_method == "native_contacts":

              if fraction_native_contacts(cgmodel,positions,native_structure) < native_fraction_cutoff:
               pass_energy_test = test_energy(energy)
               if pass_energy_test:
                  ensemble_energies.append(energy)
                  ensemble.append(positions)            
                  if len(ensemble_energies) >= ensemble_size:
                    ensemble,ensemble_energies,unchanged_iterations = improve_ensemble(energy,positions,ensemble,ensemble_energies,unchanged_iterations)
                    if unchanged_iterations >= 100:
                      return(ensemble,ensemble_energies)

        unchanged_iterations = 0
        while len(ensemble_energies) < ensemble_size or unchanged_iterations < 100:
              print("There are "+str(len(ensemble_energies))+" poses in the ensemble.")
              positions = random_positions(cgmodel)

              if ensemble_build_method == "mbar":
                replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
                configurations,energies,temperatures = get_decorrelated_samples(replica_positions,replica_energies,temperature_list)
                for configuration in range(len(configurations)):
                  if test_energy(energies[configuration]):
                    ensemble_energies.append(energy)
                    ensemble.append(positions)

              if ensemble_build_method == "native_contacts":

                if fraction_native_contacts(cgmodel,positions,native_structure) < native_fraction_cutoff:
                 simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions)
                 energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                 pass_energy_test = test_energy(energy)
                 if pass_energy_test:
                    ensemble_energies.append(energy)
                    ensemble.append(positions)

                 if len(ensemble_energies) >= ensemble_size:
                    print("Unchanged iterations = "+str(unchanged_iterations))
                    ensemble,ensemble_energies,unchanged_iterations = improve_ensemble(energy,positions,ensemble,ensemble_energies,unchanged_iterations)
                    if unchanged_iterations >= 100:
                      return(ensemble,ensemble_energies)

        return(ensemble,ensemble_energies)

def get_native_ensemble(cgmodel,native_structure,ensemble_size=10,native_fraction_cutoff=0.9,rmsd_cutoff=10.0,ensemble_build_method="native_contacts"):
        """
        """
        print("Building/retrieving native ensemble.")
        ensemble_directory = get_ensemble_directory(cgmodel,ensemble_type="native")
        if os.path.exists(ensemble_directory):
          library_ensemble,library_ensemble_energies = get_ensemble_data(cgmodel,ensemble_directory)
        else:
          os.mkdir(ensemble_directory)

        ensemble = []
        ensemble_energies = []
        unchanged_iterations = 0
        for index in range(len(library_ensemble)):

            positions = library_ensemble[index]
            energy = library_ensemble_energies[index]

            if ensemble_build_method == "native_contacts":

              if fraction_native_contacts(cgmodel,positions,native_structure) > native_fraction_cutoff:
               try:
                 pass_energy_test = energy.__lt__(9.9e5*unit.kilojoule_per_mole)
               except:
                 if energy < 9.9e5:
                   pass_energy_test = True
                 else:
                   pass_energy_test = False
               if pass_energy_test:
                  ensemble_energies.append(energy)
                  ensemble.append(positions)

               if len(ensemble_energies) == ensemble_size:
                    if unchanged_iterations < 100:
                      if any([energy < ensemble_energies[i] for i in range(len(ensemble_energies))]):
                        ensemble_energies[ensemble_energies.index(max(ensemble_energies))] = energy
                        ensemble[ensemble_energies.index(max(ensemble_energies))] = positions
                        unchanged_iterations = 0
                      else:
                        unchanged_iterations = unchanged_iterations + 1
                    if unchanged_iterations >= 100:
                      return(ensemble,ensemble_energies)

        unchanged_iterations = 0
        #print("Adding new files to database.")
        while len(ensemble_energies) < ensemble_size and unchanged_iterations < 100:

              positions = random_positions(cgmodel)

              if ensemble_build_method == "native_contacts":

                if fraction_native_contacts(cgmodel,positions,native_structure) > native_fraction_cutoff:
                 simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions)
                 energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                 try:
                   pass_energy_test = energy.__lt__(9.9e5*unit.kilojoule_per_mole)
                 except:
                   if energy < 9.9e5:
                     pass_energy_test = True
                   else:
                     pass_energy_test = False
                 if pass_energy_test:
                    ensemble_energies.append(energy)
                    ensemble.append(positions)

                 if len(ensemble_energies) == ensemble_size:
                    if unchanged_iterations < 100:
                      if any([energy < ensemble_energies[i] for i in range(len(ensemble_energies))]):
                        ensemble_energies[ensemble_energies.index(max(ensemble_energies))] = energy
                        ensemble[ensemble_energies.index(max(ensemble_energies))] = positions
                        unchanged_iterations = 0
                      else:
                        unchanged_iterations = unchanged_iterations + 1
                    if unchanged_iterations >= 100:
                      return(ensemble,ensemble_energies)

        return(ensemble,ensemble_energies)

def get_ensembles(cgmodel,native_structure,ensemble_size=None):
        """
        """
        if ensemble_size == None:
          nonnative_ensemble,nonnative_ensemble_energies = get_nonnative_ensemble(cgmodel,native_structure)
          native_ensemble,native_ensemble_energies = get_native_ensemble(cgmodel,native_structure)
        else:
          nonnative_ensemble,nonnative_ensemble_energies = get_nonnative_ensemble(cgmodel,native_structure,ensemble_size=ensemble_size)
          native_ensemble,native_ensemble_energies = get_native_ensemble(cgmodel,native_structure,ensemble_size=round(ensemble_size/10))
        return(nonnative_ensemble,nonnative_ensemble_energies,native_ensemble,native_ensemble_energies)

def z_score(topology,system,nonnative_ensemble_energies,native_ensemble_energies):
        """
        Given an ensemble of nonnative structures, and a low-energy ("native") structure, this subroutine will calculate the Z-score.

        Parameters
        ----------

        nonnative_ensemble: List( positions( np.array( float * simtk.unit ( shape = num_beads x 3 ) ) )
                  A list of the positions for all members in the high_energy ensemble.

        native_structure: positions( np.array( float * simtk.unit ( shape = num_beads x 3 ) )
                          The positions for a low energy structure.

        """

        nonnative_ensemble_energies = np.array([energy._value for energy in nonnative_ensemble_energies])
        native_ensemble_energies = np.array([energy._value for energy in native_ensemble_energies])

        average_nonnative_energy = statistics.mean(nonnative_ensemble_energies)

        stdev_nonnative_energy = statistics.stdev(nonnative_ensemble_energies)

        native_energy = statistics.mean(native_ensemble_energies)

        print(native_ensemble_energies)
        print(nonnative_ensemble_energies)
        print(average_nonnative_energy)
        print(stdev_nonnative_energy)
        z_score = ( average_nonnative_energy - native_energy ) / stdev_nonnative_energy
        print(z_score)

        return(z_score)
