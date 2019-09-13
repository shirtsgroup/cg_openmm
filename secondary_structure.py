import os, subprocess
import numpy as np
from simtk import unit
from statistics import mean
from scipy.stats import linregress
from scipy import spatial
#from spatial.transform import Rotation as R
from foldamers.utilities.util import *
from foldamers.utilities.iotools import write_pdbfile_without_topology

def fraction_native_contacts(cgmodel,positions,native_structure,cutoff_distance=None):
        """
        Given a coarse grained model, positions for that model, and positions for the native structure, this function calculates the fraction of native contacts for the model.

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :param positions: Positions for the particles in a coarse grained model.
        :type positions: np.array( float * unit.angstrom ( num_particles x 3 ) )

        :param native_structure: Positions for the native structure.
        :type native_structure: np.array( float * unit.angstrom ( num_particles x 3 ) )

        :param cutoff_distance: The maximum distance for two nonbonded particles that are defined as "native",default=None
        :type cutoff_distance: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :returns:
          - Q ( float ) - The fraction of native contacts for the model with the current structure.

        """
        if cutoff_distance == None:
          cutoff_distance = 1.1 * cgmodel.get_sigma(0)

        nonbonded_interaction_list = cgmodel.nonbonded_interaction_list
        print("There are "+str(len(nonbonded_interaction_list))+" total nonbonded interaction possibilities.")
        native_structure_distances = distances(nonbonded_interaction_list,native_structure)
        current_structure_distances = distances(nonbonded_interaction_list,positions)
        native_distances = []
        native_interaction_list = []
        
        for interaction in range(len(nonbonded_interaction_list)):
          if native_structure_distances[interaction].__lt__(cutoff_distance):
            native_distances.append(native_structure_distances[interaction])
            native_interaction_list.append(interaction)
        total_native_interactions = len(native_interaction_list)
        print("There are "+str(total_native_interactions)+" native interactions.")

        current_distances = []
        current_structure_interaction_list = []
        for interaction in range(len(nonbonded_interaction_list)):
          if interaction in native_interaction_list:
            if current_structure_distances[interaction].__lt__(1.1*native_structure_distances[interaction]):
              current_distances.append(current_structure_distances[interaction])
              current_structure_interaction_list.append(interaction)
        current_structure_native_interactions = len(current_structure_interaction_list)
        print("The current structure has "+str(current_structure_native_interactions)+" native interactions")
        Q = current_structure_native_interactions / total_native_interactions
        return(Q)

def get_helical_parameters(cgmodel):
        """
        Given a coarse grained model as input, this function uses the `kHelios software package <https://pubs.acs.org/doi/10.1021/acs.jcim.6b00721>`_ to analyze the helical properties of the model.

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :returns:
          - pitch ( float ) - The distance between monomers in adjacent turns of a helix
          - radius ( float ) - The radius of the helix
          - monomers_per_turn ( float ) - The number of monomrs per turn of the helix
          - residual ( float ) - The average distance of all backbone particles from a circle projected onto the x-y plane.  Used to determine the accuracy of the helical axis, as fit to the input data.  Units are in Angstroms.

        .. warning:: This function requires a pre-installed version of `kHelios <https://pubs.acs.org/doi/10.1021/acs.jcim.6b00721>`_ .  Because kHelios is formatted to accept input job scripts, this function writes and executes a job script for kHelios.  In order to function properly, the user must redefine the 'helios_path' variable for their system.

        """
        helios_path = str('/home/gmeek/Foldamers/foldamers/foldamers/parameters/helios.o')
        cgmodel = orient_along_z_axis(cgmodel)
        write_pdbfile_without_topology(cgmodel,"temp_pitch.pdb")
        kHelix_run_file = "run_kHelix.sh"
        file = open(kHelix_run_file,"w")
        file.write('#!/bin/bash\n')
        file.write('\n')
        file.write('cat > input << EOF\n')
        file.write('inputhelix $1\n')
        file.write('helixout_name kHelix.out\n')
        file.write('coord_type 1\n')
        file.write('num_grid 20\n')
        file.write('natoms '+str(round(cgmodel.num_beads/2))+'\n')
        file.write('nframes 1\n')
        file.write('grid_phi_beg 0\n')
        file.write('grid_phi_end 20\n')
        file.write('grid_theta_beg 0\n')
        file.write('grid_theta_end 20\n')
        file.write('helix_atom_names X1\n')
        file.write('print_to_plot 1\n')
        file.write('EOF\n')
        file.write(str(helios_path)+' input\n')
        #file.write('done\n')        
        file.close()
        subprocess.run(['chmod','+x','run_kHelix.sh'])
        subprocess.run([str(str(os.getcwd())+"/"+str(kHelix_run_file)),"temp_pitch.pdb",">","helios_output"])
        #os.remove("helios_output")
        file = open("kHelix.out",mode="r")
        output = file.readlines()
        line_index = 1
        for line in output:
          if line_index == 43:
            residual = line.split()[2]
            radius = line.split()[3]
            pitch = line.split()[4]
            sweep = float(line.split()[5])
            monomers_per_turn = cgmodel.polymer_length/(sweep/360.0)
            break
          line_index = line_index + 1
          
        return(pitch,radius,monomers_per_turn,residual)

def orient_along_z_axis(cgmodel,plot_projections=False):
        """
        Given a coarse grained model as input, this function orients the model along the z-axis.

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :param plot_projections: Variable indicating whether or not to plot intermediate projections/operations during identification of a helical axis.

        :returns:
           - cgmodel ( class ) - CGModel() class object, with positions oriented so that the helical axis is along the z-axis

        """
        plot_projections = False
        positions = np.array([[float(i.in_units_of(unit.angstrom)._value) for i in position] for position in cgmodel.positions])
        # 1) Get the backbone particle positions
        backbone_positions = []
        for particle in range(len(cgmodel.positions)):
          if cgmodel.get_particle_type(particle) == "backbone":
            backbone_positions.append(cgmodel.positions[particle])
        backbone_positions = np.array([[float(i.in_units_of(unit.angstrom)._value) for i in coord] for coord in backbone_positions])
        # 2) Project the backbone positions onto the (x,y) plane
        xy_projected_positions = backbone_positions
        x_data = []
        y_data = []
        for position in xy_projected_positions:
          position[2] = 0.0
          x_data.append(position[0])
          y_data.append(position[1])
        # 3) Calculate the best fit line for these projected positions
        slope,intercept,r,p,std_err=linregress(np.array([x for x in x_data]),np.array([y for y in y_data]))

        if plot_projections:
        # Plot this projected data, as well as the best fit line   
          file_name = "xy_projection.png"
          figure = pyplot.figure(1)
          x_data = np.array([x for x in x_data])
          y_data = np.array([y for y in y_data])
          pyplot.xlabel("x")
          pyplot.ylabel("y")
          pyplot.scatter(x_data,y_data)
          x = np.linspace(min(x_data),max(x_data),100)
          pyplot.plot(x,slope*x+intercept,label=str("y="+str(round(slope,2))+"x+"+str(round(intercept,2))))
          pyplot.legend()
          pyplot.savefig(file_name)
          pyplot.show()
          pyplot.close()

        # 4) Rotate the coordinate system so that this line is oriented along the x-axis
        # Calculate angle from z-axis:
        z_axis_angle = np.arctan(slope)
        z_axis_rotation_matrix = spatial.transform.Rotation.from_euler('xyz',[0.,0.,z_axis_angle], degrees=False)
        x_oriented_positions = z_axis_rotation_matrix.apply(positions)

        # 5) Project the positions onto the (x,z) plane
        xz_projected_positions = backbone_positions
        x_data = []
        z_data = []
        for position_index in range(len(xz_projected_positions)):
          xz_projected_positions[position_index][1] = 0.0
          x_data.append(positions[position_index][0])
          z_data.append(positions[position_index][2])

        # 6) Calculate the best fit line for these projected positions
        slope,intercept,r,p,std_err=linregress(np.array([x for x in x_data]),np.array([z for z in z_data]))

        if plot_projections:
        # Plot this projected data, as well as the best fit line   
          file_name = "xz_projection.png"
          figure = pyplot.figure(1)
          x_data = np.array([x for x in x_data])
          z_data = np.array([z for z in z_data])
          pyplot.xlabel("x")
          pyplot.ylabel("z")
          pyplot.scatter(x_data,z_data)
          x = np.linspace(min(x_data),max(x_data),100)
          pyplot.plot(x,slope*x+intercept,label=str("z="+str(round(slope,2))+"x+"+str(round(intercept,2))))
          pyplot.legend()
          pyplot.savefig(file_name)
          pyplot.show()
          pyplot.close()


        # 7) Rotate the coordinate system so that this line is oriented along the x-axis
        # Calculate angle from y-axis:
        y_axis_angle = np.arctan(slope)
        y_axis_rotation_matrix = spatial.transform.Rotation.from_euler('xyz', [0.,y_axis_angle,0.], degrees=False)

        new_positions = y_axis_rotation_matrix.apply(x_oriented_positions)

        # 8) For comparison with kHelios output, rotate the molecule again so that the helical (x) axis is oriented along the z axis instead.

        z_axis_angle = 3.1415/2.0
        z_axis_rotation_matrix = spatial.transform.Rotation.from_euler('xyz',[0.,z_axis_angle,0.0], degrees=False)
        final_positions = z_axis_rotation_matrix.apply(new_positions)

        cgmodel.positions = unit.Quantity(final_positions,unit.angstrom)

        file = open("after_rotation.pdb","w")
        PDBFile.writeFile(cgmodel.topology,cgmodel.positions,file=file)

        return(cgmodel)

def get_helical_data(cgmodel):
        """
        """
        cgmodel = orient_along_z_axis(cgmodel) 

        # Get the new backbone particle positions
        backbone_positions = []
        for particle in range(len(cgmodel.positions)):
          if cgmodel.get_particle_type(particle) == "backbone":
            backbone_positions.append(cgmodel.positions[particle])
        backbone_positions = np.array([[float(i.in_units_of(unit.angstrom)._value) for i in coord] for coord in backbone_positions])

        # radius
        axis_distances = []
        rotations = 0.0
        for position in backbone_positions:
          axis_distance = distance(unit.Quantity([float(position[0]),0.0,0.0],unit.angstrom),unit.Quantity(position,unit.angstrom))
          axis_distances.append(axis_distance)
          if len(axis_distances) > 1:
            rotation = np.arctan(position[1]/position[2]) - last_angle
            last_angle = rotation
            rotations = rotations + rotation
          else:
            rotation = np.arctan(position[1]/position[2])
            last_angle = rotation
            rotations = rotations + rotation
            
        radius = mean(np.array([float(dist.in_units_of(unit.angstrom)._value) for dist in axis_distances]))
        particles_per_turn = float(cgmodel.polymer_length/(rotations/6.28))

        # pitch
        #
        # Shift all coordinates so that the first backbone atom has z=0

        shift = - cgmodel.positions[0][2]._value

        axis_deltas = []
        for position in cgmodel.positions:
          position[0]._value = position[0]._value + shift
          if abs(position[0]._value - cgmodel.positions[0][0]._value) > 0:
            axis_deltas.append(float(position[0]._value-cgmodel.positions[0][0]._value))
        average_delta = mean(axis_deltas)
        pitch = average_delta * particles_per_turn

        return(radius,pitch,particles_per_turn)
        

