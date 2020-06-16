import os, subprocess
import numpy as np
import simtk.unit as unit
from statistics import mean
from scipy.stats import linregress
from scipy import spatial
import matplotlib.pyplot as plt
from foldamers.utilities.util import *
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from openmmtools.multistate import MultiStateReporter, ReplicaExchangeAnalyzer
import pymbar

kB = unit.MOLAR_GAS_CONSTANT_R # Boltzmann constant

def get_native_contacts(cgmodel, native_structure, native_contact_distance_cutoff):
    """
        Given a coarse grained model, positions for that model, and positions for the native structure, this function calculates the fraction of native contacts for the model.

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :param native_structure: Positions for the particles in a coarse grained model.
        :type native_structure: np.array( float * unit.angstrom ( num_particles x 3 ) )

        :param native_contact_distance_cutoff: The maximum distance for two nonbonded particles that are defined as "native",default=None
        :type native_contact_distance_cutoff: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :returns:
          - native_contact_list - A list of the nonbonded interactions whose inter-particle distances are less than the 'native_contact_cutoff_distance'.

        """

    nonbonded_interaction_list = cgmodel.nonbonded_interaction_list
    native_structure_distances = distances(nonbonded_interaction_list, native_structure)
    native_contact_list = []

    for interaction in range(len(nonbonded_interaction_list)):
        if native_structure_distances[interaction].__lt__(native_contact_distance_cutoff):
            native_contact_list.append(nonbonded_interaction_list[interaction])

    return native_contact_list


def get_number_native_contacts(cgmodel, native_structure, native_contact_distance_cutoff):
    """
        Given a coarse grained model, positions for that model, and positions for the native structure, this function calculates the fraction of native contacts for the model.

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :param native_structure: Positions for the particles in a coarse grained model.
        :type native_structure: np.array( float * unit.angstrom ( num_particles x 3 ) )

        :param native_contact_cutoff_distance: The maximum distance for two nonbonded particles that are defined as "native",default=None
        :type native_contact_cutoff_distance: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :returns:
          - contacts - The number of nonbonded interactions that are considered 'native'.

        """

    native_contact_list = get_native_contacts(
        cgmodel, native_structure, native_contact_distance_cutoff
    )

    contacts = len(native_contact_list)

    return contacts

def expectations_fraction_contacts(list_native_contacts, native_pdb, distance_cutoff, temperature_list, output_directory="output", output_data="output.nc", num_intermediate_states=0):

    max_native_contacts = len(list_native_contacts)

    # extract reduced energies and the state indices from the .nc  
    reporter = MultiStateReporter(os.path.join(output_directory,output_data), open_mode="r")
    analyzer = ReplicaExchangeAnalyzer(reporter)
    (
        replica_energies,
        unsampled_state_energies,
        neighborhoods,
        replica_state_indices,
    ) = analyzer.read_energies()

    # determine the numerical values of beta at each state in units consisten with the temperature
    Tunit = temperature_list[0].unit
    temps = np.array([temp.value_in_unit(Tunit)  for temp in temperature_list])  # should this just be array to begin with
    beta_k = 1 / (kB.value_in_unit(unit.kilojoule_per_mole/Tunit) * temps)

    # convert the energies from replica/evaluated state/sample form to evaluated state/sample form
    replica_energies = pymbar.utils.kln_to_kn(replica_energies)
    n_samples = len(replica_energies[0,:])
    
    # calculate the number of states we need expectations at.  We want it at all of the original
    # temperatures, each intermediate temperature, and then temperatures +/- from the original
    # to take finite derivatives.

    # create  an array for the temperature and energy for each state, including the
    # finite different state.
    n_sampled_T = len(temps)
    n_unsampled_states = (n_sampled_T + (n_sampled_T-1)*num_intermediate_states)
    unsampled_state_energies = np.zeros([n_unsampled_states,n_samples])
    full_T_list = np.zeros(n_unsampled_states)

    # delta is the spacing between temperatures.
    delta = np.zeros(n_sampled_T-1)

    # fill in a list of temperatures at all original temperatures and all intermediate states.
    full_T_list[0] = temps[0]  
    t = 0
    for i in range(n_sampled_T-1):
        delta[i] = (temps[i+1] - temps[i])/(num_intermediate_states+1)
        for j in range(num_intermediate_states+1):
            full_T_list[t] = temps[i] + delta[i]*j
            t += 1
    full_T_list[t] = temps[-1]
    n_T_vals = t+1

    # calculate betas of all of these temperatures
    beta_full_k = 1 / (kB.value_in_unit(unit.kilojoule_per_mole/Tunit) * full_T_list)
    
    ti = 0
    N_k = np.zeros(n_unsampled_states)
    for k in range(n_unsampled_states):
        # Calculate the reduced energies at all temperatures, sampled and unsample.
        unsampled_state_energies[k, :] = replica_energies[0,:]*(beta_full_k[k]/beta_k[0])
        if ti < len(temps):
            # store in N_k which states do and don't have samples.
            if full_T_list[k] == temps[ti]:
                ti += 1
                N_k[k] = n_samples//len(temps)  # these are the states that have samples

    # call MBAR to find weights at all states, sampled and unsampled
    mbarT = pymbar.MBAR(unsampled_state_energies,N_k,verbose=False, relative_tolerance=1e-12);
    
    # Now we have the weights at all temperatures, so we can
    # calculate the expectations.  We now need a number of native contacts for each
    # structure.
    n_samples_per_replica = n_samples//n_sampled_T
    Q = np.zeros(n_samples)
    dunit = distance_cutoff.unit

    # calculate Q (fraction of native contacts) for each structure.
    # because we need to iterate per state, we have to do some bookkeeping.
    # The sampled state energies are stored in order of the replicas, so the
    # samples need to correspond to the same order.
    iperstate = np.zeros(n_sampled_T,int)
    for step in range(n_samples_per_replica):
        for replica_index in range(n_sampled_T):
            sampler_states = reporter.read_sampler_states(iteration = step)
            positions = sampler_states[replica_index].positions
            structure_dists = distances(list_native_contacts, positions)
            dists = np.array([dist.value_in_unit(dunit) for dist in structure_dists])
            Q[iperstate[replica_index]+replica_index*n_samples_per_replica] = np.sum(dists < distance_cutoff._value)/max_native_contacts
            iperstate[replica_index]+=1

    # calculte the expectation of Q at each unsampled states         
    results = mbarT.computeExpectations(Q)  # compute expectations of Q at all points
    Q_expect = results[0]
    dQ_expect = results[1]

    # return the results in a dictionary (better than in a list)
    return_results = dict()
    return_results["T"] = full_T_list
    return_results["Q"] = Q_expect
    return_results["dQ"] = dQ_expect

    return return_results

    
def fraction_native_contacts(
    cgmodel,
    positions,
    native_structure,
    native_structure_contact_distance_cutoff=None,
    native_contact_cutoff_ratio=None,
):
    """
        Given a coarse grained model, positions for that model, and positions for the native structure, this function calculates the fraction of native contacts for the model.

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :param positions: Positions for the particles in a coarse grained model.
        :type positions: np.array( float * unit.angstrom ( num_particles x 3 ) )

        :param native_structure: Positions for the native structure.
        :type native_structure: np.array( float * unit.angstrom ( num_particles x 3 ) )

        :param native_structure_contact_distance_cutoff: The distance below which two nonbonded, interacting particles that are defined as "native contact",default=None
        :type native_structure_contact_distance_cutoff: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :param native_contact_cutoff_ratio: The distance below which two nonbonded, interacting particles in a non-native pose are assigned as a "native contact", as a ratio of the distance for that contact in the native structure, default=None
        :type native_contact_cutoff_ratio: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :returns:
          - Q ( float ) - The fraction of native contacts for the comparison pose.

        """

    if native_structure_contact_distance_cutoff == None:
        native_structure_contact_distance_cutoff = 1.1 * cgmodel.get_sigma(0)

    if native_contact_cutoff_ratio == None:
        native_contact_cutoff_ratio = 1.05

    native_contact_list = get_native_contacts(
        cgmodel, native_structure, native_structure_contact_distance_cutoff
    )

    total_native_interactions = get_number_native_contacts(
        cgmodel, native_structure, native_structure_contact_distance_cutoff
    )

    if total_native_interactions == 0:
        print("ERROR: there are 0 'native' interactions with the current cutoff distance.")
        print("Try increasing the 'native_structure_contact_distance_cutoff'")
        exit()

    native_contact_distances = distances(native_contact_list, native_structure)

    current_structure_distances = distances(native_contact_list, positions)

    current_structure_native_contact_list = []
    for interaction in range(len(native_contact_list)):
        if current_structure_distances[interaction].__lt__(
            native_contact_cutoff_ratio * native_contact_distances[interaction]
        ):
            current_structure_native_contact_list.append(native_contact_list[interaction])
    current_structure_number_native_interactions = len(current_structure_native_contact_list)
    Q = current_structure_number_native_interactions / total_native_interactions
    return Q


def optimize_Q(cgmodel, native_structure, ensemble):
    """
        Given a coarse grained model and a native structure as input

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :param native_structure: Positions for the native structure.
        :type native_structure: np.array( float * unit.angstrom ( num_particles x 3 ) )

        :param ensemble: A list of poses that will be used to optimize the cutoff distance for defining native contacts
        :type ensemble: List(positions(np.array(float*simtk.unit (shape = num_beads x 3))))

        :returns:
          - native_structure_contact_distance_cutoff ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The ideal distance below which two nonbonded, interacting particles should be defined as a "native contact"
        """

    cutoff_list = [(0.95 + i * 0.01) * cgmodel.get_sigma(0) for i in range(30)]

    cutoff_Q_list = []
    for cutoff in cutoff_list:
        Q_list = []
        for pose in ensemble:
            Q = fraction_native_contacts(
                cgmodel, pose, native_structure, native_structure_contact_distance_cutoff=cutoff
            )
            Q_list.append(Q)

        mean_Q = mean(Q_list)
        cutoff_Q_list.append(mean_Q)

    cutoff_Q_list.index(max(cutoff_Q_list))

    native_structure_contact_distance_cutoff = cutoff_Q_list.index(max(cutoff_Q_list))

    return native_structure_contact_distance_cutoff


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
    helios_path = str("../../foldamers/foldamers/parameters/helios.o")
    cgmodel = orient_along_z_axis(cgmodel)
    write_pdbfile_without_topology(cgmodel, "temp_pitch.pdb")
    kHelix_run_file = "run_kHelix.sh"
    file = open(kHelix_run_file, "w")
    file.write("#!/bin/bash\n")
    file.write("\n")
    file.write("cat > input << EOF\n")
    file.write("inputhelix $1\n")
    file.write("helixout_name kHelix.out\n")
    file.write("coord_type 1\n")
    file.write("num_grid 20\n")
    file.write("natoms " + str(round(cgmodel.num_beads / 2)) + "\n")
    file.write("nframes 1\n")
    file.write("grid_phi_beg 0\n")
    file.write("grid_phi_end 20\n")
    file.write("grid_theta_beg 0\n")
    file.write("grid_theta_end 20\n")
    file.write("helix_atom_names X1\n")
    file.write("print_to_plot 1\n")
    file.write("EOF\n")
    file.write(str(helios_path) + " input\n")
    # file.write('done\n')
    file.close()
    subprocess.run(["chmod", "+x", "run_kHelix.sh"])
    subprocess.run(
        [
            str(str(os.getcwd()) + "/" + str(kHelix_run_file)),
            "temp_pitch.pdb",
            ">",
            "helios_output",
        ]
    )
    # os.remove("helios_output")
    file = open("kHelix.out", mode="r")
    output = file.readlines()
    line_index = 1
    for line in output:
        if line_index == 43:
            residual = line.split()[2]
            radius = line.split()[3]
            pitch = line.split()[4]
            sweep = float(line.split()[5])
            monomers_per_turn = cgmodel.polymer_length / (sweep / 360.0)
            break
        line_index = line_index + 1

    return (pitch, radius, monomers_per_turn, residual)


def orient_along_z_axis(cgmodel, plot_projections=False):
    """
        Given a coarse grained model as input, this function orients the model along the z-axis.

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :param plot_projections: Variable indicating whether or not to plot intermediate projections/operations during identification of a helical axis.

        :returns:
           - cgmodel ( class ) - CGModel() class object, with positions oriented so that the helical axis is along the z-axis

        """
    positions = np.array(
        [
            [float(i.in_units_of(unit.angstrom)._value) for i in position]
            for position in cgmodel.positions
        ]
    )
    # 1) Get the backbone particle positions
    backbone_positions = []
    for particle in range(len(cgmodel.positions)):
        if cgmodel.get_particle_type(particle) == "backbone":
            backbone_positions.append(cgmodel.positions[particle])
    backbone_positions = np.array(
        [
            [float(i.in_units_of(unit.angstrom)._value) for i in coord]
            for coord in backbone_positions
        ]
    )
    # 2) Project the backbone positions onto the (x,y) plane
    xy_projected_positions = backbone_positions
    x_data = []
    y_data = []
    for position in xy_projected_positions:
        position[2] = 0.0
        x_data.append(position[0])
        y_data.append(position[1])
    # 3) Calculate the best fit line for these projected positions
    slope, intercept, r, p, std_err = linregress(
        np.array([x for x in x_data]), np.array([y for y in y_data])
    )

    if plot_projections:
        # Plot this projected data, as well as the best fit line
        file_name = "xy_projection.png"
        figure = pyplot.figure(1)
        x_data = np.array([x for x in x_data])
        y_data = np.array([y for y in y_data])
        pyplot.xlabel("x")
        pyplot.ylabel("y")
        pyplot.scatter(x_data, y_data)
        x = np.linspace(min(x_data), max(x_data), 100)
        pyplot.plot(
            x,
            slope * x + intercept,
            label=str("y=" + str(round(slope, 2)) + "x+" + str(round(intercept, 2))),
        )
        pyplot.legend()
        pyplot.savefig(file_name)
        pyplot.show()
        pyplot.close()

    # 4) Rotate the coordinate system so that this line is oriented along the x-axis
    # Calculate angle from z-axis:
    z_axis_angle = np.arctan(slope)
    z_axis_rotation_matrix = spatial.transform.Rotation.from_euler(
        "xyz", [0.0, 0.0, z_axis_angle], degrees=False
    )
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
    slope, intercept, r, p, std_err = linregress(
        np.array([x for x in x_data]), np.array([z for z in z_data])
    )

    if plot_projections:
        # Plot this projected data, as well as the best fit line
        file_name = "xz_projection.png"
        figure = pyplot.figure(1)
        x_data = np.array([x for x in x_data])
        z_data = np.array([z for z in z_data])
        pyplot.xlabel("x")
        pyplot.ylabel("z")
        pyplot.scatter(x_data, z_data)
        x = np.linspace(min(x_data), max(x_data), 100)
        pyplot.plot(
            x,
            slope * x + intercept,
            label=str("z=" + str(round(slope, 2)) + "x+" + str(round(intercept, 2))),
        )
        pyplot.legend()
        pyplot.savefig(file_name)
        pyplot.show()
        pyplot.close()

    # 7) Rotate the coordinate system so that this line is oriented along the x-axis
    # Calculate angle from y-axis:
    y_axis_angle = np.arctan(slope)
    y_axis_rotation_matrix = spatial.transform.Rotation.from_euler(
        "xyz", [0.0, y_axis_angle, 0.0], degrees=False
    )

    new_positions = y_axis_rotation_matrix.apply(x_oriented_positions)

    # 8) For comparison with kHelios output, rotate the molecule again so that the helical (x) axis is oriented along the z axis instead.

    z_axis_angle = 3.1415 / 2.0
    z_axis_rotation_matrix = spatial.transform.Rotation.from_euler(
        "xyz", [0.0, z_axis_angle, 0.0], degrees=False
    )
    final_positions = z_axis_rotation_matrix.apply(new_positions)

    cgmodel.positions = unit.Quantity(final_positions, unit.angstrom)

    file = open("after_rotation.pdb", "w")
    PDBFile.writeFile(cgmodel.topology, cgmodel.positions, file=file)

    return cgmodel


def show_helical_fit(cgmodel):
    """
        Given a coarse grained model containing positions, this function performs a helical fit for the backbone particles with `kHelios <https://pubs.acs.org/doi/10.1021/acs.jcim.6b00721>`_ , and uses 'matplotlib' to display attributes of the helical fit.
        """
    # 1) Get the backbone particle positions
    positions = np.array(
        [
            [float(i.in_units_of(unit.angstrom)._value) for i in position]
            for position in cgmodel.positions
        ]
    )
    backbone_positions = []
    for particle in range(len(cgmodel.positions)):
        if cgmodel.get_particle_type(particle) == "backbone":
            backbone_positions.append(cgmodel.positions[particle])
    backbone_positions = np.array(
        [
            [float(i.in_units_of(unit.angstrom)._value) for i in coord]
            for coord in backbone_positions
        ]
    )
    c = backbone_positions

    curves = [c]
    labels = ["helix (unrotated)"]
    for i in range(len(curves)):
        fig = plt.figure(i)
        curve = curves[i]
        label = labels[i]
        ax = fig.gca(projection="3d")
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], label=label)
        ax.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.zlabel('z') # not defined?
        plt.show()

    return


def calculate_p2(cgmodel):
    """
        Given a coarse grained model containing positions, this function returns the `'P2' <http://cmt.dur.ac.uk/sjc/thesis_dlc/node19.html>`_ orientational ordering parameter value for the current pose.

        .. warning:: By default, 'P2' is evaluated using the positions for only the backbone particles.

        :param cgmodel: CGModel() class object
        :type cgmodel: class

        :returns:
         - p2 ( float ) - The value for the 'P2' orientational ordering parameter.
        """
    positions = np.array(
        [
            [float(i.in_units_of(unit.angstrom)._value) for i in position]
            for position in cgmodel.positions
        ]
    )
    # 1) Get the backbone particle positions
    backbone_positions = []
    for particle in range(len(cgmodel.positions)):
        if cgmodel.get_particle_type(particle) == "backbone":
            backbone_positions.append(cgmodel.positions[particle])
    backbone_positions = np.array(
        [
            [float(i.in_units_of(unit.angstrom)._value) for i in coord]
            for coord in backbone_positions
        ]
    )

    c = backbone_positions

    u = np.diff(c, axis=0)
    for ui in u:
        ui /= np.sqrt(np.dot(ui, ui))

    Q = np.zeros([3, 3])
    for ui in u:
        Q += 1.5 * np.outer(ui, ui)
    Q /= len(u)
    Q -= 0.5 * np.eye(3)
    vals, vecs = np.linalg.eig(Q)
    p2 = np.mean(np.dot(u, vecs), axis=0)

    dirindices = np.argsort(np.abs(p2))

    h = vecs[:, dirindices[2]]
    l = vecs[:, dirindices[1]]
    m = vecs[:, dirindices[0]]

    # rotate the helix itself into the new coordinates
    # in many cases, this seems to not be a a great rotation.  It seems to
    # start tilting the helix a bit in many cases. Not sure why!!!!

    S = np.zeros([3, 3])
    S = vecs
    cp = np.dot(c, S)
    cp1 = cp[:, dirindices]
    cp = cp1

    up = np.dot(u, S)
    up1 = up[:, dirindices]
    up = up1

    up2 = np.diff(cp, axis=0)
    for upi in up2:
        upi /= np.sqrt(np.dot(upi, upi))

    FQ = np.zeros([3, 3])
    for upi in up:
        FQ += 1.5 * np.outer(upi, upi)
    FQ /= len(up)
    FQ -= 0.5 * np.eye(3)

    avecos = np.mean(up[:, 2])
    avesin = np.sqrt(1 - avecos ** 2)

    zaxis = np.array([0, 0, 1])
    upr = np.zeros(np.shape(u))
    for i in range(np.shape(upr)[0]):
        scal = np.sqrt(1 - up[i, 2] ** 2)
        # project out into x,y plane
        ax1 = np.array([up[i, 0] / scal, up[i, 1] / scal, 0])
        # normal from the plane
        nm = np.cross(zaxis, ax1)  # the normal to the plane
        v = up[i]  # the vector to rotate
        # R(theta)v = nm(nm.v) + cos(theta) (nm x v) x nm + sin(-theta)(nm x v)  # from wikipedia
        upr[i] = (
            nm * np.dot(nm, v) + avecos * np.cross(np.cross(nm, v), nm) - avesin * np.cross(nm, v)
        )

    # cmid = 0.5*(cp[0:-1,:] + cp[1:,:])
    # z = cmid[0:,2]/length

    curves = [c, cp, u, up, upr]
    labels = [
        "helix (unrotated)",
        "helix (rotated)",
        "directors (unrotated)",
        "directors (rotated)",
        "directors (rotated to helix)",
    ]
    for i in range(len(curves)):
        fig = plt.figure(i)
        curve = curves[i]
        label = labels[i]
        ax = fig.gca(projection="3d")
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], label=label)
        ax.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.zlabel('z') # not defined?
        plt.show()

    return p2


def get_helical_data(cgmodel):
    """
        """
    cgmodel = orient_along_z_axis(cgmodel)

    # Get the new backbone particle positions
    backbone_positions = []
    for particle in range(len(cgmodel.positions)):
        if cgmodel.get_particle_type(particle) == "backbone":
            backbone_positions.append(cgmodel.positions[particle])
    backbone_positions = np.array(
        [
            [float(i.in_units_of(unit.angstrom)._value) for i in coord]
            for coord in backbone_positions
        ]
    )

    # radius
    axis_distances = []
    rotations = 0.0
    for position in backbone_positions:
        axis_distance = distance(
            unit.Quantity([float(position[0]), 0.0, 0.0], unit.angstrom),
            unit.Quantity(position, unit.angstrom),
        )
        axis_distances.append(axis_distance)
        if len(axis_distances) > 1:
            rotation = np.arctan(position[1] / position[2]) - last_angle
            last_angle = rotation
            rotations = rotations + rotation
        else:
            rotation = np.arctan(position[1] / position[2])
            last_angle = rotation
            rotations = rotations + rotation

    radius = mean(
        np.array([float(dist.in_units_of(unit.angstrom)._value) for dist in axis_distances])
    )
    particles_per_turn = float(cgmodel.polymer_length / (rotations / 6.28))

    # pitch
    #
    # Shift all coordinates so that the first backbone atom has z=0

    shift = -cgmodel.positions[0][2]._value

    axis_deltas = []
    for position in cgmodel.positions:
        position[0]._value = position[0]._value + shift
        if abs(position[0]._value - cgmodel.positions[0][0]._value) > 0:
            axis_deltas.append(float(position[0]._value - cgmodel.positions[0][0]._value))
    average_delta = mean(axis_deltas)
    pitch = average_delta * particles_per_turn

    return (radius, pitch, particles_per_turn)
