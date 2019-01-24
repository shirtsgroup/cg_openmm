# This script was written using the following URL as a reference:
# https://openmmtools.readthedocs.io/en/latest/gettingstarted.html#parallel-tempering

from simtk import openmm, unit
import math, os
from random import random, randint
from openmmtools import *
from openmmtools.testsystems import TestSystem, get_data_filename
from simtk.openmm import app
import remd
from remd import ReplicaExchange

# !! BEGIN USER_DEFINED SIMULATION INPUT !!

topology_file_name="/home/gmeek/software/anaconda3/pkgs/openmmtools-0.16.0-py27_0/lib/python2.7/site-packages/openmmtools/data/cb7-b2/complex-vacuum.prmtop"
coordinate_file_name="/home/gmeek/software/anaconda3/pkgs/openmmtools-0.16.0-py27_0/lib/python2.7/site-packages/openmmtools/data/cb7-b2/complex-vacuum.inpcrd"
replica_list=[300, 310, 330, 370, 450] # List of temperatures for each replica
time_step=2.0 # femtoseconds
n_steps=50
constraints=None

# !! END OF USER-DEFINED SIMULATION INPUT

class CoarseGrainOctamer(testsystems.TestSystem):

    def __init__(self, constraints=constraints, hydrogenMass=None, **kwargs):

        TestSystem.__init__(self, **kwargs)
        prmtop_filename = get_data_filename("data/cb7-b2/complex-vacuum.prmtop")
        crd_filename = get_data_filename("data/cb7-b2/complex-vacuum.inpcrd")

        prmtop = app.AmberPrmtopFile(prmtop_filename)
        system = prmtop.createSystem(implicitSolvent=app.GBn, constraints=constraints, nonbondedCutoff=None, hydrogenMass=hydrogenMass)

        # Extract topology
        self.topology = prmtop.topology

        # Read positions.
        inpcrd = app.AmberInpcrdFile(crd_filename)
        positions = inpcrd.getPositions(asNumpy=True)

        self.system, self.positions = system, positions


# Parallel tempering begins here:

# Initialize thermodynamic states at different temperatures.
host_guest = CoarseGrainOctamer()
protocol = {'temperature': replica_list * unit.kelvin}
thermo_states = states.create_thermodynamic_state_protocol(host_guest.system, protocol)

# Initialize replica initial configurations.
sampler_states = [states.SamplerState(positions=host_guest.positions) for _ in thermo_states]

# Propagate the replicas with Langevin dynamics.
langevin_move = mcmc.LangevinSplittingDynamicsMove(timestep=time_step*unit.femtosecond,n_steps=n_steps)

# Run the parallel tempering simulation.
parallel_tempering = ReplicaExchange(thermo_states, sampler_states, langevin_move)
parallel_tempering.run()

