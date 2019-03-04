###!/usr/local/bin/env python

# This test involved the following changes:
#
# 1) Changed the test system from harmonic oscillators to the alanine dipeptide
# 2) Added MPI capability
# 3) Increased the number of MC iterations, in an attempt to decrease the maximum error
#    during prediction of the dimensionless free energy differences
# 4) Changed select module names

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

# Non-scientific python packages needed for this protocol

import contextlib
import copy
import inspect
import math
import os
import pickle
import sys
import timeit
from io import StringIO

# Scientific and molecular simulation packages needed for this protocol

import numpy as np
import openmmtools as mmtools
import scipy.integrate
import yaml
from nose.plugins.attrib import attr
from nose.tools import assert_raises
from openmmtools import testsystems
from simtk import openmm, unit

# This is where replica exchange utilities are imported from Yank

from yank import mpi
from yank.multistate import MultiStateReporter, MultiStateSampler, ReplicaExchangeSampler, ParallelTemperingSampler, SAMSSampler
from yank.multistate import ReplicaExchangeAnalyzer, SAMSAnalyzer
from yank.multistate.multistatereporter import _DictYamlLoader
from yank.utils import config_root_logger  # This is only function tying these test to the main YANK code
from yank.commands import analyze

#-netcdf=FILEPATH             Path to the NetCDF file.
#  --checkpoint=FILEPATH         Path to the NetCDF checkpoint file if not the default name inferned from "netcdf" option
#  --state=STATE_IDX             Index of the alchemical state for which to extract the trajectory
#  --replica=REPLICA_IDX         Index of the replica for which to extract the trajectory
#  --trajectory=FILEPATH         Path to the trajectory file to create (extension determines the format)

analyze.dispatch_extract_trajectory({'--checkpoint': 'test_storage_checkpoint.nc',
 '--fulltraj': True,
 '--netcdf': 'test_storage.nc',
 '--output': 'traj1',
 '--replica': 1,
 '--trajectory': 1,
 '--state': 1,
 'extract-trajectory': True,
 '--discardequil': False,
 '--distcutoff': None,
 '--end': None,
 '--energycutoff': None,
 '--format': None,
 '--serial': None,
 '--skip': None,
 '--skipunbiasing': False,
 '--start': None,
 '--imagemol': None,
 '--nosolvent': None,
 '--verbose': False})
