This repository contains tools to perform coarse grained simulations in OpenMM.  The repository also enables the use of OpenMM utilities such as Yank.

## To use this repository install it using standard Python conventions:

python setup.py install

## Dependencies:

simtk.unit
simtk.openmm

## Getting started:

Test your installation by opening a new Python session and typing the following:

import cg_openmm

or

from cg_openmm import *

If this test does not work, please check to make sure that the foldamers path is included in the $PYTHONPATH system variable.  This is the most common place for the package installation to fail.

**For full documentation please refer to 'manual.pdf'.**
