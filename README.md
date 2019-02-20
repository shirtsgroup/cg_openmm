# OpenMM_CG_polymers
Contains CG heteropolymer examples and data for the nonbiological foldamers project.

Last updated by Garrett A. Meek on 2-20-2019

We recommend the following steps to setup a new computing environment for OpenMM simulations:

# NOTE: IF YOU PLAN TO USE 'MD-Traj' (for which the latest compatible python version is 3.5),
# you may experience issues if you also try to run 'yank' (which requires python3.7).
# One solution is to use python2.7, which is compatible with both packages

1.  Install the latest release of Anaconda 3
2.  Using the 'conda' package environment, and the interpreter for Python3.7 (in $Anaconda3/bin/):
    A) Install openmm
    B) Install openmmtools
    C) Install yank
    D) Install MSM-Builder
3.  Install an IDE (like PyCharm or Visual Studio)
4.  Import the current repository in the IDE
5.  Setup the IDE environment to reference the Python3.7 interpreter (in $Anaconda3/bin/)
