# OpenMM_CG_polymers
Contains CG heteropolymer examples and data for the nonbiological foldamers project.

Last updated by Garrett A. Meek on 2-6-2019

We recommend the following steps to setup a new computing environment for OpenMM simulations:

1.  Install the latest release of Anaconda 3
2.  Using the 'conda' package environment, and the interpreter for Python3.7 (in $Anaconda3/bin/):
    A) Install openmm
    B) Install openmmtools
    C) Install yank
    D) Install MSM-Builder
3.  Install an IDE (like PyCharm or Visual Studio)
4.  Import the current repository in the IDE
5.  Setup the IDE environment to reference the Python3.7 interpreter (in $Anaconda3/bin/)

Possible issues:

From minimal use, Yank seems to be more compatible with Anaconda2.  In at least one instance I've had to manually install the Python source to get the Anaconda-based Python3.7 interpreter to import it correctly.  (As opposed to installing directly with the conda package manager or using the IDE's package manager, if it has one.)
