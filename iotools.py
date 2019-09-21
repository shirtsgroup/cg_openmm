from simtk.openmm.app.pdbfile import PDBFile
import numpy as np
import math, random
from simtk import unit

def write_bonds(CGModel,pdb_object):
        """
        Writes the bonds from an input CGModel class object to the file object 'pdb_object', using PDB 'CONECT' syntax.

        :param CGModel: CGModel() class object
        :type CGModel: class

        :param pdb_object: File object to which we will write the bond list
        :type pdb_object: file

        """
        bond_list = CGModel.bond_list
        for bond in bond_list:
         if int(bond[0]) < int(bond[1]):
          pdb_object.write("CONECT"+str("{:>5}".format(bond[0]+1))+str("{:>5}".format(bond[1]+1))+"\n")      
         else:
          pdb_object.write("CONECT"+str("{:>5}".format(bond[1]+1))+str("{:>5}".format(bond[0]+1))+"\n")
        pdb_object.write(str("END\n"))
        return

def write_pdbfile_without_topology(CGModel,filename,energy=None):
        """
        Writes the positions from an input CGModel class object to the file 'filename'.

        :param CGModel: CGModel() class object
        :type CGModel: class

        :param filename: Path to the file where we will write PDB coordinates.
        :type filename: str

        :param energy: Energy to write to the PDB file, default = None
        :type energy: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        """

        pdb_object = open(filename,"w")
        if energy != None:
          pdb_object.write("## The OpenMM potential energy for this structure is: "+str(energy)+"\n")
          pdb_object.write("## with the following parameter settings:\n")
          pdb_object.write("## sigma = "+str(CGModel.sigmas['bb_bb_sigma'])+", epsilon = "+str(CGModel.epsilons['bb_bb_eps'])+"\n")

        coordinates = CGModel.positions
        bead_index = 1
        for monomer_index in range(CGModel.polymer_length):
          monomer_type = CGModel.sequence[monomer_index]
          element_index = 1
          for backbone_bead in range(monomer_type['backbone_length']):

            if monomer_index in list([0,CGModel.polymer_length-1]):
             pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+" X"+str(element_index)+str("{:>6}".format(str("MT")))+" A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(round(coordinates[bead_index-1][0].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][1].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][2].in_units_of(unit.angstrom)._value,2)))+"  1.00  0.00\n"))
            else:
             pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+" X"+str(element_index)+str("{:>6}".format(str("M")))+" A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(round(coordinates[bead_index-1][0].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][1].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][2].in_units_of(unit.angstrom)._value,2)))+"  1.00  0.00\n"))
            bead_index = bead_index + 1
            element_index = element_index + 1

            if backbone_bead in [monomer_type['sidechain_positions']]:
              for sidechain_bead in range(monomer_type['sidechain_length']):
                if monomer_index in list([0,CGModel.polymer_length-1]):
                 pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+" A"+str(element_index)+str("{:>6}".format(str("MT")))+" A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(round(coordinates[bead_index-1][0].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][1].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][2].in_units_of(unit.angstrom)._value,2)))+"  1.00  0.00\n"))
                else:
                 pdb_object.write(str("ATOM"+str("{:>7}".format(bead_index))+" A"+str(element_index)+str("{:>6}".format(str("M")))+" A"+str("{:>4}".format(monomer_index+1))+"     "+str("{:>7}".format(round(coordinates[bead_index-1][0].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][1].in_units_of(unit.angstrom)._value,2)))+" "+str("{:>7}".format(round(coordinates[bead_index-1][2].in_units_of(unit.angstrom)._value,2)))+"  1.00  0.00\n"))
                bead_index = bead_index + 1
                element_index = element_index + 1
        pdb_object.write(str("TER\n"))

        write_bonds(CGModel,pdb_object)
        pdb_object.close()
        return
