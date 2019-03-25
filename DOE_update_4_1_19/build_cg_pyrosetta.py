import shutil
import os

def buildCopyPyRosetta(original, copy):  # VERY SLOW
    """
    Copy's a working pyrosetta directory into a new file system
    such that we don't ruin a working copy of PyRosetta
    """
    shutil.copytree(original, copy)
    return(os.path.abspath(copy))


def buildCGPyRosetta(path, inputs):
    """
    function which takes in the path to a vanila PyRosetta package and
    adds CG functionality via copying files

    path : path to the pyrosetta package being edited
    inputs : input directory with specific file structure denoting residue types and atom types
    """
    pyrosetta_path = os.path.abspath(path)
    input_path = os.path.abspath(inputs)

    # Pull atom types out of input files

    atom_lines = []
    for files in os.listdir(os.path.join(input_path,'atom_type_sets')):
        if files.endswith(".txt"):
            with open(os.path.join(input_path,'atom_type_sets',files), 'r') as f:  
                for line in f.readlines()[1:]: # Skip header of atom_properties.txt files
                    atom_lines.append(line)


    # opening atom_properties.txt and appending new atom lines

    with open(os.path.join(pyrosetta_path,'pyrosetta','database','chemical','atom_type_sets','fa_standard','atom_properties.txt'), 'a') as atom_file:
        atom_file.write('\n')
        atom_file.write('##############################\n')
        atom_file.write('## Custom Added Atom Types ###\n')
        atom_file.write('##############################\n')
        atom_file.write('\n')
        for atom_line in atom_lines:
            atom_file.write(atom_line)

    # comment out extras.txt files

    with open(os.path.join(pyrosetta_path,'pyrosetta','database','chemical','atom_type_sets','fa_standard','extras.txt'), 'r') as exf:
        extras = exf.readlines()
    
    extras = ['# '+line for line in extras]

    with open(os.path.join(pyrosetta_path,'pyrosetta','database','chemical','atom_type_sets','fa_standard','extras.txt'), 'w') as exf:
        exf.writelines(extras)

    # add residue types into fa_standard residue types
    custom_residue_path = os.path.abspath(os.path.join(pyrosetta_path,'pyrosetta','database','chemical','residue_type_sets','fa_standard','residue_types','custom'))

    if not os.path.isdir(custom_residue_path):
        os.mkdir(os.path.join(pyrosetta_path,'pyrosetta','database','chemical','residue_type_sets','fa_standard','residue_types','custom'))
    
    for files in os.listdir(os.path.join(input_path,'residue_type_sets')):
        if files.endswith('.params'):
            shutil.copy(os.path.join(input_path, 'residue_type_sets', files), custom_residue_path)

    # add residue types to 'residue_types.txt' at the above l-caa residues (give priority for io strings)
    
    with open(os.path.join(pyrosetta_path, 'pyrosetta','database','chemical','residue_type_sets','fa_standard','residue_types.txt'), 'r') as rtf:
        residue_type_lines = rtf.readlines()

    # writting custom lines first
    for files in os.listdir(os.path.join(input_path,'residue_type_sets')):
        residue_type_lines.insert(7, os.path.join('residue_types','custom',files)+'\n')
    # then adding header
    residue_type_lines.insert(7,'### custom residues\n')
    with open(os.path.join(pyrosetta_path, 'pyrosetta','database','chemical','residue_type_sets','fa_standard','residue_types.txt'), 'w') as rtf:
        rtf.writelines(residue_type_lines)        

def add_import_path(path):
    """
    singler liner to import custom version of pyrosetta given any path
    """
    import sys
    sys.path.insert(0, path)


def unBuildCGPyRosetta(path):
    """
    Intentions is to 'unbuild' our CG PyRosetta
    """
    pass

def main():
    pyrosetta_path = os.path.abspath('PyRosetta4.modified')
    
    if not os.path.isdir(pyrosetta_path):
        buildCopyPyRosetta('PyRosetta4.fresh',pyrosetta_path)
        buildCGPyRosetta(pyrosetta_path,'inputs')
    
    add_import_path('PyRosetta4.modified')
    import pyrosetta

    pyrosetta.init()
    print(pyrosetta.__file__)

if __name__ == '__main__':
    main()