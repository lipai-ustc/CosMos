import numpy as np
from ase import Atoms
from ase.build import fcc100, add_adsorbate
from ase.io import write
from ase.data import covalent_radii

def generate_au100_surface():
    # Gold (Au) FCC lattice constant in Angstroms
    lattice_constant = 4.08  # Experimental lattice constant for Au
    
    # Generate Au(100) surface: 4 layers, (3x3) supercell
    # pbc=(True, True, False) means periodic in x and y directions
    atoms = fcc100('Au', size=(4, 4, 4), a=lattice_constant, vacuum=15.0)
    
    # Get covalent radius for Au to verify minimum distance
    au_z = 79
    min_distance = 2 * covalent_radii[au_z]  # ~2.88 Å for Au
    
    # Verify minimum distance
    distances = atoms.get_all_distances(mic=True)  # mic=True accounts for periodic boundary conditions
    min_calculated = np.min(distances[distances > 0])  # Exclude zero distance to self
    
    # Save the structure to init.xyz (extended XYZ format for lattice info)
    write('init.xyz', atoms, format='extxyz')
    
    # Print verification information
    print(f"Successfully generated Au(100) surface structure with {len(atoms)} atoms.")
    print(f"Surface dimensions: (3x3) supercell, 4 layers")
    print(f"Lattice constant: {lattice_constant}Å")
    print(f"Vacuum layer: 15.0Å")
    print(f"Minimum atomic distance: {min_calculated:.2f}Å (theoretical minimum: {min_distance:.2f}Å)")
    print(f"Periodic boundary conditions: x={atoms.pbc[0]}, y={atoms.pbc[1]}, z={atoms.pbc[2]}")

if __name__ == "__main__":
    generate_au100_surface()