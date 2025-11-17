"""
Generate initial structure for Al-Cu nanoparticle using ASE
"""
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import write
import os

def generate_alcu_nanoparticle(num_al=30, num_cu=20, radius=10.0, cell_size=30.0):
    """
    Generate Al-Cu binary nanoparticle initial structure using ASE
    Parameters:
        num_al: Number of Al atoms
        num_cu: Number of Cu atoms
        radius: Nanoparticle radius (Å)
        cell_size: Simulation box size (Å)
    Returns:
        ASE Atoms object
    """
    # Create atom list
    symbols = ['Al'] * num_al + ['Cu'] * num_cu
    total_atoms = num_al + num_cu
    atoms = Atoms(symbols, cell=[cell_size, cell_size, cell_size], pbc=True)
    center = np.array([cell_size/2, cell_size/2, cell_size/2])

    # Get covalent radii for distance checking
    atomic_numbers = atoms.get_atomic_numbers()
    radii = [covalent_radii[num] for num in atomic_numbers]
    min_distances = np.zeros((total_atoms, total_atoms))
    for i in range(total_atoms):
        for j in range(total_atoms):
            if i != j:
                min_distances[i][j] = (radii[i] + radii[j]) * 0.95  # 95% of covalent radius sum

    # Generate random atomic positions within sphere
    positions = []
    for i in range(total_atoms):
        while True:
            # Generate random point in spherical coordinates
            r = np.random.random() * radius
            theta = np.random.random() * np.pi
            phi = np.random.random() * 2 * np.pi

            # Convert to Cartesian coordinates
            x = center[0] + r * np.sin(theta) * np.cos(phi)
            y = center[1] + r * np.sin(theta) * np.sin(phi)
            z = center[2] + r * np.cos(theta)
            pos = np.array([x, y, z])

            # Check distance to existing atoms
            valid = True
            for j, p in enumerate(positions):
                if np.linalg.norm(pos - p) < min_distances[i][j]:
                    valid = False
                    break
            if valid:
                positions.append(pos)
                break

    atoms.positions = np.array(positions)
    return atoms

if __name__ == '__main__':
    # Generate structure
    nanoparticle = generate_alcu_nanoparticle(num_al=30, num_cu=20, radius=10.0)
    
    # Get lattice parameters for XYZ file
    cell = nanoparticle.get_cell()
    lattice_comment = f"Lattice=\"{cell[0][0]} {cell[0][1]} {cell[0][2]} {cell[1][0]} {cell[1][1]} {cell[1][2]} {cell[2][0]} {cell[2][1]} {cell[2][2]}\" pbc=\"T T T\""
    
    # Save in extended XYZ format with lattice information
    write('init.xyz', nanoparticle, comment=lattice_comment)
    print(f"Successfully generated Al-Cu nanoparticle structure with {len(nanoparticle)} atoms")
    print(f"Structure saved to init.xyz with lattice parameters")
    
    # Optional: Display structure information
    print(f"Chemical formula: {nanoparticle.get_chemical_formula()}")
    print(f"Unit cell dimensions: {nanoparticle.get_cell()}")