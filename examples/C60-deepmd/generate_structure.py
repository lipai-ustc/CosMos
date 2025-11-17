import numpy as np
from ase import Atoms
from ase import Atom
from ase.io import write


def generate_c60_structure():
    # Create a sufficiently large cubic box (40Å x 40Å x 40Å)
    box_size = 40.0  # Angstroms
    atoms = Atoms(cell=[box_size, box_size, box_size])
    
    # Carbon covalent radius and bond parameters
    covalent_radius = 0.77  # Covalent radius of carbon in Å
    cc_bond_length = 1.4  # Average C-C bond length in C60
    min_threshold = 0.7 * cc_bond_length  # Minimum allowed distance (0.98 Å)
    max_threshold = 1.5 * cc_bond_length  # Maximum allowed distance (2.1 Å)
    max_placement_threshold = 2.0  # Maximum radius expansion per atom
    max_attempts = 1000  # Maximum attempts to place each atom
    
    # Place atoms in the middle region (avoid edges)
    center_region = 0.7  # Use 70% of the box in the center
    min_coord = box_size * (1 - center_region) / 2
    max_coord = box_size - min_coord
    
    # Add 60 carbon atoms
    # Initialize atoms object
    atoms = Atoms()
    box_size = 40.0
    atoms.set_cell([box_size, box_size, box_size])
    atoms.set_pbc([True, True, True])
    
    # Add first atom directly at the center of the box
    center = [box_size / 2, box_size / 2, box_size / 2]
    atoms.append(Atom('C', center))
    current_max_radius = 0  # First atom is at center
    
    # Generate remaining 59 atoms
    for i in range(1, 60):
        placed = False
        attempts = 0
        while not placed and attempts < max_attempts:
            # Create temporary copy to test atom placement
            atoms_attempt = atoms.copy()
            
            # Generate random coordinates in spherical shell distribution
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0, placement_radius_max)
            
            # Convert spherical to Cartesian coordinates
            x = center[0] + r * np.sin(theta) * np.cos(phi)
            y = center[1] + r * np.sin(theta) * np.sin(phi)
            z = center[2] + r * np.cos(theta)
            coords = [x, y, z]
            
            # Calculate cluster center and maximum radius dynamically
            if len(atoms) == 0:
                # For first atom, use center of the box
                center = np.array([box_size/2, box_size/2, box_size/2])
                current_max_radius = 0.0
            else:
                # Calculate geometric center of existing cluster
                center = np.mean(atoms.get_positions(), axis=0)
                # Calculate distances from center to all atoms
                distances_from_center = np.linalg.norm(atoms.get_positions() - center, axis=1)
                current_max_radius = np.max(distances_from_center)
            
            # Define new atom placement range based on cluster geometry
            placement_radius_max = current_max_radius + max_placement_threshold
            
            # Generate random coordinates in spherical shell around cluster center
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0, placement_radius_max)
            
            # Convert spherical to Cartesian coordinates
            coords = center + np.array([
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ])
            
            # Ensure coordinates stay within simulation box
            coords = np.clip(coords, min_coord, max_coord)
            
            # Use copy to test atom placement
            atoms_attempt = atoms.copy()
            atoms_attempt.append(Atom('C', coords))
            
            # Calculate distances from new atom to all existing atoms
            if len(atoms_attempt) > 1:
                distances = atoms_attempt.get_distances(-1, range(len(atoms_attempt)-1))
            else:
                distances = []
            current_min_distance = np.min(distances) if len(distances) > 0 else min_threshold + 0.1
            
            # Check if minimum distance is within valid range
            if current_min_distance > min_threshold and current_min_distance < max_threshold:
                atoms = atoms_attempt
                placed = True
                print(f"Successfully placed atom {len(atoms)}/60. Minimum distance: {current_min_distance:.2f} Å")
            else:
                placed = False
                if attempts % 100 == 0:
                    print(f"Attempt {attempts}: Minimum distance {current_min_distance:.2f} Å not in valid range")
            
            attempts += 1
        
        if not placed:
            raise RuntimeError(f"Could not place atom after {max_attempts} attempts. Try increasing max_placement_threshold.")
    
    # Center the final structure in the box
    atoms.center()
    
    # Save the structure to init.xyz
    write('init.xyz', atoms)
    print(f"Successfully generated C60 structure with {len(atoms)} atoms.")
    print(f"Box size: {box_size}Å x {box_size}Å x {box_size}Å")
    print(f"Minimum atomic distance threshold: {min_threshold:.2f}Å")
    print(f"Maximum atomic distance threshold: {max_threshold:.2f}Å")

if __name__ == "__main__":
    generate_c60_structure()