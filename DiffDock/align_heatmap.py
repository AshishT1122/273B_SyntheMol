import pandas as pd
import numpy as np
from rdkit import Chem
import pymol
from pymol import cmd
from pymol.cgo import COLOR, ALPHA, SPHERE
from scipy.spatial.transform import Rotation as R

# Initialize PyMOL in interactive mode
pymol.finish_launching()

# Load data
df = pd.read_csv('model_data.csv')
df = pd.concat([df[df['Comment'] == 'inactive'].sample(20), df[df['Comment'] == 'active'].sample(20)])

# Extract file paths
active_sdf_files = [f"diffdock_chembl_output/{row['Molecule ChEMBL ID']}/{row['Filepath']}" for index, row in df.iterrows() if row['Comment'] == 'active']
inactive_sdf_files = [f"diffdock_chembl_output/{row['Molecule ChEMBL ID']}/{row['Filepath']}" for index, row in df.iterrows() if row['Comment'] == 'inactive']

# Function to extract coordinates from an SDF file
def extract_coordinates(sdf_file):
    mol = Chem.SDMolSupplier(sdf_file)[0]
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return coords

# Extract coordinates
active_coords = [extract_coordinates(f) for f in active_sdf_files]
inactive_coords = [extract_coordinates(f) for f in inactive_sdf_files]
active_coords_flat = np.vstack(active_coords)
inactive_coords_flat = np.vstack(inactive_coords)

# Function to create 3D heatmap data
def create_heatmap_data_3d(coords):
    heatmap, edges = np.histogramdd(coords, bins=50)
    return heatmap, edges

# Create 3D heatmaps
active_heatmap, edges = create_heatmap_data_3d(active_coords_flat)
inactive_heatmap, edges = create_heatmap_data_3d(inactive_coords_flat)

def create_pymol_cgo_heatmap_3d(heatmap, edges, color):
    cgo = []
    max_val = np.max(heatmap)
    xedges, yedges, zedges = edges
    for i in range(len(xedges)-1):
        for j in range(len(yedges)-1):
            for k in range(len(zedges)-1):
                if heatmap[i, j, k] > 0:
                    alpha = heatmap[i, j, k] / max_val * 0.5  # Adjust alpha for transparency
                    cgo.extend([
                        COLOR, color[0], color[1], color[2],
                        ALPHA, alpha,
                        SPHERE, (xedges[i] + xedges[i+1])/2, (yedges[j] + yedges[j+1])/2, (zedges[k] + zedges[k+1])/2, 0.5  # Adjust sphere size as needed
                    ])
    return cgo

# Create 3D CGO objects for active and inactive heatmaps
active_cgo = create_pymol_cgo_heatmap_3d(active_heatmap, edges, [1.0, 0.0, 0.0])  # Red color
inactive_cgo = create_pymol_cgo_heatmap_3d(inactive_heatmap, edges, [0.0, 0.0, 1.0])  # Blue color

# Check if CGO objects are created correctly
print(f"Active CGO length: {len(active_cgo)}")
print(f"Inactive CGO length: {len(inactive_cgo)}")

# Print some coordinates for debugging
print("Sample coordinates for active heatmap:")
print(active_coords_flat[:5])
print("Sample coordinates for inactive heatmap:")
print(inactive_coords_flat[:5])

import pymol.cmd as cmd

# Load GLP-1 structure
cmd.load("GLP1.pdb")

# Load CGO objects into PyMOL
cmd.load_cgo(active_cgo, "active_heatmap")
cmd.load_cgo(inactive_cgo, "inactive_heatmap")

# Adjust PyMOL visualization settings
cmd.set('cgo_transparency', 0.8)  # Set transparency level for CGO objects
cmd.set('transparency', 0.2, "GLP1")
cmd.show("surface", "GLP1")  # Show surface for GLP1
cmd.color("gray70", "GLP1")  # Color GLP1 surface

# Ensure the CGO objects are visible
cmd.show("dots", "active_heatmap")
cmd.show("dots", "inactive_heatmap")

# Function to align heatmap coordinates to the GLP-1 structure
def align_coordinates(source_coords, target_coords):
    # Calculate centroids
    source_centroid = np.mean(source_coords, axis=0)
    target_centroid = np.mean(target_coords, axis=0)

    # Translate coordinates to origin
    source_coords_centered = source_coords - source_centroid
    target_coords_centered = target_coords - target_centroid

    # Singular Value Decomposition (SVD) for rotation matrix
    H = np.dot(source_coords_centered.T, target_coords_centered)
    U, S, Vt = np.linalg.svd(H)
    rotation_matrix = np.dot(Vt.T, U.T)

    # Apply rotation and translation
    aligned_coords = np.dot(source_coords_centered, rotation_matrix) + target_centroid
    return aligned_coords

# Identify three non-colinear reference points
def get_reference_points(pdb_file, atom_indices):
    cmd.load(pdb_file, "temp_structure")
    coords = [cmd.get_atom_coords(f"temp_structure and id {idx}") for idx in atom_indices]
    cmd.delete("temp_structure")
    return np.array(coords)

# Example of how to get atom IDs programmatically
atom_ids = []

def capture_id(selection):
    stored.ids = []
    cmd.iterate(selection, "stored.ids.append(ID)")
    if len(stored.ids) == 0:
        raise ValueError(f"No atoms found for selection: {selection}")
    return stored.ids[0]

cmd.select("atom1", "/GLP1/A/TYR`241/CA")
atom_ids.append(capture_id("atom1"))

cmd.select("atom2", "/GLP1/B/ARG`105/CG")
atom_ids.append(capture_id("atom2"))

cmd.select("atom3", "/GLP1/A/GLU`50/OE1")
atom_ids.append(capture_id("atom3"))

print("Selected Atom IDs:", atom_ids)

reference_atom_indices = [atom_ids[0], atom_ids[1], atom_ids[2]]  # Replace with your actual atom IDs
reference_points = get_reference_points("GLP1.pdb", reference_atom_indices)
print("Reference points (non-colinear):", reference_points)

# Apply alignment to CGO objects
aligned_active_coords = align_coordinates(active_coords_flat, reference_points)
aligned_inactive_coords = align_coordinates(inactive_coords_flat, reference_points)

# Create 3D CGO objects for aligned heatmaps
aligned_active_cgo = create_pymol_cgo_heatmap_3d(create_heatmap_data_3d(aligned_active_coords)[0], edges, [1.0, 0.0, 0.0])  # Red color
aligned_inactive_cgo = create_pymol_cgo_heatmap_3d(create_heatmap_data_3d(aligned_inactive_coords)[0], edges, [0.0, 0.0, 1.0])  # Blue color

# Load aligned CGO objects into PyMOL
cmd.load_cgo(aligned_active_cgo, "aligned_active_heatmap")
cmd.load_cgo(aligned_inactive_cgo, "aligned_inactive_heatmap")

# Final visualization adjustments
cmd.zoom("GLP1")
cmd.orient()

# Keep the PyMOL application open
cmd.quit = lambda: None  # Disable quit command to keep PyMOL open
# cmd.do('quit')  # Placeholder to ensure script runs to end
