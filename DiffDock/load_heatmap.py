import pandas as pd
import numpy as np
from rdkit import Chem
import pymol
from pymol import cmd
from pymol.cgo import COLOR, ALPHA, SPHERE

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

# Function to create heatmap data
def create_heatmap_data(coords):
    heatmap, xedges, yedges = np.histogram2d(coords[:, 0], coords[:, 1], bins=50)
    return heatmap, xedges, yedges

# Create heatmaps
active_heatmap, xedges, yedges = create_heatmap_data(active_coords_flat)
inactive_heatmap, xedges, yedges = create_heatmap_data(inactive_coords_flat)

def create_pymol_cgo_heatmap(heatmap, xedges, yedges, color):
    cgo = []
    max_val = np.max(heatmap)
    for i in range(len(xedges)-1):
        for j in range(len(yedges)-1):
            if heatmap[i, j] > 0:
                alpha = heatmap[i, j] / max_val * 0.5  # Adjust alpha for transparency
                cgo.extend([
                    COLOR, color[0], color[1], color[2],
                    ALPHA, alpha,
                    SPHERE, xedges[i], yedges[j], 0, 0.5  # Adjust sphere size as needed
                ])
    return cgo

# Create CGO objects for active and inactive heatmaps
active_cgo = create_pymol_cgo_heatmap(active_heatmap, xedges, yedges, [1.0, 0.0, 0.0])  # Red color
inactive_cgo = create_pymol_cgo_heatmap(inactive_heatmap, xedges, yedges, [0.0, 0.0, 1.0])  # Blue color

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

# Align CGO objects to GLP1
cmd.matrix_copy("GLP1", "active_heatmap")
cmd.matrix_copy("GLP1", "inactive_heatmap")

# Final visualization adjustments
cmd.zoom("GLP1")
cmd.orient()

# Keep the PyMOL application open
cmd.quit = lambda: None  # Disable quit command to keep PyMOL open
# cmd.do('quit')  # Placeholder to ensure script runs to end
