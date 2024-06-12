import os
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from pymol import cmd

# Initialize PyMOL in interactive mode
pymol.finish_launching()

# Load data
df = pd.read_csv('model_data.csv')
df = pd.concat([df[df['Comment'] == 'inactive'].sample(20), df[df['Comment'] == 'active'].sample(20)])

# Extract file paths
active_sdf_dirs = [f"diffdock_chembl_output/{row['Molecule ChEMBL ID']}" for index, row in df.iterrows() if row['Comment'] == 'active']
inactive_sdf_dirs = [f"diffdock_chembl_output/{row['Molecule ChEMBL ID']}" for index, row in df.iterrows() if row['Comment'] == 'inactive']

# Load the modified GLP-1 protein
modified_glp1 = "modified_glp1.pdb"
cmd.load(modified_glp1, "glp1")
cmd.show("surface", "glp1")
cmd.set("transparency", 0.08, "glp1")
cmd.color("gray70", "glp1")

# Function to load SDF and color it
def load_color_sdf(sdf_file, chembl_id, color):
    # Load the SDF file
    cmd.load(sdf_file, chembl_id)
    
    # Color the compound
    cmd.color(color, chembl_id)
    
    # Set the title of the molecule
    cmd.set_title(chembl_id, 0, chembl_id)

# Process active compounds
for sdf_dir in active_sdf_dirs:
    sdf_file = os.path.join(sdf_dir, 'rank1.sdf')
    if os.path.exists(sdf_file):
        chembl_id = os.path.basename(sdf_dir)
        load_color_sdf(sdf_file, chembl_id, 'red')

# Process inactive compounds
for sdf_dir in inactive_sdf_dirs:
    sdf_file = os.path.join(sdf_dir, 'rank1.sdf')
    if os.path.exists(sdf_file):
        chembl_id = os.path.basename(sdf_dir)
        load_color_sdf(sdf_file, chembl_id, 'blue')

# Function to extract coordinates from an SDF file
def extract_coordinates(sdf_file):
    mol = Chem.SDMolSupplier(sdf_file)[0]
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return coords

# Extract coordinates for active and inactive molecules
active_sdf_files = [os.path.join(sdf_dir, 'rank1.sdf') for sdf_dir in active_sdf_dirs if os.path.exists(os.path.join(sdf_dir, 'rank1.sdf'))]
inactive_sdf_files = [os.path.join(sdf_dir, 'rank1.sdf') for sdf_dir in inactive_sdf_dirs if os.path.exists(os.path.join(sdf_dir, 'rank1.sdf'))]

active_coords = [extract_coordinates(f) for f in active_sdf_files]
inactive_coords = [extract_coordinates(f) for f in inactive_sdf_files]

# Flatten the list of coordinates
active_coords_flat = np.vstack(active_coords)
inactive_coords_flat = np.vstack(inactive_coords)

# Create heatmap data
def create_heatmap_data(coords):
    heatmap, xedges, yedges = np.histogram2d(coords[:, 0], coords[:, 1], bins=50)
    return heatmap, xedges, yedges

active_heatmap, xedges, yedges = create_heatmap_data(active_coords_flat)
inactive_heatmap, xedges, yedges = create_heatmap_data(inactive_coords_flat)

# Plotting the heatmaps
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.heatmap(active_heatmap.T, norm=Normalize(), cmap='Reds', cbar=True)
plt.title('Active Molecules Heatmap')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

plt.subplot(1, 2, 2)
sns.heatmap(inactive_heatmap.T, norm=Normalize(), cmap='Blues', cbar=True)
plt.title('Inactive Molecules Heatmap')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

plt.tight_layout()
plt.savefig("active_and_inactive_heatmap.png")

# Keep the PyMOL window open
#cmd.quit()
