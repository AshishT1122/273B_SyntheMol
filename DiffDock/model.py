from rdkit import Chem
from torch_geometric.data import Data
import torch
import pandas as pd
from torch_geometric.data import DataLoader, Batch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import random
from sklearn.metrics import classification_report
import numpy as np

df = pd.read_csv('model_data.csv')

graphs = []
labels = []

def get_atom_features(mol):
    features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        aromatic = atom.GetIsAromatic()
        chirality = atom.GetChiralTag()
        formal_charge = atom.GetFormalCharge()
        num_hydrogens = atom.GetTotalNumHs()
        num_valence = atom.GetTotalValence()
        hybridization = atom.GetHybridization()
        is_in_ring = atom.IsInRing()

        # Converting hybridization to an integer
        hybridization = {
            Chem.rdchem.HybridizationType.SP: 1,
            Chem.rdchem.HybridizationType.SP2: 2,
            Chem.rdchem.HybridizationType.SP3: 3,
            Chem.rdchem.HybridizationType.SP3D: 4,
            Chem.rdchem.HybridizationType.SP3D2: 5,
            Chem.rdchem.HybridizationType.UNSPECIFIED: 0
        }.get(hybridization, 0)
        
        # Append all atom features to the feature list
        feature_vector = [
            atomic_num,
            int(aromatic),
            int(chirality != Chem.ChiralType.CHI_UNSPECIFIED),
            formal_charge,
            num_hydrogens,
            num_valence,
            hybridization,
            int(is_in_ring)
        ]
        features.append(feature_vector)
    
    return features

def molecule_to_graph(mol):
    bonds = mol.GetBonds()
    
    node_features = get_atom_features(mol)
    edge_index = []
    edge_features = []

    for bond in bonds:
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append((start, end))
        edge_index.append((end, start))  # because graph is undirected
        edge_features += [bond.GetBondTypeAsDouble(), bond.GetBondTypeAsDouble()]  # simple edge feature

    # Convert to tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

for index, row in df.iterrows():
    mol = Chem.MolFromMolFile(f"diffdock_chembl_output/{row['Molecule ChEMBL ID']}/{row['Filepath']}")
    graphs.append(molecule_to_graph(mol))
    labels.append(1 if row['Comment'] == 'active' else 0)

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.pool(x, batch)
        return F.log_softmax(x, dim=1)
    
# Determine the number of instances per class needed to balance the classes
count_active = sum(1 for label in labels if label == 1)
count_inactive = len(labels) - count_active
difference = count_inactive - count_active

# Collect indices of the minority class
active_indices = [i for i, label in enumerate(labels) if label == 1]

# Randomly choose entries from the minority class to duplicate
additional_indices = np.random.choice(active_indices, difference)

# Create a new balanced dataset
graphs_resampled = graphs + [graphs[i] for i in additional_indices]
labels_resampled = labels + [labels[i] for i in additional_indices]

# Shuffle the new dataset to mix the duplicated entries
combined = list(zip(graphs_resampled, labels_resampled))
random.shuffle(combined)
graphs_resampled, labels_resampled = zip(*combined)

# Split into training and validation sets
split_size = int(0.8 * len(graphs_resampled))
train_data = combined[:split_size]
val_data = combined[split_size:]

# Assuming num_features is the number of node features and num_classes is 2 (active, inactive)
model = GCN(num_features=8, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()

def collate(data_list):
    batch, labels = Batch.from_data_list([data[0] for data in data_list]), torch.tensor([data[1] for data in data_list])
    return batch, labels

# Update DataLoader needed
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate)

def train():
    model.train()
    total_loss = 0
    for batch, labels in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_func(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate():
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch, labels in val_loader:
            out = model(batch)
            loss = loss_func(out, labels)
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())  # Store predictions
            all_labels.extend(labels.cpu().tolist())  # Store true labels
            correct += preds.eq(labels).sum().item()

    
    # Calculate confusion matrix
    cr = classification_report(all_labels, all_preds, target_names=['Inactive', 'Active'])

    return total_loss / len(val_loader), correct / len(val_loader.dataset), cr

# Running the training process
for epoch in range(100):  # Number of epochs
    train_loss = train()
    val_loss, val_acc, cr = validate()
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

val_loss, val_acc, cr = validate()
print(f'Validation Loss: {val_loss}')
print("Confusion Matrix:\n", cr)



