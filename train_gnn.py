import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Optional

# --- Configuration ---
DATA_PATH = 'data/training_data.xyz'
MODEL_SAVE_PATH = 'best_model.pth'
BOX_SIZE = 11.76  # 3 * 3.92
LEARNING_RATE = 0.01
EPOCHS = 51
LAMBDA_F = 10.0   # Weight for force loss

def get_pbc_distances(pos: torch.Tensor, box_dims: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise Euclidean distances respecting Periodic Boundary Conditions (MIC).
    
    Args:
        pos: (N, 3) tensor of atomic positions.
        box_dims: (3,) tensor of box dimensions (Lx, Ly, Lz).
    
    Returns:
        (N, N) tensor of pairwise distances.
    """
    # Broadcast to get displacement matrix (N, N, 3)
    delta = pos.unsqueeze(0) - pos.unsqueeze(1)
    
    # Minimum Image Convention
    box = box_dims.view(1, 1, 3)
    delta = delta - box * torch.round(delta / box)
    
    # Euclidean Distance (epsilon added for numerical stability)
    return torch.sqrt((delta**2).sum(dim=2) + 1e-8)

class PtGNN(nn.Module):
    """
    Graph Neural Network for Platinum defects.
    Predicts scalar potential energy based on local atomic environments.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 16):
        super(PtGNN, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, positions: torch.Tensor, features: torch.Tensor, box_dims: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute total system energy.
        """
        N = positions.shape[0]
        
        # 1. Compute Soft Adjacency Matrix (Gaussian Cutoff)
        dists = get_pbc_distances(positions, box_dims)
        mu, sigma = 2.77, 0.5
        A = torch.exp(-(dists - mu)**2 / sigma**2)
        
        # Zero out diagonal (self-interactions)
        mask = 1.0 - torch.eye(N).to(positions.device)
        A = A * mask

        # 2. Message Passing
        h = self.encoder(features)
        h_aggr = torch.matmul(A, h)
        h_new = self.activation(h_aggr)

        # 3. Readout
        e_atomic = self.decoder(h_new)
        e_total = torch.sum(e_atomic)
        
        return e_total

def load_dataset(filename: str, n_atoms: int = 108, limit: Optional[int] = None) -> List[Tuple]:
    """
    Parses LAMMPS custom dump file and normalizes energy targets.
    
    Returns:
        List of tuples: (pos, feat, target_f, target_e_delta, box_dims)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Training data not found at {filename}")

    print(f"Parsing {filename}...")
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    lines_per_frame = 9 + n_atoms
    total_frames = len(lines) // lines_per_frame
    if limit:
        total_frames = min(total_frames, limit)
    
    raw_data = []
    all_energies = []
    box_dims = torch.tensor([BOX_SIZE, BOX_SIZE, BOX_SIZE], dtype=torch.float32)

    for i in range(total_frames):
        start = i * lines_per_frame
        atom_lines = lines[start + 9 : start + lines_per_frame]
        
        # Parse frame
        data = np.loadtxt(atom_lines)
        data = data[data[:, 0].argsort()]  # Sort by Atom ID
        
        # Create Tensors
        # requires_grad=True is essential for Force calculation via Autograd
        pos = torch.tensor(data[:, 2:5], dtype=torch.float32, requires_grad=True)
        feat = torch.ones((n_atoms, 1), dtype=torch.float32) 
        target_f = torch.tensor(data[:, 5:8], dtype=torch.float32)
        
        e_frame = np.sum(data[:, 8])
        target_e = torch.tensor(e_frame, dtype=torch.float32).view(1)
        
        raw_data.append((pos, feat, target_f, target_e, box_dims))
        all_energies.append(e_frame)

    # Calculate Reference Energy (E_ref) for Normalization
    mean_total_energy = np.mean(all_energies)
    E_ref = mean_total_energy / n_atoms
    print(f"Dataset Stats: Loaded {len(raw_data)} frames.")
    print(f"Normalization: E_ref calculated as {E_ref:.4f} eV/atom")
    
    # Normalize Targets (Train on Delta E)
    normalized_dataset = []
    for (pos, feat, target_f, target_e, box) in raw_data:
        E_delta = target_e - (n_atoms * E_ref)
        normalized_dataset.append((pos, feat, target_f, E_delta, box))
        
    return normalized_dataset

def train():
    dataset = load_dataset(DATA_PATH, n_atoms=108, limit=100)
    
    model = PtGNN()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print("\n--- Starting Training ---")
    
    for epoch in range(EPOCHS):
        epoch_loss_e = 0.0
        epoch_loss_f = 0.0
        
        for (pos, feat, target_f, target_e, box) in dataset:
            optimizer.zero_grad()
            if pos.grad is not None:
                pos.grad.zero_()
            
            # Forward Pass
            pred_e = model(pos, feat, box)
            
            # Compute Forces (Negative Gradient of Energy)
            # create_graph=True allows differentiating through the gradient itself
            grads = torch.autograd.grad(pred_e, pos, create_graph=True, retain_graph=True)[0]
            pred_f = -grads
            
            # Compute Loss
            loss_e = criterion(pred_e, target_e)
            loss_f = criterion(pred_f, target_f)
            total_loss = loss_e + (LAMBDA_F * loss_f)
            
            # Backward Pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss_e += loss_e.item()
            epoch_loss_f += loss_f.item()
            
        if epoch % 10 == 0:
            avg_e = epoch_loss_e / len(dataset)
            avg_f = epoch_loss_f / len(dataset)
            print(f"Epoch {epoch:03d}: Loss E = {avg_e:.4f} | Loss F = {avg_f:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model weights saved to '{MODEL_SAVE_PATH}'.")

if __name__ == "__main__":
    train()
