import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- Configuration ---
TEST_FILE = 'data/test_data.xyz'
MODEL_PATH = 'best_model.pth'
PLOT_OUTPUT = 'prediction_parity.png'

# CRITICAL: This must match the E_ref printed during training.
# Update this value based on your specific training run.
E_REF = -5.7359 

# --- Shared Architecture (Must match training) ---
def get_pbc_distances(pos: torch.Tensor, box_dims: torch.Tensor) -> torch.Tensor:
    delta = pos.unsqueeze(0) - pos.unsqueeze(1)
    box = box_dims.view(1, 1, 3)
    delta = delta - box * torch.round(delta / box)
    return torch.sqrt((delta**2).sum(dim=2) + 1e-8)

class PtGNN(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 16):
        super(PtGNN, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, positions: torch.Tensor, features: torch.Tensor, box_dims: torch.Tensor) -> torch.Tensor:
        N = positions.shape[0]
        dists = get_pbc_distances(positions, box_dims)
        
        mu, sigma = 2.77, 0.5
        A = torch.exp(-(dists - mu)**2 / sigma**2)
        mask = 1.0 - torch.eye(N).to(positions.device)
        A = A * mask

        h = self.encoder(features)
        h_aggr = torch.matmul(A, h)
        h_new = self.activation(h_aggr)
        e_atomic = self.decoder(h_new)
        return torch.sum(e_atomic)

# --- Prediction Logic ---
def parse_header(lines: list) -> tuple:
    """Extracts atom count and box dimensions from LAMMPS XYZ header."""
    try:
        n_atoms = int(lines[3])
        
        # Parse Box Bounds (Lines 5-7 in LAMMPS custom dump)
        xlo, xhi = map(float, lines[5].split())
        ylo, yhi = map(float, lines[6].split())
        zlo, zhi = map(float, lines[7].split())
        
        box_dims = torch.tensor([xhi-xlo, yhi-ylo, zhi-zlo], dtype=torch.float32)
        return n_atoms, box_dims
    except (ValueError, IndexError):
        print("Error: Could not parse XYZ header. Ensure LAMMPS 'custom' format.")
        return None, None

def predict():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    # 1. Load Model
    model = PtGNN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

    # 2. Read Data
    print(f"Reading {TEST_FILE}...")
    with open(TEST_FILE, 'r') as f:
        lines = f.readlines()

    n_atoms_header, box_dims = parse_header(lines)
    if n_atoms_header is None: return

    print(f"System detected: {n_atoms_header} atoms.")
    print(f"Box Dimensions: {box_dims.tolist()}")

    lines_per_frame = 9 + n_atoms_header
    total_frames = len(lines) // lines_per_frame
    print(f"Running inference on {total_frames} frames...")

    real_energies = []
    pred_energies = []

    # 3. Inference Loop
    for i in range(total_frames):
        start = i * lines_per_frame
        atom_lines = lines[start + 9 : start + lines_per_frame]
        
        # Parse frame
        data = np.loadtxt(atom_lines)
        current_N = data.shape[0]
        
        # Prepare Inputs
        # Sort by ID to ensure consistent atom ordering
        data = data[data[:, 0].argsort()] 
        pos = torch.tensor(data[:, 2:5], dtype=torch.float32)
        feat = torch.ones((current_N, 1), dtype=torch.float32)
        
        target_e_total = float(np.sum(data[:, 8]))
        
        with torch.no_grad():
            # Predict Delta E
            pred_delta = model(pos, feat, box_dims)
            # Reconstruct Total E (Extensive property)
            pred_total = pred_delta.item() + (current_N * E_REF)
            
        real_energies.append(target_e_total)
        pred_energies.append(pred_total)

    # 4. Metrics & Plotting
    real_energies = np.array(real_energies)
    pred_energies = np.array(pred_energies)
    
    mae = np.mean(np.abs(real_energies - pred_energies))
    mae_per_atom = mae / n_atoms_header
    
    print("\n" + "="*40)
    print(f"RESULTS (E_ref used: {E_REF} eV)")
    print("="*40)
    print(f"MAE Total:    {mae:.4f} eV")
    print(f"MAE Per Atom: {mae_per_atom:.4f} eV/atom")
    
    if mae_per_atom < 0.043:
        print(">> STATUS: Chemical Accuracy Achieved (< 1 kcal/mol)")
    else:
        print(">> STATUS: Above Chemical Accuracy threshold")

    # Parity Plot
    plt.figure(figsize=(6, 6), dpi=100)
    plt.scatter(real_energies, pred_energies, alpha=0.6, color='royalblue', label='Test Frames')
    
    min_val = min(real_energies.min(), pred_energies.min())
    max_val = max(real_energies.max(), pred_energies.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')
    
    plt.xlabel("Ground Truth Energy (eV)")
    plt.ylabel("Predicted Energy (eV)")
    plt.title(f"Model Validation\nMAE: {mae_per_atom:.4f} eV/atom")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT)
    print(f"Plot saved to '{PLOT_OUTPUT}'")

if __name__ == "__main__":
    predict()
