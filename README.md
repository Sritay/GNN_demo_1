# Physics-Informed GNN for Atomic Potentials (Technical Demo)

> **DISCLAIMER:** This repository is a **technical proof-of-concept** developed to demonstrate the architectural implementation of Graph Neural Networks (GNNs) from first principles. It is designed to showcase rigorous understanding of Deep Learning and Physics-ML concepts (autograd, conservative fields, differentiable geometry) rather than to serve as a production-grade tool.

## 🧪 Project Overview

This project implements a custom **Graph Neural Network (GNN)** to predict the Potential Energy Surface (PES) of a Platinum crystal with defects.

Unlike standard approaches that rely on high-level libraries (like PyG or DGL), this implementation builds the **Message Passing** and **Spatial Convolution** logic using raw PyTorch tensors.

### Why "From Scratch"?
In the fast-moving field of AI for Science, it is easy to import a pre-made model. This project takes the opposite approach to demonstrate architectural control:
1.  **Low-Level Implementation:** We manually implement the adjacency matrix construction ($A$) and feature aggregation ($A \cdot H \cdot W$). This makes the computational graph explicit.
2.  **Physics-First Design:** We do not treat atoms as generic nodes. We explicitly model physical constraints, such as the **Minimum Image Convention** for Periodic Boundary Conditions (PBCs).
3.  **Conservative Force Field:** We do not predict forces as a separate output. We predict scalar Energy and utilize PyTorch's `autograd` engine to derive Forces analytically ($F = -\nabla E$), guaranteeing a curl-free vector field.

### Why EAM Data?
The model is trained on synthetic data generated via LAMMPS using an Embedded Atom Method (EAM) potential.
* **Choice:** EAM was chosen over DFT for this demo to allow for rapid iteration and high-volume data generation (1,000+ frames) during the architectural debugging phase.
* **Implication:** The model serves as a surrogate for the EAM potential. The pipeline is designed so that the training data can be swapped for DFT/Ab-Initio data without changing the model architecture.

---

## 🚀 Key Features

* **Custom GNN Layer:** A lightweight spatial convolution with Gaussian-expanded edge features and SiLU activation.
* **Differentiable PBCs:** A custom `get_pbc_distances` function that handles boundary wrapping within the compute graph, allowing gradients to flow through periodic boundaries.
* **Hybrid Loss Function:** Trains on a weighted combination of Energy (global scalar) and Forces (local vectors).
    $$ \mathcal{L} = \text{MSE}(E_{pred}, E_{ref}) + \lambda \cdot \text{MSE}(-\nabla E_{pred}, F_{ref}) $$
* **Extensivity:** The model learns local environments, allowing it to scale from 100 atoms (training) to 1,000+ atoms (inference) without retraining.

---

## 📊 Results

The model was trained on small **108-atom** supercells and tested on an unseen **864-atom** system.

| Metric | Result | Target (Chemical Accuracy) |
| :--- | :--- | :--- |
| **MAE (Energy)** | **0.7 meV/atom** | 43 meV/atom (1 kcal/mol) |
| **Transferability** | ✅ Perfect | Model generalized to 8x larger box |

### Parity Plot (Unseen Test Data)
![Parity Plot](prediction_parity.png)
*The tight diagonal correlation demonstrates that the model has learned the underlying physics of the Potential Energy Surface, distinct from simple coordinate memorization.*

---

## 📂 Repository Structure

```text
.
├── data/
│   ├── training_data.xyz    # Generated via LAMMPS (Melt-Quench-Rattle)
│   ├── test_data.xyz        # Unseen trajectory (different random seed)
│   └── Pt_u3.eam            # EAM potential file
├── train_gnn.py             # Main training loop (PyTorch)
├── predict.py               # Inference & Validation script
├── runscript_gnn.slurm      # HPC submission script
└── best_model.pth           # Saved model weights
```

---

## 💻 Usage

### 1. Prerequisites
* Python 3.8+
* PyTorch
* NumPy
* Matplotlib

### 2. Training
The training script parses the LAMMPS trajectory, normalizes the energy against a reference ($E_{ref}$), and optimizes the network using Adam.

```bash
python train_gnn.py
```
* **Input:** `data/training_data.xyz`
* **Output:** `best_model.pth`

### 3. Testing / Inference
To validate the model on the held-out dataset and generate the parity plot:

```bash
python predict.py
```
* **Input:** `data/test_data.xyz` & `best_model.pth`
* **Output:** `prediction_parity.png` & Metrics printed to console.

---

## 🔧 Technical Details (The Math)

The core operation is a spatial convolution defined as:

$$ h_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} W^{(l)} \cdot h_j^{(l)} \cdot e^{-\frac{(r_{ij} - \mu)^2}{\sigma^2}} \right) $$

Where:
* $h_i$: Feature vector of atom $i$
* $r_{ij}$: Distance under Periodic Boundary Conditions
* $\mathcal{N}(i)$: Neighbors within cutoff
* $\sigma(\cdot)$: SiLU Activation

---

## 📜 License
MIT License. Feel free to use this code for educational purposes or as a template for custom MLFF development.
