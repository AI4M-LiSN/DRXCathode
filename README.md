# CE Workflow and Random Structure Generation for the polyanion-induced regulation of oxygen redox in disordered rocksalt cathodes

**DOI License: MIT**

##Repository Structure

├── README.md
├── CE-model/                          # Cluster expansion model construction
│   ├── get_ce.py                      # CE fitting workflow 
│   ├── str_energy/                    # Input data for CE training
│   │   ├── LMO_pri.vasp               # Primitive structure for defining cluster space
│   │   ├── P0L0.vasp                  # Example structure for P0L0 composition
│   │   ├── P0L0_energy                # Energy file (structure ID + energy per f.u.)
│   │   ├── P0L1.vasp / P0L1_energy
│   │   └── ...                        # Other compositions (P1L1, P2L3, etc.)
│   └── mixing_energy.ce               # Trained CE model (ARD method, output)
│
├── example/                           # Workflow examples for a selected composition
│   └── P27/                           # P content = 27 atoms per supercell
│       └── Mn846/                     # Mn content = 846 atoms per supercell
│           ├── random_666/            # Step 1: Random structure generation
│           │   ├── random_structure.py    # Script for random shuffling (MC_sweep)
│           │   ├── LMO666.vasp            # Pristine 6×6×6 supercell
│           │   ├── Random_0_666.vasp      # Generated random structure #0
│           │   ├── Random_1_666.vasp      # ...
│           │   └── ...
│           │
│           ├── T1000/                 # Step 2: Canonical MC at 1000 K (CEMC)
│           │   ├── CEMC.py                # MC simulation using CE model
│           │   ├── mixing_energy.ce       # CE model (copied from CE-model/)
│           │   ├── MC_0_666.vasp          # Optimised structure #0
│           │   ├── MC_1_666.vasp          # ...
│           │   └── ...
│           │
│           └── Embed_structure_3/     # Step 3: Embedding & oxygen connectivity analysis
│               ├── embeding_structure.py  # Script to embed supercell from CEMC structure
│               ├── supercell_3_0.vasp     # Embedded supercell #0
│               ├── supercell_3_1.vasp     # ...
│               └── ...
│               └── R30/                # Sphere radius = 30 Å
│                   ├── analyze_oxygen_network_in_sphere.py   # O connectivity analysis
│                   ├── Empty_R30_0.vasp         
│                   ├── Replace_R30_0.vasp       
│                   ├── Empty_R30_1.vasp
│                   ├── Replace_R30_1.vasp
│                   └── ...


Methodology Overview

Step 1: Cluster Expansion (CE) Model Construction

The CE model is constructed using the ICET package with a primitive structure (LMO_pri.vasp) containing 14 atoms. The cluster space is defined with:
· Cutoffs: 7.0 Å (pairs), 4.0 Å (triplets), 4.0 Å (quadruplets)
· Chemical species:
  · Tetrahedral sites: ['Li', 'P']
  · Octahedral sites: ['Na', 'Mn', 'Ni'] — Note that Na is used as a proxy for vacancies (i.e., unoccupied sites) within the cluster expansion formalism.
  · Oxygen sites: ['O']
Training data are collected from multiple chemical compositions (P0L0, P0L1, ..., P2L5), each containing structures with DFT-calculated energies. 
ARD (Automatic Relevance Determination): Automatically selects relevant clusters, preventing overfitting (primary model)

Step 2: Random Structure Generation

For a given composition (e.g., P27/Mn846), a pristine 6×6×6 supercell (LMO666.vasp, 1296 atoms) is read. The atoms are divided into:
· Tetrahedral sites (first 432 atoms): occupied by Li, Mn, and P
· Octahedral sites (last 864 atoms): occupied by Na, Mn, and P

The random_structure.py script performs a MC_sweep operation:
1. Counts the number of each species in both site types
2. Randomly shuffles the indices within each site type
3. Reassigns species according to the shuffled order

This process generates 100 random configurations (Random_*_666.vasp) for statistical sampling.

Step 3: Canonical Monte Carlo at 1000 K

Using the trained CE model (mixing_energy.ce) as the energy calculator, canonical ensemble simulations are performed with the mchammer package:

· Temperature: 1000 K (isothermal annealing)
· Steps: 1,000,000 Monte Carlo steps per simulation
· Algorithm: Metropolis–Hastings

For each random structure, the simulation produces a thermodynamically relaxed configuration (MC_*_666.vasp). This step captures the energy‑lowering structural rearrangements driven by the CE potential.

Step 4: Sphere Embedding and Oxygen Connectivity Analysis

A spherical region (radius = 30 Å) is embedded into each MC-optimised supercell:

1. Oxygen atoms are labelled as Cl based on Mn coordination (0-2 Mn neighbours), representing redox-active O species, while P-adjacent O atoms remain unmodified. 
2. A redox-active O-Mn connectivity graph is then constructed; Cl atoms belonging to the same connected component as any surface-bound Cl (within 1 Å of the sphere surface) are replaced by F, which explicitly identifies surface-connected redox-active O. 
3. This workflow enables statistical analysis of percolation thresholds and surface-connected oxygen networks across 100 MC sweeps.


Requirements

· Python 3.8+
· Core libraries: ase, icet
· Numerical: numpy, networkx


License

This project is licensed under the MIT License - see the LICENSE file for details.


Author

Zhefeng Chen - chenzhefeng@pku.edu.cn
