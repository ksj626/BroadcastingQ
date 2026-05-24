# Masked DNA Discovery Environment

This application simulates the discovery of valid 12-mer DNA barcode sequences from a known database (53 sequences from QIIME 2 and 10x Genomics).
The agent starts with a sequence of 12 unknown masks (`MMMMMMMMMMMM`).
At each step, it chooses an index (0-11) and a base ('A', 'C', 'G', 'T') to reveal/assign.
The environment enforces **Dead-end Pruning**:
- If the agent modifies an already assigned index, the episode terminates with a penalty (-0.1).
- If the current sequence state is no longer compatible with any of the 53 target sequences in the database, the episode terminates early with a penalty (-0.1).
- If the agent successfully assigns all 12 bases and matches a target sequence, it receives a reward (+1.0).

This environment highlights the efficiency of algorithms like **SBQ (Spatial Broadcasting Q-learning)** over standard Tabular Q-learning. Dead-end states provide strong negative signals, and SBQ broadcasts these signals to similar sequence states, radically accelerating the search through the exponentially large state space.
