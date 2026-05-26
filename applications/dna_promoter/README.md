# DNA Promoter Application

This application models a strict sparse-reward search over 6-base DNA sequences.
The default targets are Pribnow-like promoter motifs:

```text
TATAAT, TATAAA, TATATT, TAAAAT
```

Observations are integer-encoded bases in a `MultiDiscreteSpace([4, 4, 4, 4, 4, 4])`.
The action space has 18 point mutations: 6 positions times 3 possible base changes.
Only reaching one of the configured target sequences gives reward `+1`; all other steps give `0`.

Run examples:

```bash
python main.py --config applications/dna_promoter/config_qlearning.yaml
python main.py --config applications/dna_promoter/config_sarsa.yaml
python main.py --config applications/dna_promoter/config_dqn.yaml
```
