# Appendix: reproducibility material

Paper: *Total Dominating Energy and Signless Total Dominating Laplacian Energy of Hesitancy Fuzzy Graphs with Application to Supply Chain Network Analysis* (Siddharthan R, Jenitha G, Kungumaraj E).

## Contents
| Folder / file | What it is |
|---|---|
| `code/` | the seven Python scripts, the helper `tds_lib.py`, `supp_independent_check.py` and `run_all.sh` |
| `expected_output/` | the printed output (`out_*.txt`) and result files (`results_*.json`) from a clean run, to compare with your own run |
| `figures/` | the figures produced by the scripts (PDF and PNG) |
| `requirements.txt` | Python packages (tested with Python 3.13, numpy 2.5, scipy 1.18, mpmath 1.4, matplotlib 3.11, networkx 3.7) |

## How to run
```
pip install -r requirements.txt
cd code
sh run_all.sh        # or run the scripts one by one, in the order below
```
Every script prints its results and stops with an error if an asserted identity or inequality fails. Random experiments use fixed seeds, so the numbers are reproducible; runtimes depend on the machine.

## Which script reproduces what
| Script | Reproduces (paper section) |
|---|---|
| `supp_example.py` | Numerical Illustration: HFG conditions, strong arcs, minimum total dominating sets, E, LE, SLE, TDE, STDLE, bounds, properties of TDE, values for the three minimum sets (tables of Sections 4 and 5.1) |
| `supp_optionB.py` | Numerical Illustration: high-precision (mpmath) recomputation of every eigenvalue result of the example |
| `supp_identities.py` | TDE section: trace, interlacing, bounds and equality cases on the example and on 30,000 random weighted graphs |
| `supp_tds.py` | Computing the minimum total dominating set: exhaustive search, integer program, greedy heuristic, the 8-vertex counterexample |
| `supp_experiments.py` | Strong-arc test against the literal definition; exact versus heuristic experiments (Table 14, Figure 5) |
| `supp_stdle.py` | STDLE section: structure of Q_T, centring constants, bounds, sufficient condition, the search over about 55,000 cases and counterexamples (Tables 6 and 9, Figures 2-4) |
| `supp_models.py` | Effect of vertices, edges and hesitancy (Tables 10-12) and the comparison with fuzzy and intuitionistic fuzzy graphs (Table 13) |

Table and figure numbers refer to the version of the paper in which they were generated; if the journal layout renumbers them, use the section titles.

## Independent check
`supp_independent_check.py` recomputes, with separate code, every number printed for the example and for the named counterexamples and looks each one up in `main_SOIC.tex` (place the file next to it, or edit the path). The only entry it reports is `-0.0000`, a rounding artefact of an eigenvalue that is exactly 0.

## Timings
Runtimes (Figure 5 and the sentence about it) depend on the machine; all other numbers are deterministic.

## Note on the data
The vertex and edge triples of the supply chain example are illustrative inputs, as stated in the paper; no external data set is used.
