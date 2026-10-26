#!/bin/sh
# Runs every script in the order used in the paper. Each script stops with an error if an asserted identity fails.
# Total time: about 4-5 minutes on a laptop. Figures are written to ./figs, results to ./results_*.json.
set -e
for f in supp_example supp_optionB supp_identities supp_tds supp_experiments supp_stdle supp_models; do
  echo "### $f"
  python3 $f.py > out_$f.txt
  tail -1 out_$f.txt
done
echo "All scripts finished."
