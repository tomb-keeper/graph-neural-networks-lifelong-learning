set -e 

OUTDIR="data/label_rate_ablation"
BACKEND="dgl"

# Preprocess datasets 'dblp-easy', 'dblp-hard', and 'pharmabio' assuming they reside within a ./data/<dataset> directory relat