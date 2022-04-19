DATA="pharmabio"
YEAR=1999
INITIAL_EPOCHS=0
ANNUAL_EPOCHS=200
NLAYERS=1
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --rescale_lr 1.0 --rescale_wd 1. --annual_epochs $ANNUAL_EPOCHS"
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $INITIAL_EPOCHS"
OUTFILE="results/gat-gs-mlp-pharma.csv"

for SEED in 1 2 3 4 5 6 7 8 9 10; do
  # HISTORY 1
  python3 run_experiment.py --seed 