DATA="dblp-easy"
YEAR=2004
INITIAL_EPOCHS=0
ANNUAL_EPOCHS=200
NLAYERS=1
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --rescale_lr 1.0 --rescale_wd 1. --annual_epochs $ANNUAL_EPOCHS --backend dgl"
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $INITIAL_EPOCHS"
OUTFILE="results/gat-gs-mlp-easy.csv"

for SEED in 1 2 3 4 5 6 7 8 9 10; do
  # HISTORY 1
  python3 run_experiment_new.py --seed "$SEED" --model mlp --n_hidden 64 --start cold --lr "0.001" --history 1 $ARGS $PRETRAIN_ARGS --dataset "$DATA" --save "$OUTFILE"
  python3 run_experiment_new.py --seed "$SEED" --mode