# origin: ../experiments-journal/jk-hpopt3layers.bash but changed LAYERS=2
DATA="dblp-easy"
YEAR=2004
INITIAL_EPOCHS=200
ANNUAL_EPOCHS=200
NLAYERS=2
OUTFILE="results/jk-hpopt.csv"
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5  --rescale_lr 1. --rescale_wd 1."
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $INITIAL_EPOCHS"
HIDDEN=16

set -e

for LR in "0.1" "0