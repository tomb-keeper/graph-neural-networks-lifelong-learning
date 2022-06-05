# origin: ../experiments-journal/saint-hpopt3layers.bash but changed LAYERS=2
DATA="dblp-easy"
YEAR=2004
INITIAL_EPOCHS=200
ANNUAL_EPOCHS=200
NLAYERS=2
OUTFILE="results/saint-hpopt.csv"
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5  --rescale_lr 1. --rescale_wd 1. --sampling rw --saint_coverage 0"
PRETRAIN_ARG