DATA="dblp-hard"
YEAR=2004
INITIAL_EPOCHS=200
ANNUAL_EPOCHS=200
NLAYERS=2
OUTFILE="results/saint-hard.csv"
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5  --rescale_lr 1. --rescale_wd 1. --n_hidden 16 --sampling rw --batch_size 0.5 --saint_coverage 0"
PRETRAIN_ARGS="--t_start $YEA