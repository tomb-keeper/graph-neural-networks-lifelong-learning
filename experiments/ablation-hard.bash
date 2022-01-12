DATA="dblp-hard"
YEAR=2004
HISTORY=3
NLAYERS=1 # 2 graph conv layers
ARGS="--start warm --n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --history $HISTORY --rescale_lr 1. --rescale_wd 1."
DATA_ARGS="--data "$DATA" --t_start $YEAR"
STATIC_MODEL_ARGS="--initial_epochs 400 --annual_epochs 0"
UPTRAIN_MODEL_ARGS="--initial_epochs 0 --annual_epochs 200"
OUTFIL