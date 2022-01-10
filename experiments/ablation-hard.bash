DATA="dblp-hard"
YEAR=2004
HISTORY=3
NLAYERS=1 # 2 graph conv layers
ARGS="--start warm --n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --history $HISTORY --rescale_lr 1. --rescale_wd 1."
DATA_ARGS="--data "$DATA" --t_st