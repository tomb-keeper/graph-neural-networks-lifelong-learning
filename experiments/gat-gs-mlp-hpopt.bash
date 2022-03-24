DATA="dblp-easy"
YEAR=2004
INITIAL_EPOCHS=0
ANNUAL_EPOCHS=200
NLAYERS=1
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --rescale_lr 1.0 --rescale_wd 1. --annual_epochs $ANNUAL_EPOCHS"
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $INITI