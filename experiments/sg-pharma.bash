DATA="pharmabio"
YEAR=1999
INITIAL_EPOCHS=200
ANNUAL_EPOCHS=200
NLAYERS=2 # K Hops in SG Net
OUTFILE="results/sg-pharma.csv"
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5  --rescale_lr 1. --rescale_wd 1."
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $INITIAL_EPOCHS"
hparams=("--history 1 --start cold --lr 0.005"
            "--history 4 --start cold --lr 0.01"
            "--history 8 --start cold --lr 0.01"
            "--history 21 --start cold --lr 0.