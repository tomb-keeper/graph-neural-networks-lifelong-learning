DATA="dblp-hard"
YEAR=2004
ANNUAL_EPOCHS=200
NLAYERS=1
BACKEND="dgl"
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --rescale_lr 1.0 --rescale_wd 1. --annual_epochs $ANNUAL_EPOCHS --backend $BACKEND"
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $ANNUAL_EPOCHS"
OUTFILE="results/open_DOC_CW_risk-reduction_tau75.csv"

# Exit on error
set -e

HPARAMS=(
	"--history 1 --start cold --lr 0.005"
	"--history 1 --start warm --lr 0.0005"
	"--history 3 --start cold --lr 0.005"
	"--history 3 --start warm --lr 0.001"
	"--h