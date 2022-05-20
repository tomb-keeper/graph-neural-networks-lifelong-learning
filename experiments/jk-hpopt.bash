# origin: ../experiments-journal/jk-hpopt3layers.bash but changed LAYERS=2
DATA="dblp-easy"
YEAR=2004
INITIAL_EPOCHS=200
ANNUAL_EPOCHS=200
NLAYERS=2
OUTFILE="results/jk-hpopt.csv"
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5 