#!/usr/bin/bash 
python3 analysis/compute_tdiff_dist.py --max-hops 1 --save tdiff/12v-1hop data/dblp-easy
python3 analysis/compute_tdiff_dist.py --max-hops 2 --save tdiff/12v-2hop data/dblp-easy
python3 analysis/compute_tdiff_dist.py --max-hops 3 --save tdiff/12v-3hop data/dblp-easy
python3 analysis/compute_tdiff_dist.py --