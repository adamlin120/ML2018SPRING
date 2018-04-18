wget -O 'model_ensembled_reg23v2-reg11-reg5-res2-incept1.h5' 'https://www.dropbox.com/s/l2yb6eqqqf06bqx/model_ensembled_reg23v2-reg11-reg5-res2-incept1.h5?dl=1'
python3 combined_ensemble_test.py $1 $2 $3
