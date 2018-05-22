wget -O 'hw5_model.h5' 'https://www.dropbox.com/s/qv9rlnvqc7zspzq/hw5_model.h5?dl=1'
python3 hw5_test.py -md ./hw5_model.h5 -sub $2 --test_data $1
