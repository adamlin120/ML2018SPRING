wget 'https://www.dropbox.com/s/he1jefptkttn1t9/test_mfcc60%2Bd%2Bdd.npy?dl=1' -O ./model/test_mfcc60+d+dd.npy
wget 'https://www.dropbox.com/s/tnx3nruynywn8ud/train_mfcc60%2Bd%2Bdd.npy?dl=1' -O ./model/train_mfcc60+d+dd.npy
wget 'https://www.dropbox.com/s/dslj7qc2hgela9a/test_1d.npy?dl=1' -O ./model/test_1d.npy
wget 'https://www.dropbox.com/s/tqo9y44q3k4xhas/train_1d.npy?dl=1' -O ./model/train_1d.npy
python3 ./src/test.py
