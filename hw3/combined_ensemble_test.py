import hw3_utilities as my
from keras.models import load_model
import sys

#DATA_PATH = "~/Desktop/machine_learning/hw/hw3/dataset/test.csv"
DATA_PATH = sys.argv[1]
version_name = 'reg23v2-reg11-reg5-res2-incept1'
MODEL_PATH_Ens = './model_ensembled_'+version_name+'.h5'
#PREDICTION_NAME = './model_ensembled_'+version_name+'.csv'
PREDICTION_NAME = sys.argv[2]

# load ensemble model
modelEns = load_model(MODEL_PATH_Ens)
modelEns.summary()
# loading test data
X_test = my.load_test_data(DATA_PATH)
# predict and change it to class number
pred = modelEns.predict(X_test)
pred_class = my.pred2pred_class(pred)
# create csv file
my.submission(pred_class, PREDICTION_NAME)
