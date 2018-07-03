# Editor.md
## Data Preparation
1. "Download "audio_test.zip" and "audio_train.zip" in https://www.kaggle.com/c/freesound-audio-tagging/data  at the "data" folder
2. Unzup "audio_test.zip" and "audio_train.zip" respectively at FOLDER "audio_test" and "audio_train" in "data folder"

## Training
1. After downloading data, Run the scipt train.sh by command `bash train.sh` and current directory is at "final folder"
2. Run the \src\train_con1d.ipynb in IPython Notebook and run all codes
3. All files required will be at model folder

## Testing
1. Run test.sh by command `bash test.sh` and current directory is at "final folder"
Models are download from Dropbox
2. The prediction file is saved at "final folder" as `prediction_final.csv`

## Library Required
|   library | version   |
| ------------ | ------------ |
|  Python | 3.6.4  |
|  ipython |  6.2.1 |
| Keras  |  2.2.0 |
| tensorboard  | 1.8.0  |
|  tensorflow | 1.8.0  |
|  librosa |  0.6.0 |
| numpy  |  1.14.0 |
|  pandas | 0.22.0  |
|  scikit-learn | 0.19.1  |


