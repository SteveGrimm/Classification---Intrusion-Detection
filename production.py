# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from constants import ESTIMATOR, PREDS_DEST, PATH

def load_data(path):
    return pd.read_csv(path)

def freq_target_table(column):
  table=pd.crosstab(data[column],data[target])
  return table.div(table.sum(1).astype(float), axis=0)

def salar_sales_binarizer(data):
    data['flag'] = np.where(data['flag'].isin(freq_target_table('flag')[freq_target_table('flag')[1]==0].index.tolist()),
                        'NoHarm',data['flag'])
    data['service'] = np.where(data['service'].isin(freq_target_table('service')[freq_target_table('service')[1]==0].index.tolist()),
                        'NoHarm',data['service'])
    data['service'] = np.where(data['service'].isin(freq_target_table('service')[freq_target_table('service')[1]==1].index.tolist()),
                        'Harm',data['service'])

def data_splitter(data):
    cat_var = data[['protocol_type','service','flag']]
    target = np.where(data['outlier@{no,yes}'] == 'yes', 1, 0)
    num_var = data.drop(columns= cat_var + ['num_outbounds_cmds','land','Unnamed: 0', 'id', target], axis=1)
    return cat_var, num_var, target

def predict_on_data(ESTIMATOR, X, y):
    predictions = ESTIMATOR.fit(X, y).predict(X)
    return predictions

def write_predictions_csv(y_pred):
    pd.DataFrame(y_pred).to_csv(PREDS_DEST)

def main():
    X_train, y_train = data_splitter(salar_sales_binarizer(load_data(PATH)))
    write_predictions_csv(predict_on_data(ESTIMATOR, X_train, y_train))

if __name__ == "__main__":
    main()
