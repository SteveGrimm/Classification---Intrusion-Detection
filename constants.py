# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

WD = '/content/Drive'
drive.mount(WD)

PATH = 'intrusion_detection_notonehot.csv'
PREDS_DEST = 'predictions.csv'
ESTIMATOR = Pipeline(steps = [('scaler', StandardScaler()), 
                              ('clf', LogisticRegression(penalty = 'l2', 
                                                        class_weight = 'balanced', 
                                                        solver = 'liblinear', 
                                                        random_state=42,
                                                        C = 1e-06))])
