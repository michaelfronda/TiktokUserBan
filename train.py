import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import resample
import pickle

# Read Data
df = pd.read_csv('./data/final_data.csv')

# Upsample
minority = df[df['author_ban_status_banned'] == 1]
majority = df[df['author_ban_status_banned'] == 0]
upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
df_upsampled = pd.concat([majority, upsampled]).reset_index(drop=True)

# Split and Set Target: 80/20 
X = df_upsampled.drop(columns = ['author_ban_status_banned']) 
y = df_upsampled['author_ban_status_banned']


X_train, X_test, y_train, y_test = train_test_split(X, # Features; X
                                                    y, # Label/Target; y
                                                    test_size = 0.20,
                                                    random_state = 42) 

train_dicts = X_train.to_dict(orient='records')
dv = DictVectorizer(sparse=True) # pickle
X_train = dv.fit_transform(train_dicts)

val_dicts = X_test.to_dict(orient='records')
X_test = dv.transform(val_dicts)

# Init and Train RF 
rand_for = RandomForestClassifier(max_depth=10, max_features=1, min_samples_leaf=2,
                       n_estimators=25, random_state=42) # chosen from GV search from EDA
rand_for.fit(X_train, y_train)

# Export model using pickle (Only if satisfied with the evaluation)
model_filename = 'rf_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(rand_for, file)

# Export vectorizer
dv_filename = 'dv.pkl'
with open(dv_filename, 'wb') as file:
    pickle.dump(dv, file)