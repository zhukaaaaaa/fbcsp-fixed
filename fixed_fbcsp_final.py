#!/usr/bin/env python
# coding: utf-8

# In[2]:


import mne
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
from mne.decoding import CSP
import os
from joblib import dump

DATA_DIR = r'C:\Users\user\Desktop\BCI project\BCICIV_2a_gdf (1)'
SUBJECTS = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']
FREQ_BANDS = [[4,8],[8,12],[10,14],[12,16],[16,20],[20,24],[22,26],[26,30]]
N_SHUFFLE = 10
T_MIN, T_MAX = 0.5, 3.0
EVENTS = {'Left':769, 'Right':770, 'Foot':771, 'Tongue':772}

results = []
os.makedirs('results_final', exist_ok=True)

for subj in SUBJECTS:
    path = f'{DATA_DIR}/{subj}.gdf'
    if not os.path.exists(path): 
        print(f"SKIP: {path}")
        continue
    print(f"PROCESSING: {subj}")

    raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    events, event_id = mne.events_from_annotations(raw)
    event_id_4 = {k: event_id[str(v)] for k, v in EVENTS.items()}
    raw.filter(8, 28, fir_design='firwin')
    raw.set_eeg_reference('average')

    
    ica = mne.preprocessing.ICA(n_components=15, random_state=42)
    ica.fit(raw.copy())
    ica.exclude = [0]  # Исключаем первую компоненту (артефакт)
    raw_clean = ica.apply(raw.copy()).pick('eeg')
    

    epochs = mne.Epochs(raw_clean, events, event_id_4, tmin=T_MIN, tmax=T_MAX,
                        baseline=None, preload=True, verbose=False)
    epochs.drop_bad(reject={'eeg': 150e-6})
    X, y = epochs.get_data(), epochs.events[:, 2]

 
    features = []
    for low, high in FREQ_BANDS:
        eps_f = epochs.copy().filter(low, high)
        X_f = eps_f.get_data()
        for cls in np.unique(y):
            y_ovr = (y == cls).astype(int)
            csp = CSP(n_components=8, reg='ledoit_wolf', transform_into='average_power')
            csp.fit(X_f, y_ovr)
            features.append(csp.transform(X_f))
    X_fbcsp = np.concatenate(features, axis=1)


    pipeline = Pipeline([
        ('select', SelectKBest(mutual_info_classif)),
        ('scale', StandardScaler()),
        ('clf', OneVsRestClassifier(LinearSVC(dual=False, max_iter=10000)))
    ])

    param_grid = {'select__k': [16, 32, 64]}
    outer_cv = StratifiedShuffleSplit(n_splits=N_SHUFFLE, test_size=0.2, random_state=42)
    inner_cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

    grid = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='accuracy')
    scores = []
    kappas = []
    for train_idx, test_idx in outer_cv.split(X_fbcsp, y):
        grid.fit(X_fbcsp[train_idx], y[train_idx])
        pred = grid.predict(X_fbcsp[test_idx])
        scores.append((pred == y[test_idx]).mean())
        kappas.append(cohen_kappa_score(y[test_idx], pred))

    acc = np.mean(scores)
    kappa = np.mean(kappas)
    print(f" -> Acc: {acc:.3f} | Kappa: {kappa:.3f}")
    results.append({'Subject': subj, 'Accuracy': acc, 'Kappa': kappa})

    grid.best_estimator_.fit(X_fbcsp, y)
    dump(grid.best_estimator_, f'results_final/model_{subj}.pkl')

df = pd.DataFrame(results)
df.to_csv('results_final/all_subjects_final.csv', index=False)
print(f"\nFINAL: Avg Acc: {df['Accuracy'].mean():.3f} | Kappa: {df['Kappa'].mean():.3f}")


# In[ ]:




