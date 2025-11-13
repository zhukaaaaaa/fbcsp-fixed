
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
from mne.preprocessing import ICA
import warnings
import os
from joblib import dump
warnings.filterwarnings("ignore")
plt.ioff()

DATA_DIR = '.'
SUBJECTS = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']
FREQ_BANDS = [[4,8],[8,12],[10,14],[12,16],[16,20],[20,24],[22,26],[26,30]]
N_CSP = 8
K_BEST = 32
N_SHUFFLE = 15
T_MIN, T_MAX = 0.5, 3.0
EVENTS = {'Left':769, 'Right':770, 'Foot':771, 'Tongue':772}
results = []
os.makedirs('results', exist_ok=True)
print("FBCSP PROCESSING FOR ALL 9 SUBJECTS...\n")

for subj in SUBJECTS:
    path = f'{DATA_DIR}/{subj}.gdf'
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        continue
    print(f"PROCESSING: {subj}")
    
    raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    events, event_id = mne.events_from_annotations(raw)
    event_id_4 = {k: event_id[str(v)] for k, v in EVENTS.items()}
    
    raw.filter(8, 28, fir_design='firwin')
    raw.set_eeg_reference('average')
    
    ica = ICA(n_components=15, method='fastica', random_state=42)
    ica.fit(raw.copy())
    ica.exclude = []
    raw_clean = ica.apply(raw.copy()).pick('eeg')
    
    epochs = mne.Epochs(raw_clean, events, event_id_4, tmin=T_MIN, tmax=T_MAX,
                        baseline=None, preload=True, verbose=False)
    epochs.drop_bad(reject={'eeg': 150e-6})
    X, y = epochs.get_data(), epochs.events[:, 2]
    
    features = []
    for low, high in FREQ_BANDS:
        raw_f = raw_clean.copy().filter(low, high)
        eps_f = mne.Epochs(raw_f, events, event_id_4, tmin=T_MIN, tmax=T_MAX,
                           baseline=None, preload=True, verbose=False)
        X_f = eps_f.get_data()
        for cls in np.unique(y):
            y_ovr = (y == cls).astype(int)
            csp = CSP(n_components=N_CSP, reg='ledoit_wolf', transform_into='average_power')
            csp.fit(X_f, y_ovr)
            features.append(csp.transform(X_f))
    X_fbcsp = np.concatenate(features, axis=1)
    
    selector = SelectKBest(mutual_info_classif, k=K_BEST)
    X_sel = selector.fit_transform(X_fbcsp, y)
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_sel)
    
    clf = OneVsRestClassifier(LinearSVC(dual=False, max_iter=10000))
    cv = ShuffleSplit(n_splits=N_SHUFFLE, test_size=0.2, random_state=42)
    scores = cross_val_score(clf, X_final, y, cv=cv, n_jobs=-1)
    acc = scores.mean()
    kappa = acc * 4 - 1
    print(f" -> Acc: {acc:.3f} | Kappa: {kappa:.3f}")
    results.append({'Subject': subj, 'Accuracy': acc, 'Kappa': kappa})
    clf.fit(X_final, y)
    dump(clf, f'results/model_{subj}.pkl')
    
    plt.figure(figsize=(6,4))
    plt.bar(range(1, N_SHUFFLE+1), scores, color='#36A2EB', alpha=0.7)
    plt.axhline(acc, color='r', linestyle='--', label=f'Mean: {acc:.3f}')
    plt.title(f'{subj} | Accuracy: {acc:.3f}')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/acc_{subj}.png', dpi=200)
    plt.close()

df = pd.DataFrame(results)
df.to_csv('results/all_subjects.csv', index=False)
mean_kappa = df['Kappa'].mean()
mean_acc = df['Accuracy'].mean()
print(f"\nDONE!")
print(f"Average Accuracy: {mean_acc:.3f}")
print(f"Average Kappa: {mean_kappa:.3f}")
print(f"-> results/all_subjects.csv")
print(f"-> results/acc_*.png")
print("READY")
