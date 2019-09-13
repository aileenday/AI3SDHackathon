import math
from math import log

import joblib
# Plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem as rdkit
# PCA on the fingerprints.
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
# Neural network 
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Grid searching
from sklearn.model_selection import GridSearchCV, train_test_split

training_data = pd.read_excel('/Users/stevenbennett/Documents/AI3SD_Hackathon/Solubility/soldata_trainingset.xls')
training_data.drop(columns=['Substance', 'Temperature', 'assays', 'Ionic Strength (M)', 'Kinetic Solubility (mM)', 'SD of Kinetic Solubility (mM)', 'SMILES', 'Unnamed: 8', 'Unnamed: 9'], inplace=True)
training_data.dropna(axis=0, inplace=True)

test_data = pd.read_excel('/Users/stevenbennett/Documents/AI3SD_Hackathon/Solubility/soldata_prediction_test.xls')
test_data.drop(columns=['SMILES', 'name'], inplace=True)

training_mols = [rdkit.MolFromInchi(mol) for _, mol in training_data['InChI'].iteritems()]
test_mols = [rdkit.MolFromInchi(mol) for _, mol in test_data['InChI'].iteritems()]
all_mols = [rdkit.MolFromInchi(mol) for _, mol in training_data['InChI'].iteritems()]

all_fps = [np.array(rdkit.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype='int') for mol in all_mols]

fps = [np.array(rdkit.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype='int') for mol in training_mols]
y = training_data['S0 (mM)'].values
y_log = np.log(y)

np.mean(y_log)
np.max(y_log)
np.min(y_log)

X_train, X_test, y_train, y_test = train_test_split(fps, y_log, test_size=0.33)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

error = mean_squared_error(y_test, predictions)
math.sqrt(error)
big_training_dataset
big_training_dataset = pd.read_csv('/Users/stevenbennett/Documents/AI3SD_Hackathon/AI3SDHackathon/LargerDataset/combined_dataset_without_pubchem.csv')
big_training_dataset.drop(columns=['Compound_Identifier_raw', 'Compound_Identifier', 'Source', 'Solubility(microgram/mL)', 'Solubility(micromolar)','Notes', 'Recalculated_SMILES'], inplace=True)
big_training_dataset.dropna(axis=0, inplace=True)
big_dataset_mol = [rdkit.MolFromSmiles(mol) for _, mol in big_training_dataset['SMILES'].iteritems()]
for idx, mol in enumerate(big_dataset_mol):
    if mol == None:
        big_training_dataset.drop(labels=idx, inplace=True)
        big_dataset_mol.remove(mol)

fps_large = [np.array(rdkit.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype='int') for mol in big_dataset_mol]

pca = PCA(n_components=2)
pca.fit(fps_large)
res = pca.fit_transform(fps_large)


logS = pd.to_numeric(big_training_dataset['LogS(M)'], errors='coerce')

fig = plt.figure()
ax = fig.add_subplot()
scatter = ax.scatter(res[:,0], res[:,1], c=colors, s=0.7)
fig.colorbar(scatter, ax=ax)

# Big data ML
X_train, X_test, y_train, y_test = train_test_split(fps_large, logS, test_size=0.33)

clf = RandomForestRegressor(n_estimators=100)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
predictions

error = mean_absolute_error(y_test, predictions)
error = mean_squared_error(y_test, predictions)
rmse = math.sqrt(error)

# Test original dataset
original_predictions = clf.predict(all_fps)
training_data['S0 (mM)'].values
logS_original = np.log10(training_data['S0 (mM)'].values*(10**-6))
error_for_original = math.sqrt(mean_squared_error(original_predictions, logS_original))

# Checking if molecules are the same.
for idx, i in enumerate(big_dataset_mol):
    mol_to_check = rdkit.MolToInchiKey(i)
    logS_1 = logS.iloc[idx]
    for idx_2, j in enumerate(all_mols):
        logS_2 = logS_original[idx_2]
        converted = 10^(logS_2)
        key = rdkit.MolToInchiKey(j)
        mol = j
        if mol_to_check == key:
            print(f'The molecule is: {rdkit.MolToSmiles(mol)} \n \
                    The values of logS are Big: {logS_1} Small: {logS_2}')



krr = KernelRidge(alpha=1.0, kernel='rbf')
krr.fit(X_train, y_train)
predictions = krr.predict(X_test)
original_predictions = krr.predict(all_fps)
r2_score(y_test, predictions)

r2_score(logS_original, original_predictions)

# Plotting the fit
model_correlation = plt.figure()
ax = model_correlation.add_subplot()
scatter = ax.scatter(original_predictions, logS_original)


# Grid search for searching model.
krr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})

joblib.dump(krr, 'krr_model.joblib')
