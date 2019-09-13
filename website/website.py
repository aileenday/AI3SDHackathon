from flask import Flask
from rdkit.Chem import AllChem as rdkit
import joblib
from flask import render_template
from flask import request
import numpy as np

app = Flask(__name__)

krr = joblib.load('/Users/stevenbennett/Documents/AI3SD_Hackathon/AI3SDHackathon/website/krr_model.joblib')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    mol = request.form['SMILES']
    fp = rdkit.GetMorganFingerprintAsBitVect(rdkit.MolFromSmiles(mol),2, nBits=1024)
    fp_for_pred = np.array(fp)
    val = krr.predict(fp_for_pred.reshape(1, -1))
    print(mol)
    return render_template('site.html', val=val[0], SMILES=mol)

@app.route('/')
def render_homepage():
    return render_template('site.html', val=None)


