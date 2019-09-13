import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import datasets
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import math
import matplotlib.pyplot as plt

def makefingerprintsarrayfromsmilesdfcolumn(inputdf,inputcolumnname):
	output=np.zeros((len(inputdf.index), 128))
	for index, row in inputdf.iterrows():
		thissmiles=str(row[inputcolumnname])
		thisms = Chem.MolFromSmiles(thissmiles)
		fps = FingerprintMols.FingerprintMol(thisms)
		fpslist=list(DataStructs.BitVectToText(fps))
		if index >= len(output): 
			np.append(output, fpslist)
		else:
			for i in range(0, 128):
				output[index,i]=fpslist[i]
	return output

trainingdf = pd.read_excel('data/soldataswap.xls',sheet_name='100 data')
trainingdf=trainingdf.drop(["Temperature","SD of S0 (mM)","SD of Kinetic Solubility (mM)","assays","Kinetic Solubility (mM)","Kinetic Solubility (mM)","Unnamed: 8","Unnamed: 9"], axis=1)
trainingdf = trainingdf.rename(columns={'S0 (mM)': 'S0'})
trainingdf=trainingdf[trainingdf["S0"].notnull()]
trainingfingerprintarray=makefingerprintsarrayfromsmilesdfcolumn(trainingdf,"SMILES")

testdf = pd.read_excel('data/soldata_prediction_withSvalues.xlsx',sheet_name='32 data')
testdf = testdf.rename(columns={"Solubility (from findings) (micro M)": 'Actual Solubility'})
testdf=testdf[testdf["Actual Solubility"]!="too soluble"]
testfingerprintarray=makefingerprintsarrayfromsmilesdfcolumn(testdf,"SMILES")

yvalues=trainingdf["S0"].values
xvalues=trainingfingerprintarray
testvalues=testfingerprintarray

clf = svm.SVC(gamma=0.001, C=100.)

lab_enc = preprocessing.LabelEncoder()
yvalues_encoded = lab_enc.fit_transform(yvalues)
#print(yvalues_encoded)
#print(utils.multiclass.type_of_target(yvalues))
#print(utils.multiclass.type_of_target(yvalues.astype('int')))
#print(utils.multiclass.type_of_target(yvalues_encoded))

print(clf.fit(xvalues,yvalues_encoded))
encodedpredictions=clf.predict(testvalues)
predictedsolubilities=lab_enc.inverse_transform(encodedpredictions)
testdf["PredictedSolubilities"]=predictedsolubilities

testdf["%Diff"]=100*(testdf["Actual Solubility"]-testdf["PredictedSolubilities"])/testdf["Actual Solubility"]
testdf["HitOrMiss"]="Hit"
testdf.loc[testdf["%Diff"]>10.0,"HitOrMiss"]="Miss"
testdf.loc[testdf["%Diff"]<-10.0,"HitOrMiss"]="Miss"
testdf.to_csv("predictedsolubilities.csv",header=True, index=False, encoding='utf-8')

yvalues_pred=lab_enc.inverse_transform(clf.predict(xvalues))
rmse = math.sqrt(mean_squared_error(yvalues, yvalues_pred))
print(rmse )

plt.scatter(testdf["Actual Solubility"], testdf["PredictedSolubilities"])
plt.xlabel("Actual Solubility")
plt.ylabel("PredictedSolubilities")
plt.savefig("plot.png")
	
dump(clf, 'CLFTry1.joblib') 
clf = load('CLFTry1.joblib')


