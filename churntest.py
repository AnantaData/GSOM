import pandas as pd
import numpy as np
from gsom import gsomap
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
import sys

data_original = pd.read_csv("tourn_1_calibration_csv.csv")
import time
data_original = data_original[::50]
#print np.array(data_original.columns)
churned=np.array(np.where(data_original['churn']>=0.85))[0]
churns=np.zeros(shape=(data_original.shape[0]))
for i in range(churned.shape[0]):
    churns[churned[i]]=1
predictors= data_original.drop('churn',1)
predictors= predictors.drop('Customer_ID',1)



#print predictors.shape
start_time = time.time()
gmap = gsomap(SP=0.9095,dims=20,nr_s=4,lr_s=0.9,fd=0.7375)

preds = np.array(predictors)

label_encoder = LabelEncoder()
enc_preds = label_encoder.fit_transform(preds[:,0])

for i in range(1, preds.shape[1]):
    label_encoder = LabelEncoder()
    enc_preds = np.column_stack((enc_preds, label_encoder.fit_transform(preds[:,i])))


for i in range(enc_preds.shape[1]):
    enc_preds[:,i]= enc_preds[:,i]/np.max(enc_preds[:,i])

ch2 = SelectKBest(chi2, k=20)
enc_preds = ch2.fit_transform(enc_preds, churns)

print enc_preds.shape
gmap.process_batch(enc_preds,1)

coords =[]
i=1;
for inp in np.array(enc_preds):
    sys.stdout.write(" predicting %d%% \r"%(i*100.00/(1000)))
    coords.append(gmap.predict_point(inp))
    i+=1

end_time = time.time()

print "time elapsed :::: "
print end_time-start_time
print "::::::::::::::::::"

X = np.array(coords).astype(int)[:,0]
Y = np.array(coords).astype(int)[:,1]

colors={0:"green", 1:"red"}
colorlist=[]
for x in churns:
    colorlist.append(colors[int(x)])

colorlist=np.array(colorlist)

sizes = [20*2**2 for n in range(data_original.shape[0])]

plt.scatter(X, Y, c=colorlist, s=sizes)
plt.show()
gmap.viewmap()