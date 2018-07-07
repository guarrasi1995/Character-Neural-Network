import sys
import numpy as np
from keras.models import model_from_json
import pickle
import string
import csv

#-load model
#-load path\to_file.npy
# load json and create model
json_file = open('../data_out/model/model.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
# load weights into new model
model.load_weights("../data_out/model/model.h5")
print("Loaded model from disk")

#X_test = np.load("../data/clean_data_white.npy")
X_test = np.load(sys.argv[1])
np.save('../data_out/img/data_michele',X_test)

#normalize
'''for i in range(len(X_test)):
    X_test = 1000*X_test
    min_X_test = np.min(X_test[i])
    max_X_test = np.max(X_test[i])
    X_test[i] = (X_test[i] -min_X_test)/max_X_test'''
#-prediction
prediction = model.predict(X_test, verbose = 1)

#referemets
fonts = sorted(list(set(pickle.load( open( "../data/fonts.p", "rb" ) ))))
char = sorted(list(string.printable[:-6]))
bold =[False, True]
italics =[False, True]

col_0 = []
col_1 = []
col_2 = []
col_3 = []
for i in range(4):
    if i == 0:#font
        for probs in prediction[i].tolist():
            position = probs.index(max(probs))
            col_0.append(fonts[position])
    elif i == 1:#char
        for probs in prediction[i].tolist():
            position = probs.index(max(probs))
            col_1.append(char[position])
    elif i == 2:#bold
        for probs in prediction[i].tolist():
            position = probs.index(max(probs))
            col_2.append(bold[position])
    else:#italics
        for probs in prediction[i].tolist():
            position = probs.index(max(probs))
            col_3.append(italics[position])
    
#-save PRED-->DATA_OUT/TEST/output.csv
rows = zip(col_1,col_0,col_2,col_3)

with open('../data_out/test/'+sys.argv[2]+'.csv', "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
        

