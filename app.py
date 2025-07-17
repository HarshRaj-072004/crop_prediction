

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import kagglehub
import os
import pickle

path=kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")

print("Path: ",path)

print("path",os.listdir(path))

data=pd.read_csv(os.path.join(path,"Crop_recommendation.csv"))

data.head()

data.isnull().sum()

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42);

model=RandomForestClassifier()

model.fit(X_train,Y_train)

predictions=model.predict(X_test)

accuracy=model.score(X_test,Y_test)

print(accuracy)

#Model_deployment

pickle.dump(model,open('model.pkl',"wb"))

from flask import Flask, request, render_template


# ✅ Define your Flask app
flask_app = Flask(__name__)





model=pickle.load(open("model.pkl","rb"))

@flask_app.route("/")
def Home():
  return render_template("index.html")

@flask_app.route("/predict",methods=["POST"])

def predict():
  float_feature=[float(x) for x in request.form.values()]
  features=[np.array(float_feature)]
  prediction=model.predict(features)
  return render_template("index.html",prediction_text=f"The predicted crop: {prediction[0]}")

if __name__=="__main__":
  flask_app.run(debug=True)