# best performing model is GBDT with TF-IDF vectors along with hand crafted features
# let's use these to build an api for finding the similarity between given two questions
print("importing libraries...\n")

import warnings
warnings.filterwarnings('ignore')
from flask import Flask, jsonify, request, redirect, url_for, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from feauture import extract_features
from sklearn.calibration import CalibratedClassifierCV


print("loading models...\n")
d = "./models/"
with open(d+"std_tfidf.pkl", "rb") as f:
    std = pickle.load(f)
with open(d+"tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open(d+"tfidf_GBDT_model.pkl", "rb") as f:
    model = pickle.load(f)


app = Flask(__name__)

@app.route('/')
def inputs():
   return render_template('index.html')

@app.route('/output/', methods=["POST"])
def output():
    a="me"
    data = request.form.to_dict()
    q1 = data.get('q1')
    q2 = data.get('q2')
    prob = data.get('probabiliy')

    #convert it into dataframe
    new_df = pd.DataFrame(columns = ['Marking Scheme','Student Response'])
    new_df = new_df.append({'Marking Scheme': q1, 'Student Response':q2}, ignore_index = True)
    new_df = extract_features(new_df) #getting advance and basic features
    #get the tfidf vectorizer of text
    x_q1 = vectorizer.transform(new_df["Marking Scheme"])
    x_q2 = vectorizer.transform(new_df["Student Response"])
    cols = [i for i in new_df.columns if i not in ['Marking Scheme', 'Student Response']]
    new_df = new_df.loc[:,cols].values
    #get the hand crafted features
    X = hstack((x_q1, x_q2, new_df)).tocsr()
    X = std.transform(X)

    y_q = model.predict(X)
    y_q_proba = model.predict_proba(X)
    result = dict()
    result["Marking Scheme"] = q1
    result["Student Response"] = q2
    
    
    if y_q == 1:
        result["Predicted Class"] = 'Correct'
    else:
        result["Predicted Class"] = "Wrong"

    if prob=="yes":
        result["Probabiliy"] = y_q_proba[0]
        
    print(max(y_q_proba[0]))
    return render_template('output.html', result = result)

if __name__ == "__main__":
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)
    app.run(debug=True)
    
