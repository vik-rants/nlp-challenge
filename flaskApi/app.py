from flask import Flask, request,render_template, json
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/',methods=["Get","POST"])
def home():
    return render_template("index.html")

@app.route('/predict',methods=["Get","POST"])
def predict():
    uploaded_file = request.files['file']
    df = pd.read_csv(uploaded_file)
    df = data_validation(df)
    with open("../data/external/model.pkl", 'rb') as file:
            classifier = joblib.load(file)
    #xgb_clf = joblib.load('xgb_clf.pkl')
    predictions_test = classifier.predict(df['cleaned_review'].values.astype('U'))

    df['Predicted Sentiment'] = predictions_test
    return df.to_json(orient="split")

def data_validation(df):
     # some logic
     if df.empty:
         print("The csv is empty")
     return df

if __name__ == '__main__':
     app.run(debug=True, port=5002)