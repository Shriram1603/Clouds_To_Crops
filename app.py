from flask import Flask,render_template,request
import pickle
import numpy as np
from crop import res
# from sklearn.preprocessing import LabelEncoder

app=Flask(__name__)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

#ML model sheeesh
model=pickle.load(open('model.pkl','rb'))

app.debug = True

@app.route('/',methods=['GET'])
def hello():
    return render_template("crop.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict_proba(final)
    # crop_name = res[prediction[0]]  # Map index to crop name
     # Get the index of the predicted class
    predicted_class_index = np.argmax(prediction)
    
    # Use the label_encoder to map the index back to the original crop name
    crop_name = label_encoder.inverse_transform([predicted_class_index])[0]

    return render_template('crop.html', prediction_text=crop_name)