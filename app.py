from flask import Flask,render_template,request
import pickle
import numpy as np
from crop import res

app=Flask(__name__)

#ML model sheeesh
model=pickle.load(open('model.pkl','rb'))

app.debug = True

@app.route('/',methods=['GET'])
def hello():
    return render_template("crop.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict_proba(final)
    print("Prediction Array:", prediction)
    # crop_name = res[prediction[0]]  # Map index to crop name
    predicted_class_index = np.argmax(prediction)
    crop_name = res[predicted_class_index]

    return render_template('crop.html', prediction_text=crop_name)