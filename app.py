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
temp=pickle.load(open('temp_model.pkl','rb'))
rain=pickle.load(open('rain_model.pkl','rb'))

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

@app.route('/weather',methods=['POST','GET'])
def weather():
    if request.method == 'POST':
        # Get the input values from the form
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])

        # Create an input array for both models
        input_data = np.array([[latitude, longitude, year, month, day]])

        # Use the temperature model to predict temperature
        temperature_prediction = temp.predict(input_data)[0]

        # Use the rainfall model to predict rainfall
        rainfall_prediction = rain.predict(input_data)[0]

        # Prepare the result to be displayed
        result = f"Predicted Temperature: {temperature_prediction:.2f} °C<br>Predicted Rainfall: {rainfall_prediction:.2f} mm"

        return render_template('weather.html', result=result)

    return render_template('weather.html')

