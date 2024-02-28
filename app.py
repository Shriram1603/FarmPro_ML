from flask import Flask,render_template,request
import pickle
import numpy as np
import pickle

app=Flask(__name__)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)


#ML model sheesh :
model=pickle.load(open('model.pkl','rb'))

Yield=pickle.load(open('Yeild.pkl','rb'))

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
@app.route('/predict_yield', methods=['GET','POST'])
def predict_yield_get():
     
     if request.method == 'POST':
        # Get input values from the form
        crop = int(request.form.get('crop'))
        season = int(request.form.get('season'))
        state = int(request.form.get('state'))
        area = float(request.form.get('area'))
        annual_rainfall = float(request.form.get('annual_rainfall'))
        fertilizer = float(request.form.get('fertilizer'))
        pesticide = float(request.form.get('pesticide'))


        # Make prediction
        prediction = Yield.predict([[crop, season, state, area, annual_rainfall, fertilizer, pesticide]])

        # Display prediction
        return render_template('predict_yield.html', prediction_text=prediction[0])
     return render_template('predict_yield.html')
