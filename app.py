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
fertilizer=pickle.load(open('fertilizer.pkl','rb'))
demand=pickle.load(open('demand.pkl','rb'))


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


@app.route('/recommend_fertilizer', methods=['GET', 'POST'])
def recommend_fertilizer():
    if request.method == 'POST':
        # Parse input values from the form
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        moisture = float(request.form.get('moisture'))
        soil_type = int(request.form.get('soil_type'))
        crop_type = int(request.form.get('crop_type'))
        nitrogen = int(request.form.get('nitrogen'))
        potassium = int(request.form.get('potassium'))
        phosphorous = int(request.form.get('phosphorous'))

        # Make prediction using the trained model
        # input_data = [[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]]
        recommended_fertilizer = fertilizer.predict([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]])

        # Render template with prediction result
        return render_template('fertilizer_input_form.html', recommended_fertilizer=recommended_fertilizer[0])

    # Render the input form template for GET requests
    return render_template('fertilizer_input_form.html')

@app.route('/predict_demand', methods=['POST','GET'])
def predict_demand():
    if request.method == 'POST':
        # Get input values from the form
        state = int(request.form.get('state'))
        month = int(request.form.get('month'))
        crop = int(request.form.get('crop'))

        # Make prediction
        prediction = demand.predict([[state, month, crop]])

        # Display prediction
        return render_template('predict_demand.html', prediction_text=prediction[0])
    return render_template('predict_demand.html')




