import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import PolynomialFeatures


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data =request.form['msg']
    data=float(data)
    #int_features = (float(x) for x in request.form.values())
    final_features = np.array(data)
    poly = PolynomialFeatures(degree = 5)
    prediction = poly.fit_transform(final_features.reshape(-1,1))

    prediction = model.predict(prediction)
    
    

    #output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Water Temperature should be degrees {}'.format(prediction))




if __name__ == "__main__":
    app.run(debug=True)
