import numpy as np
from flask import Flask, request, render_template
import pickle

# flask app
app = Flask(__name__,template_folder='templates')
# loading model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict' ,methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    #output=prediction[0];
    return render_template('home.html', output='Price of the Course is ₹ {}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
    
    
    
    
    
    
    
    
    