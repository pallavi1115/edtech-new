import numpy as np
from flask import Flask, request, render_template
import pickle

# flask app
app = Flask(__name__)
# loading model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template('home.html')

@app.route('/Output', methods = ['POST'])
def home1():
    x = str(request.form['Time_Period'])
    if x == 'Data Science':
        return render_template('datascience.html')
    elif x == "Cloud Computing":
        return render_template('cloud_computing.html')
    elif x == "Deep Learning":
        return render_template('deep_learning.html')
    elif x == "Java":
        return render_template('Java.html')
    elif x == "Python":
        return render_template('python.html')
    elif x == "tableau":
        return render_template('tabluea.html')
    else:
        return render_template('home.html')

@app.route('/datascience' ,methods = ['POST'])
def datascience():
    final_features = [int(x) for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)

    return render_template('datascience.html', output='Price of the Data Science Course is ₹ {}'.format(prediction[0]))

@app.route('/cloud_computing' ,methods = ['POST'])
def cloud_computing():
    final_features = [int(x) for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)

    return render_template('cloud computing.html', output='Price of the Colud Computing Course is ₹ {}'.format(prediction[0]))

@app.route('/deep_learning' ,methods = ['POST'])
def deep_learning():
    final_features = [int(x) for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)

    return render_template('deep_learning.html', output='Price of the Deep Learning Course is ₹ {}'.format(prediction[0]))

@app.route('/Java' ,methods = ['POST'])
def Java():
    final_features = [int(x) for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)

    return render_template('Java.html', output='Price of the Java Course is ₹ {}'.format(prediction[0]))

@app.route('/python' ,methods = ['POST'])
def python():
    final_features = [int(x) for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)

    return render_template('python.html', output='Price of the python Course is ₹ {}'.format(prediction[0]))

@app.route('/tabluea' ,methods = ['POST'])
def tabluea():
    final_features = [int(x) for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)

    return render_template('tabluea.html', output='Price of the tabluea Course is ₹ {}'.format(prediction[0]))



if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)