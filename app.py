from flask import Flask, render_template
from keras.models import load_model
import numpy as np


app = Flask(__name__)

model = load_model('model.h5')
data = np.load('data.npy')


@app.route('/')
def index():
    prediction = model.predict(data[-7:])
    return  render_template('index.html',prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)