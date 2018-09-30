import pickle
import numpy as np
from flask import Flask, render_template, request, url_for
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


@app.route('/')
def home():
    return render_template('home.html')


class Predict(Resource):
    def post(self):
        data = request.get_json(force=True)
        with open('model', 'rb') as f:
            model = pickle.load(f)
        arr = np.array([[float(data['pclass']), float(data['sex']), float(data['age']), float(data['sibsp']), float(data['parch']),0.14, float(data['port'])]])
        answer = model.predict(arr)[0]
        return {'survived': str(answer)}

api.add_resource(Predict, '/predict')
