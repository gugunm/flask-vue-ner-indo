# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './backend')
import ner
import requests
from flask_cors import CORS
from random import *
from flask_restful import Resource, Api
from flask import Flask, render_template, jsonify


app = Flask(__name__,
            static_folder="./dist/static",
            template_folder="./dist")

api = Api(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


# @app.route('/api/random')
# def random_number():
#     response = {
#         'randomNumber': randint(1, 100)
#     }
#     return jsonify(response)


# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def catch_all(path):
#     if app.debug:
#         return requests.get('http://localhost:8080/{}'.format(path)).text
#     return render_template("index.html")

class Status(Resource):    
     def get(self):
         try:
            return {'data': 'Api running'}
         except(error): 
            return {'data': error}

class Predict(Resource):
    def __init__(self):
        super().__init__()
        self.ner_result = ner.predictNer()

    def get(self):
        return jsonify({'data': self.ner_result})

api.add_resource(Predict, '/api/predict')

api.add_resource(Status, '/')


if __name__ == '__main__':
    app.run()
