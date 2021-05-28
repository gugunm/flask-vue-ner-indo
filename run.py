# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './backend')
import ner
import requests
from flask_cors import CORS
from random import *
from flask_restful import Resource, Api
from flask import Flask, render_template, jsonify, request


app = Flask(__name__,
            static_folder="./dist/static",
            template_folder="./dist")

api = Api(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

class Status(Resource):    
     def get(self):
         try:
            return {'data': 'Api running'}
         except(error): 
            return {'data': error}

class PredictNerOfSentence(Resource):
    def __init__(self):
        super().__init__()
        self.ner_result = ''

    def get(self):
        # self.ner_result = ner.predictNerOfSentence()
        return jsonify({'data': self.ner_result})
    
    def post(self):
        json_data = request.get_json(force=True)
        sentence = json_data["text"]
        self.ner_result = ner.predictNerOfSentence(sentence)
        return jsonify({'data': self.ner_result})


class PredictByDate(Resource):
    def __init__(self):
        super().__init__()
        self.ner_by_date = ner.predictByDate()

    def get(self):
        return jsonify({
            'data': self.ner_by_date
        })


api.add_resource(PredictNerOfSentence, '/api/predict')

api.add_resource(PredictByDate, '/api/predict-by-date')

api.add_resource(Status, '/')


if __name__ == '__main__':
    app.run()




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