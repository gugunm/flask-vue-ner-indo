# from flask import Flask, jsonify
# from flask_restful import Resource, Api
# from flask_cors import CORS

# import ner

# app = Flask(__name__)
# api = Api(app)
# CORS(app)


# class Status(Resource):    
#      def get(self):
#          try:
#             return {'data': 'Api running'}
#          except(error): 
#             return {'data': error}

# class Sum(Resource):
#     def get(self, a, b):
#         return jsonify({'data': a+b})


# class Predict(Resource):
#     def __init__(self):
#         super().__init__()
#         self.ner_result = ner.predictNer()

#     def get(self):
#         return jsonify({
#             'data': self.ner_result
#         })


# class PredictByDate(Resource):
#     def __init__(self):
#         super().__init__()
#         self.ner_by_date = ner.predictByDate()

#     def get(self):
#         return jsonify({
#             'data': self.ner_by_date
#         })

# api.add_resource(Status,'/')
# api.add_resource(Sum,'/add/<int:a>,<int:b>')

# api.add_resource(Predict,'/predict')
# api.add_resource(PredictByDate, '/predict-by-date')

# if __name__ == '__main__':
#     app.run()