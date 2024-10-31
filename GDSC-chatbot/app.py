from flask import Flask, request
from flask_restx import Api, Resource, fields
import logging
from flask_cors import CORS

from reco import start_point, go

# Flask 객체 생성
app = Flask(__name__)
api = Api(app, title='API 문서')
CORS(app)
#CORS(app, resources={r"/api/*": {"origins": "https://catholicagg.web.app/home"}})

# 로깅 설정: INFO 수준의 로그를 콘솔에 출력
logging.basicConfig(level=logging.INFO)

start_point()

# 모델 정의
data_model = api.model('Data', {
    #'major': fields.String(required=True, description='The name of the user'),
    #'student_id': fields.Integer(required=True, description='The age of the user'),
    'question': fields.String(required=True, description='The question of the user')
})

test_api = api.namespace('test', description="조회 API")

@test_api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {"hello": "world!"}

@test_api.route('/data')
class Data(Resource):
    @api.expect(data_model)
    def post(self):
        # 요청의 Content-Type 검사
        if request.content_type != "application/json":
            return {"message": "Content type must be application/json"}, 415

        # 데이터 수신
        json_data = request.json

        if json_data is None:
            return {'message': "no json data provided"}, 400

        # 로그에 요청 정보 기록
        logging.info(f"Received data: {json_data}")

        major = json_data.get('major')
        student_id = json_data.get('student_id')
        question = json_data.get('question')

        processed_response = go(question)

        return {"response": processed_response}, 201


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)