'''
curl -H "Content-Type: application/json" -d '{"data":"I hate you"}' http://localhost:5001/
'''

from flask import Flask, current_app, request, jsonify, abort
from flask_cors import CORS, cross_origin
from gevent.pywsgi import WSGIServer

from app.LSTM_MultiAttention.initialize_service import predict_on_text

app = Flask(__name__)
CORS(app)
#app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', methods=['POST', 'GET'])
@cross_origin(["http://www.khumbu.ai", "https://www.khumbu.ai"])
def predict():
    print(request.get_json())
    try:
        text = request.get_json()['data']
    except KeyError:
        current_app.logger.info('Data type: %s', type(request.get_json()))
        return jsonify(status_code='400', msg='Bad Request'), 400

    current_app.logger.info('Data: %s', text)
    try:
        predictions = predict_on_text(text)
    except:
        return jsonify(status_code='400', msg='Text {} could not be processed'.format(text)), 400

    current_app.logger.info('Predictions: %s', predictions)

    try:
        return jsonify(predictions)
    except Exception as e:
        print(e)
        abort(404)


if __name__ == '__main__':
    #app.run(port=5001, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5001), app)
    http_server.serve_forever()
