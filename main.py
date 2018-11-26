"""
Flask will run on port 5000
Check functionality with image cat.jpg Response is a text file which can be displayed with check_b64decode.py
(echo -n '{"data": "'; base64 cat.jpg; echo '"}') |
curl -X POST -H "Content-Type: application/json" -d @- http://0.0.0.0:5000 > response.txt
"""

from flask import Flask, current_app, request, jsonify, send_file, abort
from io import StringIO
import os
import numpy as np
from src.models.LSTM.predict import ToxicLSTMModel
#from initialize_app import initialize

app = Flask(__name__)
#predict_on_text = initialize()
ToxicLSTM = ToxicLSTMModel()


@app.route('/', methods=['POST', 'GET'])
def predict_lstm():
    '''
    :return: byte64 decoded colorized image
    '''
    try:
        text = request.get_json()['text']
        current_app.logger.info('Data type: %s', type(request.get_json()))
        print(text)
    except KeyError:
        current_app.logger.info('Data type: %s', type(request.get_json()))
        return jsonify(status_code='400', msg='Bad Request'), 400

    current_app.logger.info('Data: %s', text)
    #try:
    toxicity = ToxicLSTM.predict(text)
    #t2 = toxicity.to_list()
    print(toxicity)
    #print(t2)
    #except:
    #    return jsonify(status_code='400', msg='Text not understood'), 400

    current_app.logger.info('Predictions: %s', toxicity)


    #try:
    return jsonify(toxicity)
        #return send_file('colorized_img.jpg')
    #except:
    #    abort(404)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001)
