"""
Flask will run on port 5000
Check functionality with image cat.jpg Response is a text file which can be displayed with check_b64decode.py
(echo -n '{"data": "'; base64 cat.jpg; echo '"}') |
curl -X POST -H "Content-Type: application/json" -d @- http://0.0.0.0:5000 > response.txt
"""

from flask import Flask, current_app, request, jsonify, send_file, abort
from io import StringIO
from skimage.io import imsave
import io
import numpy as np
from initialize_app import initialize
import base64
import scipy.misc
from io import BytesIO
from PIL import Image
from flask_cors import CORS, cross_origin
from gevent.pywsgi import WSGIServer
import skimage

app = Flask(__name__)
CORS(app)
#app.config['CORS_HEADERS'] = 'Content-Type'
predict_on_text = initialize()

def serve_pil_image(pil_img):
    img_io = StringIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


@app.route('/', methods=['POST', 'GET'])
@cross_origin(["http://www.khumbu.ai", "https://www.khumbu.ai"])
def predict():
    '''
    :return: byte64 decoded colorized image
    '''
    try:
        text = request.get_json()['data']
    except KeyError:
        current_app.logger.info('Data type: %s', type(request.get_json()))
        return jsonify(status_code='400', msg='Bad Request'), 400

    current_app.logger.info('Data: %s', text)
    try:
        toxicity, attentions = predict_on_text(text)
        attentions = attentions[:len(text)]
    except:
        return jsonify(status_code='400', msg='Image not understood'), 400

    current_app.logger.info('Predictions: %s', toxicity)


    try:
        return toxicity + attentions
        #return send_file('colorized_img.jpg')
    except:
        abort(404)


if __name__ == '__main__':
    #app.run(port=5000, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
