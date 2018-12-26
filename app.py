#!flask/bin/python
import json
from flask import Flask, request
from run_prediction import start_prediction

app = Flask(__name__)

@app.route('/get_prediction', methods=['POST'])
def index():
    payload = request.get_json()
    ret_val = start_prediction(payload)
    return str(ret_val)

if __name__ == '__main__':
    app.run(debug=True)
