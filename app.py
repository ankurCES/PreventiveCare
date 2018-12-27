#!flask/bin/python
import json
from flask import Flask, request
from run_prediction import start_prediction
from readmission_prediction import run_modelling

app = Flask(__name__)

@app.route('/get_prediction', methods=['POST'])
def index():
    payload = request.get_json()
    ret_val = start_prediction(payload)
    ret_val = json.dumps(ret_val)
    return ret_val

@app.route('/run_modelling', methods=['GET'])
def learn():
    ret_val = run_modelling()
    return json.dumps(ret_val)

if __name__ == '__main__':
    app.run(debug=True)
