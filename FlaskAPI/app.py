import flask
from flask import jsonify,Flask,request
import json
import pickle
from data import test_data
import os
import pandas as pd

def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "model_file.p")
    # file_name = 'models\model_file.p'
    with open(model_path,'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model




app = Flask(__name__)
@app.route('/predict',methods=['GET'])
def predict():
    request_json = request.get_json()
    x = request_json['input']
    input_data= pd.DataFrame(x)
    model = load_model()
    result= model.predict(input_data)
    
    # response = json.dumps({'response':str(f'<h1> {result} </h1>')})
    response = json.dumps(str(f' {result} '))
    html = f'''
    <table border="1" cellspacing="0" cellpadding="8">
    <tr>
        <th>Prediction for 1st Input</th>
        <th>Prediction for 2nd Input</th>
        <th>Prediction for 3rd Input</th>
        <th>Prediction for 4th Input</th>
        <th>Prediction for 5th Input</th>
    </tr>
    <tr>
        <td>{round(result[0],2)}</td>
        <td>{round(result[1],2)}</td>
        <td>{round(result[2],2)}</td>
        <td>{round(result[3],2)}</td>
        <td>{round(result[4],2)}</td>
    </tr>
</table>

    '''
    # response = json.dumps(html)
    return  response,200

if __name__ =='__main__':
    application.run()
