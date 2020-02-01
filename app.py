import pandas as pd
from flask import Flask, jsonify, request
import pickle, json

# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)
    
    # convert data into dataframe
    data = json.dumps(data)
    
    data_df = pd.read_json(data)

    # predictions
    pred = model.predict_proba(data_df.drop('uuid',axis=1))
    
    # Prepare the results
    rs_df = pd.DataFrame()
    rs_df["uuid"] = data_df["uuid"].values
    rs_df["PD"] = pred[:,1]
    result = rs_df.to_json(orient='records')
    
    # Response
    output = {'results': result}
    
    # return output
    return jsonify(output)
if __name__ == '__main__':
	app.run(threaded=True, port=5000)
