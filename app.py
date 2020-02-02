import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import pickle, json

# load model
model = pickle.load(open('model.pkl','rb'))
le = pickle.load(open('labelencoder.pkl','rb'))

# get featurs names that used in the model
feature_names = model.get_booster().feature_names

def lbl_encoder(df,le):
    #convert all features to float64
    quantitative = [f for f in df.columns if df.dtypes[f] != 'object']
    df[quantitative] = df[quantitative].astype(np.float64)

    qualitative = [f for f in df.columns if df.dtypes[f] == 'object']
    for c in qualitative:
        le = le.fit(list(df[c].values))
        df[c] = le.transform(list(df[c].values))
        
    return df


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
    
    # preprocessing
    data_df_encode = data_df.copy()[feature_names]
    data_df_encode = lbl_encoder(data_df_encode,le)
    
    # predictions
    pred = model.predict_proba(data_df_encode)
    
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
    app.run(port = 5000, debug=True)
