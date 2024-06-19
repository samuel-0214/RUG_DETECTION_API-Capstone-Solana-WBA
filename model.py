import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split


path = ('preProcessedTokens.json')

def load_data(path):
    return pd.read_json(path)

data = load_data(path)
data.head() # to get the output you need to use print command

##Lets split the data and preprocess it 

def pro_data(df):
    df = df.drop(['address', 'lastTradeUnixTime', 'mc'], axis=1)
    X = df.drop('Risk', axis=1)
    y = df['Risk'].map({'Danger':1,'Warning':1,'Good':0}).astype(int)
    return train_test_split(X,y,test_size=0.3,random_state=42)

def preprocessing(X_train):
    num_features = ['decimals','liquidity','v24hChangePercent','v24hUSD','Volatility','holders_count']
    cat_features = ['logoURI','name','symbol']

    num_tranforming = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='mean')),
        ('scaler',StandardScaler())
    ])

    cat_transforming = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ('num',num_tranforming,num_features),
            ('cat',cat_transforming,cat_features)
        ],
        remainder='passthrough'
    )

    return preprocess


def train_model(X_train,y_train,preprocess):
    model = Pipeline(steps=[
        ('preprocess',preprocess),
        ('classifier',xgb.XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42))
    ])
    model.fit(X_train,y_train)
    return model


## Evaluation of the model

def evaluation(model,X_test,y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    classification_report_results = classification_report(y_test,y_pred)
    conf_matrix = confusion_matrix(y_test,y_pred)

    print(f"Model Accuray: {accuracy}")
    print(f"Classification Report:{classification_report_results}")
    print("Confusion Matrix:",conf_matrix)


## Main function 

def main():
    file_path = 'preProcessedTokens.json'
    df = load_data(file_path)
    X_train,X_test,y_train,y_test = pro_data(df)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    preprocess = preprocessing(X_train)
    model = train_model(X_train,y_train,preprocess)
    evaluation(model,X_test,y_test)

    # joblib.dump(model,"predictModel.pkl")
    # joblib.dump(preprocess,"importantPreprocessing.pkl")

    # Example prediction

#     single_item_corrected = {
#     "decimals": 6,
#     "liquidity": 62215.15524335994,
#     "logoURI": "https://img.fotofolio.xyz/?url=https%3A%2F%2Fbafkreifhqihaiwyo4g2aogdu4qyfqftkxy3aq4xxbhoxdkbkufrobsnjwm.ipfs.nftstorage.link",
#     "name": "SBF",
#     "symbol": "SBF",
#     "v24hChangePercent": -49.17844813082829,
#     "v24hUSD": 18220.724466666383,
#     "Volatility": 76.06539722778419,
#     "holders_count": 0
# }

#     # Convert to DataFrame
#     single_item_df = pd.DataFrame(single_item_corrected, index=[0])
#     prediction = model.predict(single_item_df)  # Predict
#     print(f'Prediction for the single item: {prediction}')

if __name__ == "__main__":
    main()