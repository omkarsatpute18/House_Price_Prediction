import joblib
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE ="model.pkl"
PIPELINE_FILE = "pipline.pkl"

def build_pipeline(num_attributs,cat_attributs):
    #For numerical pipline
    num_pipeline = Pipeline([
        ("impute",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    # For categorical pipeline
    cat_pipeline = Pipeline([
        ("encoding",OneHotEncoder(handle_unknown="ignore"))
    ])

    # construct full pipeline
    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attributs),
        ("cat",cat_pipeline,cat_attributs)
    ])
    return full_pipeline
if not os.path.exists(MODEL_FILE):
    #Load the dataset
    housing = pd.read_csv("housing.csv")

    #create a stratified split
    housing["income_cat"] = pd.cut(housing["median_income"],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,train_size=0.8,random_state=42)

    for train_set,test_set in split.split(housing,housing["income_cat"]):
        housing.iloc[test_set].drop("income_cat",axis=1).to_csv("input.csv",index=False)
        housing = housing.iloc[train_set].drop("income_cat",axis=1)
    
    housing_labels = housing["median_house_value"].copy() 
    housing_features = housing.drop("median_house_value",axis=1)

    num_attributs = housing_features.drop("ocean_proximity",axis=1).columns.tolist()
    cat_attributs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attributs,cat_attributs)
    housing_prepared = pipeline.fit_transform(housing_features)
    # print(housing_prepared)
    model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
    model.fit(housing_prepared,housing_labels)

    joblib.dump(model, MODEL_FILE, compress=3)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model is trained!!")

else:
    #Lets do inferance
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    tranformed_input = pipeline.transform(input_data)
    predictions = model.predict(tranformed_input)
    input_data["median_house_value"] = predictions

    input_data.to_csv("output.csv",index=False)
    print("Inferance is complete results saved to output.csv")