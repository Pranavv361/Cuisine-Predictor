#Libraries Used
import os
import pandas as pd
from project2 import load_data, main
from modules import PreprocessAndModelling
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import joblib

#Testing load_data function with test data
def test_load_data():
     
    dir_path = os.path.join(os.getcwd(), 'test')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    data_file_name = 'test.json'
    data_file_path = os.path.join(dir_path, data_file_name)

    test_data = [{'id':'1', 'cuisine':'test', 'ingredients':'[salt, pepper]'},
                  {'id':'2', 'cuisine':'test2', 'ingredients':'[salt, sugar]'}
                  ]
    pd.DataFrame(test_data).to_json(data_file_path)
    data = load_data(data_file_path)
    #breakpoint()
    #Assert the output   
    assert isinstance(data, pd.DataFrame)
    assert data.columns.tolist() == ['id', 'cuisine','ingredients']
    assert data.values.tolist() == [[1, 'test', '[salt, pepper]'], [2,'test2', '[salt, sugar]']]


#Testing normalizing data function with test inputs
def test_normalize_data():
    input_ingredients = ["salt", "pepper", 
                         "almond flour","ice water","corn starch",
                         "chopped cilantro","plain yogurt"]

    normalized_ingred = PreprocessAndModelling.normalize_data(input_ingredients)
    #breakpoint()
    #Assert the output
    assert normalized_ingred == "salt, pepper, almond_flour, ice_water, corn_starch, cilantro, plain_yogurt"


#Testing model function by loading 'yummly.json' file
def test_modeltrain():
    file = 'yummly.json'
    data = load_data(file)

    cuisine, score, closest = PreprocessAndModelling.modeltrain(data, ['bananas','milk','salt'], 2)
    #breakpoint()
    # Assert the output
    assert isinstance(cuisine, list)
    assert isinstance(score, list)
    assert isinstance(closest, list)
    assert len(cuisine) == 3
    assert len(score) == 3
    assert len(closest) == 2