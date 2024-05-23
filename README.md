## Cuisine Predictor
## Author - Pranav Vichare
## Cuisine Predictor
**About:**  Project 2's purpose is to produce applications that take a user's list of components and attempt to forecast the type of cuisine and related meals. Assume a chef has a list of ingredients and wants to update the present meal without changing the components. The following steps should be taken.
1. Train (or index for search) the specified food data.
2. Request that the user enter all of the elements in which they are interested (through command line arguments).
3. Predict the type of cuisine using the model (or search index) and inform the user.
4. Find the top-N most nearby foods (N specified via a command line option). Return to the user the IDs of those dishes. If a dataset does not have IDs assigned to it, you can add them at your leisure.

To run the code, Use the commands below in terminal or Visual Studio Code
```python
pipenv --python 3.9  #use your version of python which is installed on your system. This code is also used to create a virtual environment
pipenv install scikit-learn #to install scikit-learn library in virtual environment
pipenv install nltk #to install nltk library in virtual environment
pipenv install pandas #to install pandas library in virtual environment
pipenv install pytest #to install pytest library in virtual environment
pipenv lock #to create piplock file
```

```python
import json
import sys
import argparse
import numpy as np
import os
import pandas as pd
from modules import PreprocessAndModelling
import traceback
import re

import nltk
nltk.download('wordnet',quiet = True)
nltk.download('omw-1.4',quiet = True)

from nltk.stem import WordNetLemmatizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib
```
To run the code, Import all the libraries listed above in the python code.

To run the code from the terminal use below command:
```
pipenv run python project2.py --N 5 --ingredient bananas --ingredient paprika --ingredient "rice krispies"
```

### Code Explanation  
```python
#project2.py and PreprocessAndModelling.py has 4 functions as shown below.
def load_data()
def main()
def normalize_data()
def modeltrain()
```
**load_data()** reads data from a JSON file and returns it as a pandas DataFrame.

When the script is executed, **the main()** method is invoked. It takes an args parameter, which is parsed by argparse. The function calls **load_data()** to load data from a JSON file, and then **modeltrain()** to train and test a LinearSVC model on the loaded data.

The function **normalize_data()** is used to normalize a list of ingredients. It strips out specific verbs from the ingredients, lemmatizes the remaining words, removes non-alphabetic characters and spaces, and replaces spaces with underscores. The normalized ingredients are returned as a comma-separated string.

**modeltrain()** trains and tests a LinearSVC model on the given data. It normalizes the ingredients and removes duplicates, divides the input data into features (ingredients) and target (cuisine), encodes the target variable, uses TfidfVectorizer to convert the text data into numerical form, builds a pipeline with a preprocessor and estimator, performs cross-validation, fits the model on training data, predicts on testing data, and saves the model to a joblib file. If the model is already there, the function loads it from the file. The function accepts three arguments: data, pandas, and The input data, ingred - a list of ingredients for which the cuisine must be predicted, and top_n - a number indicating how many of the top closest matches to return. The function produces a tuple containing the names and scores of the top N closest matches, as well as a list of dictionaries containing the closest matches.

These values are stored in the stats files.
*The output video of the execution with the images is stored in **Output videos and images** folder*

### Code Output
The code output is stored in stored in file with name : **output.json**.
![image](https://github.com/Pranavv361/cs5293sp23-project2/blob/main/Output%20videos%20and%20images/project2.py%20Execution.png)

### Test Output
The test cases are created in single file **test_project2.py**. The purpose of the test_project2.py is to check the functions with sample input and make sure we get correct output. The attached image below shows the output for test cases of all the functions.
To run the test_project2.py using pytest library use the following code.
```
pipenv run python -m pytest
```
![image](https://github.com/Pranavv361/cs5293sp23-project2/blob/main/Output%20videos%20and%20images/test_project2.py%20Execution.png)

The **load_data()** function is tested using the **test_load_data()** method. It makes a test directory, generates a JSON file with test data named **test.json** in **docs** folder, then loads it with the **load_data()** function. The function then checks to see if the output is a pandas DataFrame, if the column names are correct, and if the data matches the expected values.

The function **test_normalize_data()** verifies the **normalize_data()** function defined in the PreprocessAndModelling module. It sends a list of ingredient names to the **normalize_data()** method and checks for a string of normalized ingredient names.

The **modeltrain()** function created in the PreprocessAndModelling module is tested using the **test_modeltrain()** method. It uses the **load_data()** method to load the yummly.json file, and then gives the loaded data, a list of ingredients, and a number to the **modeltrain()** function. After that, the function checks to see if the output lists contain the expected lengths and data types.

### Assumptions:
1. The data file should always be .json file.

### Bugs:   
1. The LSVC model will not work for datapoints less than 10 as it uses crossvalidation with cv = 10.
