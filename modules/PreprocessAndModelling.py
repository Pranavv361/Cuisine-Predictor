#Libraries
import pandas as pd
import numpy as np
import re
import os

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

#Normalizing the ingredients
def normalize_data(ingredients: list[str]):
    skip_words = ["crushed", "crumbles", "ground", "minced", "powder",
                  "chopped", "sliced", "grilled", "boneless", "skinless", "steamed"]
    
    def remove_verbs(ingredient: str) :
        pattern = "|".join(skip_words)
        return re.sub(pattern, "", ingredient)
    
    def lemmatize(ingredient: str) :
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word.lower()) for word in ingredient.split()])
    
    ingredients = [remove_verbs(ingredient) for ingredient in ingredients]
    ingredients = [lemmatize(ingredient) for ingredient in ingredients]
    ingredients = [re.sub("[^A-Za-z ]", "", ingredient) for ingredient in ingredients]
    ingredients = [re.sub(" +", " ", ingredient) for ingredient in ingredients]
    ingredients = [ingredient.strip().replace(" ", "_") for ingredient in ingredients]
    
    return ", ".join(ingredients)



#Training Linear SVC model
def modeltrain(data, ingred, top_n):
    #if the model already exist skip the generation of model.
    if not os.path.exists("lsvc_pipe.joblib") or any(file.endswith(".joblib") for file in os.listdir()):
        # Normalize ingredients and remove duplicates
        data["ingredients"] = data["ingredients"].map(normalize_data)
        data = data[~data.duplicated(["cuisine", "ingredients"], keep="first")]

        # Separate X and Y
        y = data["cuisine"]
        x = data.drop(["cuisine"], axis=1)

        # Encoding target variable to transform them into numerical values
        le = LabelEncoder()
        y_transformed = le.fit_transform(y)
        
        # Splitting the data into training and testing sets
        x_train, x_test , y_train, y_test = train_test_split(x, y_transformed, test_size=0.2)

        # Apply TfidfVectorizer to convert the text data into numerical form
        preprocessor = ColumnTransformer(
        transformers=[
            ('vectorizer', TfidfVectorizer(
                ngram_range=(1,1), stop_words="english"), "ingredients")
            ])

        # Creating the pipeline
        lsvc_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', CalibratedClassifierCV(LinearSVC(C=0.9, penalty='l2')))
        ])

        #Performing Cross-validation
        cross_val_score(lsvc_pipe, x, y_transformed, cv=10)

        # Fitting the model on training data
        lsvc_pipe.fit(x_train, y_train)

        # Make predictions on testing data
        y_pred = lsvc_pipe.predict(x_test)

        # Saving the model
        joblib.dump(lsvc_pipe, "lsvc_pipe.joblib")

    # Loading the trained pipeline
    loaded_svc = joblib.load("lsvc_pipe.joblib")

    # normalize the input ingredient list
    test_data = pd.DataFrame([normalize_data(ingred)], columns=["ingredients"])

    # predicting the cuisine probabilities for the input ingredient list using the loaded pipeline
    pred_data = loaded_svc.predict_proba(test_data)

    # Sorting the cuisines by similarity score and top N
    cuisines = [(c, s) for c, s in zip(le.inverse_transform(loaded_svc.classes_), pred_data[0])]
    top_n_cuisines = sorted(cuisines, key=lambda x: x[1], reverse=True)[:top_n + 1]

    # Storing the cuisine names and scores
    cuisine_names = [c for c, _ in top_n_cuisines]
    cuisine_scores = [round(s,2) for _, s in top_n_cuisines]

    # dictionary with cuisine name and score for each of the top N closest matches
    closest = [{"id": data[data["cuisine"]==c]["id"].iloc[0], "score": round(s,2)} for c, s in top_n_cuisines[1:]]

    # Returning the cuisine name, scores and closest matches
    return cuisine_names, cuisine_scores, closest


