import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, scorer
from sklearn.preprocessing import FunctionTransformer
def ch(text):
    x_t = pd.DataFrame()
    m = []
    m.append(text)
    x_t['content'] = m

    count_vect = CountVectorizer()
    x_te = count_vect.transform(x_t)
    x_test_tf = transformer.transform(x_te)

    classifer = joblib.load("injection_model.pkl")

    return classifer.predict(x_test_tf)

if __name__ == '__main__':
    print(ch('default'))