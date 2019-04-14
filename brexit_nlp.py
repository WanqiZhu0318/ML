from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np

df = pd.read_csv("brexit.csv")
df.describe

import numpy as np
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report 
X= df["title"]
y = df["EUR/GBP"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, 
random_state=0)

tfid_vec = TfidfVectorizer()
x_tfid_train = tfid_vec.fit_transform(X_train)
x_tfid_test = tfid_vec.transform(X_test)

mnb_tfid = MultinomialNB()
mnb_tfid.fit(x_tfid_train, y_train)
mnb_tfid_y_predict = mnb_tfid.predict(x_tfid_test)
print(mnb_tfid_y_predict)
print("TfidVectorizer model accuracy：", mnb_tfid.score(x_tfid_test, y_test))
print("classification_report", classification_report(mnb_tfid_y_predict, y_test))