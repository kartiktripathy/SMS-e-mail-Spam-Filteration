'''SMS/e-mail filteration using Natural Language Processing'''

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('SPAM text message 20170820 - Data.csv')
ds = pd.DataFrame(dataset)

#Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0,5572):
    msg = re.sub('[^a-zA-Z]',' ',dataset['Message'][i])
    msg = msg.lower() #to convert everything to lowercase
    msg = msg.split() #to split the words of a particular review as elements of a list
    ps = PorterStemmer() #creating an object of the Porter Stemmer Class
    all_stopwords = stopwords.words('english')#storing all the stopwords of eng in a variable
    all_stopwords.remove('not')#removing 'not' from the list of stopwords ...so that they are not removed from the reviews
    msg = [ps.stem(word) for word in msg if not word in set(all_stopwords)]
    msg = ' '.join(msg)# we join all the stemmed words with a space between them
    corpus.append(msg) # add the cleaned review to the corpus list

#Creating the Bag OF Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,0].values

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
y=labelencoder_X.fit_transform(y)

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#Training XGBoost on the Training Set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

#Predicting the test Set Results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
print("Confusion Matrix :")
cm = confusion_matrix(y_test,y_pred)
print(cm)
ac = accuracy_score(y_test,y_pred)
print("Accuracy on the Test Set : ",ac*100,'%')

#Applying the K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

#Predicting if a single review is positive or negative
new_msg = input('Enter your message/e-mail :')
new_msg = re.sub('[^a-zA-Z]', ' ', new_msg)
new_msg = new_msg.lower()
new_msg = new_msg.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_msg = [ps.stem(word) for word in new_msg if not word in set(all_stopwords)]
new_msg = ' '.join(new_msg)
new_corpus = [new_msg]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
if(new_y_pred ==0):
    print("Thats not a spam. Dont lose it ! might be important .")
else:
    print("Its a spam .")