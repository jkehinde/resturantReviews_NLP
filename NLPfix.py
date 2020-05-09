# Natural Language Processing

# 1.Importing the libraries
import numpy as np
import seaborn as sns
import pandas as pd


# 2.Importing the dataset
# A good way of ignoring "" is by using the quoting = 3 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# 3.Cleaning the texts
# stopwords contains  a list of all irrelevant word (i.e. the, that, of etc)
# corpus is a collection of text 
import re  
import nltk  
nltk.download('stopwords')   
from nltk.corpus import stopwords 
# Stemming is when we take the root meaning of a word e.g hating, hater, I hated = Hate
# This will make our matrix easier to analyze and reduce the sparsity
from nltk.stem.porter import PorterStemmer 
corpus = [] 
for i in range(0, 1000):
    # This removes the numbers and leave us with just the letters.
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) 
    # Converts all word in review to lower case 
    review = review.lower() 
    # splits each review so it is now a list of strings 
    review = review.split() 
    # Gooes through all the words removing any that are in the stopwords list
    ps = PorterStemmer() 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Return our list back to a string. ' ' keeps the spaces
    review = ' '.join(review) 
    corpus.append(review)


# 5.Creating the Bag of Words model
# Bag of Words Model is a sparse matrix where each row is the review and each column is a unique 
# word from the reviews.
# Tokenization - The process of  creating columns for each unique word in our review 
from sklearn.feature_extraction.text import CountVectorizer 
# max_features keeps the words that appear the most and removes least frequent one
# max_feature reduces sparsity and increases precision, helping our machine make better predicitions
cv = CountVectorizer(max_features = 1500) 
X = cv.fit_transform(corpus).toarray() 
y = dataset.iloc[:, 1].values  


# 6.Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# 7.Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# 8.Predicting the Test set results
y_pred = classifier.predict(X_test)


# 9.Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# 10.Visualising Confusion Matrix using Seaborn 
group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Reds')



# 11.The Math behind our confusion matrix 
TP = cm[0,0] # 55 - True Positive
FP = cm[0,1] # 42 - False Positive
FN = cm[1,0] # 12 - False Negative 
TN = cm[1,1] # 91 - True Positive


# Accuracy Fuction of Confusion Matrix
def accuracy(confusion_matrix):
    diagonal_sum = TP + TN
    sum_of_all_elements = TP + FP + FN + TN
    return diagonal_sum / sum_of_all_elements 


# Precision Fuction of Confusion Matrix
def recall(confusion_matrix):
    return TP/(TP + FN)

# Recall Fuction of Confusion Matrix 
def precision(confusion_matrix):
    return TP/(TP + FN)

# One full output of the accuracy, recall, precision 
print("Accuracy:", accuracy(cm))
print("Recall:", recall(cm))
print("Precision:", precision(cm))

