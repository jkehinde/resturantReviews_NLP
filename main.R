# Natural Language Processing

# Even though this is simple example of natural language processing. You can apply this on books 
# to find out which genre it is. you can do this on html pages that have had data scraped 
# from it and you can perform different analysis on  it.
# lastly we can do NLP on newpapers to see what category a paper belongs to.

# We use the tsv because the review could have columns in it and this will seperate 
# the data incorrectly 
# We use read.delim and it's default seperator is '\t'


# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)


# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review)) # creates a dataset just of review we will clean them step by step
corpus = tm_map(corpus, content_transformer(tolower)) # turns to lower case
corpus = tm_map(corpus, removeNumbers) # removes numbers
corpus = tm_map(corpus, removePunctuation) # removes punctuation
corpus = tm_map(corpus, removeWords, stopwords()) # removes stop words. i.e that them etc.
corpus = tm_map(corpus, stemDocument) # this gets us the root of a word. loving -> love 
corpus = tm_map(corpus, stripWhitespace) # removes the extra spaces cause by cleanings above 


# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus) # create one column for each word. check the size in enviroment
dtm = removeSparseTerms(dtm, 0.999) 
# The first argument in the removeSparseTerms function above is the sparse matrix, 
# the second parameter is 99.9 percent of the most frequent words in our matrix
dataset = as.data.frame(as.matrix(dtm)) # this turns our matrix to a dataframe 
dataset$Liked = dataset_original$Liked # this allows us to add a new column to this dataset
# we will give it the same name as the dependent variable column 


# for NLP, the algorithms that work best are Naive bayes, Decision tree classification and
# random forest classification.
# For this tutorial we are going to use random forest classification 


# Random forest classification model
# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1)) # change last columns to factor 


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8) # 80% training split 
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692], 
                          y = training_set$Liked, 
                          ntree = 10) 


# Analysing our confusion matrix 

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])  
y_pred # view our predicition  

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
cm # view our correct and incorrect review 

# Accuracy of the model 
accuracy = (cm[1,1]+cm[2,2])/sum(cm)
round(accuracy, 2)

# Precision of the model
precison = cm[1,1]/sum(cm[,1])
precison  

# Recall of the model
recall = cm[1,1]/sum(cm[1,])
recall
