#this exercise is to show you how to build a Machine Learning algathrim with R: An Irresponsibly Fast Tutorial
#tutoral link http://will-stanton.com/machine-learning-with-r-an-irresponsibly-fast-tutorial/


install.packages("caret", dependencies = TRUE)
install.packages("randomForest")
install.packages("fields")
library(caret)   # load caret package 
library(randomForest) # Load Random Forest package
library(fields)

setwd("~/MSDS650")  # set working directoy 
rm(list=ls())     # clear global enviroment 
trainSet <- read.table("titanic_train.csv", sep = ",", header = TRUE)  # load training data - reads in the file “titanic_train.csv”, using the delimiter “,”, including the header row as the column names, and assigns it to the R object trainSet.
testSet <- read.table("titanic_test.csv", sep = ",", header = TRUE)  # load test data

head (trainSet)   # preview the first few rows of the test dataset
head (testSet)  # preview the first few rows of the test dataset

#this part show us how to pick the best possible picking the best features to use in the model. In machine learning, a “feature” is really just a variable or some sort of combination of variables (like the sum or product of two variables).
#the most straightforward way to pick verables (but by no means the only way) is to use crosstabs and conditional box plots.

# this setup a crosstab . the Crosstabs hows the interactions between two variables in a very easy to read way
# check the data set to determine which data to use. 
table(trainSet[,c("Survived", "Pclass")])  # cross tab survived and Pclass   # yes 
table(trainSet[,c("Survived", "Name")])  # cross tab survived and Name    # no
table(trainSet[,c("Survived", "Sex")])  # cross tab survived and Sex     # yes 
table(trainSet[,c("Survived", "Age")])  # cross tab survived and Age   # yes (No)
table(trainSet[,c("Survived", "SibSp")])  # cross tab survived and SibSp   # no 
table(trainSet[,c("Survived", "Parch")])  # cross tab survived and Parch  # no 
table(trainSet[,c("Survived", "Ticket")])  # cross tab survived and Ticket   # no 
table(trainSet[,c("Survived", "Fare")])  # cross tab survived and Fare      # no (yes)
table(trainSet[,c("Survived", "Cabin")])  # cross tab survived and Cabin   # possible
table(trainSet[,c("Survived", "Embarked")])  # cross tab survived and Embarked   # possible 

# Comparing Age and Survived using a box plot - Plots are often a better way to identify useful continuous variables 
# using a “conditional” box plots to compare the distribution of each continuous variable, conditioned on whether the passengers survived or not ('Survived' = 1 or 'Survived' = 0). 
bplot.xy(trainSet$Survived, trainSet$Age)  # box plot age and survived  
summary(trainSet$Age)
bplot.xy(trainSet$Survived, trainSet$Fare)
summary(trainSet$Fare)

# Convert Survived to Factor
trainSet$Survived <- factor(trainSet$Survived)
# Set a random seed (so you will get the same results as me)
set.seed(42)
# Train the model using a "random forest" algorithm
model <- train(Survived ~ Pclass + Sex + SibSp +   
                 Embarked + Parch + Fare, # Survived is a function of the variables we decided to include
               data = trainSet, # Use the trainSet dataframe as the training data
               method = "rf",# Use the "random forest" algorithm
               trControl = trainControl(method = "cv", # Use cross-validation
                                        number = 5) # Use 5 folds for cross-validation
               
)
model

#The value “mtry” is a hyperparameter of the random forest model that determines how many variables the model uses to split the trees.
#Caret automatically picks the value of the hyperparameter “mtry” that was the most accurate under cross validation. This approach is called using a “tuning grid” or a “grid search.”

testSet$Survived <- predict(model, newdata = testSet)
# may run into an error with missing data 
#Error in `$<-.data.frame`(`*tmp*`, "Survived", value = c(1L, 2L, 1L, 1L,  : 
#replacement has 417 rows, data has 418

summary(testSet)
#notice that fare has a missing value N/A - to fix this use and if statment to fill the N/A value with the mean 
#value (if an entry in the column “Fare” is NA, then replace it with the mean of the column (also removing the NA's when you take the mean). Otherwise, leave it the same.)

testSet$Fare <- ifelse(is.na(testSet$Fare), mean(testSet$Fare, na.rm = TRUE), testSet$Fare)  # replaces na with mean value 14.454 

submission <- testSet[,c("PassengerId", "Survived")]
write.table(submission, file = "submission.csv", col.names = TRUE, row.names = FALSE, sep = ",")
