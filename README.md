# Hate_Speech_Analyzer

Niu Kun's CMPT400 project.

### Dataset
1. [labeled_data.csv](https://github.com/t-davidson/hate-speech-and-offensive-language): From "Davidson T, Warmsley D, Macy MW, Weber I. Automated Hate Speech Detection and the                                                        Problem of Offensive Language. ICWSM. 2017;.". 
2. [Twitter_DT](https://www.google.com)

|Dataset | #Tweets | Classe (#tweets)| Targeting characteristics|
|--------|---------|-----------------|--------------------------|
|Davidson, T| 24783|hatespeech(1430) offenssivelanguage(19190)  neither(4163) |hatesppech and offenssivelanguage|

### Traditional Methods

#### Data Pre-processing
Steps that I took:
1. Tokenize sentence  
2. Remove tags, non word charactors    #word#  @kun, ().,
3. Convert words into lists of lower case tokens
4. Removing Stop words  (!!!delete whole sentence with label if this make sentence empty)
5. Lemmatizing word (transfore word with close meanings to same base word, eg, learning to learn, well to good)
6. Convert all tokens to a dictionary of unique word with frequence of occurancy as values (Bag-Of-Word feature extraction)
6. Removing word occurenvy less than 5
7. Reassign weith to values of each word with tf-idf
    tf dif: term frequency–inverse document frequency,


#### Methods
- [SKLean SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html): I use default parameters with Linear kernel.
- [MultimonialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Descision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


#### Results
Results for unbalanced vs upsampling vs downsampling experiment. The experiment is done using LinearSVC.

unbalanced data

Accuracy of LinearSVC= 0.898930804922332

                    precision    recall  f1-score   support

       hate_speech       0.52      0.27      0.36       296
         offensive       0.93      0.96      0.94      3853
           neither       0.84      0.86      0.85       808

          accuracy                           0.90      4957
         macro avg       0.76      0.70      0.72      4957
      weighted avg       0.89      0.90      0.89      4957
      cross validation score [0.88521283 0.88985273 0.89933427 0.90334948 0.89830508]
      
upsampling data

Accuracy of LinearSVC= 0.9743790168490534

                    precision    recall  f1-score   support

       hate_speech       0.96      1.00      0.98      3887
         offensive       1.00      0.92      0.96      3795
           neither       0.97      1.00      0.98      3832

          accuracy                           0.97     11514
         macro avg       0.98      0.97      0.97     11514
      weighted avg       0.98      0.97      0.97     11514
      cross validation score [0.96673615 0.96690985 0.98254299 0.97351051 0.97985062]
      
downsampling data

Accuracy of LinearSVC= 0.8193473193473193

                    precision    recall  f1-score   support

       hate_speech       0.84      0.68      0.75       297
         offensive       0.77      0.86      0.81       279
           neither       0.86      0.93      0.89       282

          accuracy                           0.82       858
         macro avg       0.82      0.82      0.82       858
      weighted avg       0.82      0.82      0.82       858
      cross validation score [0.78438228 0.81585082 0.79137529 0.74358974 0.74358974]

Overall results comparing different methods. All methods are using upsampling.

results for MultimonialNB

Accuracy of MultinomialNB= 0.9402466562445718

                    precision    recall  f1-score   support

       hate_speech       0.88      1.00      0.94      3801
         offensive       0.99      0.83      0.91      3860
           neither       0.96      0.99      0.97      3853

          accuracy                           0.94     11514
         macro avg       0.95      0.94      0.94     11514
      weighted avg       0.95      0.94      0.94     11514
      cross validation score [0.9235713  0.92496092 0.94120201 0.93859649 0.93590412]

results fore LogisticRegression

Accuracy of logistic regression= 0.9543164842800069

                    precision    recall  f1-score   support

     hate_speech       0.93      0.99      0.96      3788
       offensive       0.99      0.88      0.93      3833
         neither       0.95      0.99      0.97      3893

        accuracy                           0.95     11514
       macro avg       0.96      0.95      0.95     11514
    weighted avg       0.96      0.95      0.95     11514
    cross validation score [0.94102831 0.94311273 0.96525968 0.95935383 0.95961438]

Results for DecisionTree

Accuracy of DecisionTreeClassifier= 0.9535348271669272

                    precision    recall  f1-score   support

       hate_speech       0.90      1.00      0.95      3801
         offensive       1.00      0.86      0.93      3860
           neither       0.97      1.00      0.98      3853

          accuracy                           0.95     11514
         macro avg       0.96      0.95      0.95     11514
      weighted avg       0.96      0.95      0.95     11514
    cross validation score [0.96030919 0.96769151 0.97550808 0.97238145 0.97403161]

Results for RandomForest

Accuracy of RandomForestClassifier= 0.9773319437206879

                    precision    recall  f1-score   support

       hate_speech       0.97      1.00      0.98      3801
         offensive       0.99      0.94      0.97      3860
           neither       0.97      0.99      0.98      3853

          accuracy                           0.98     11514
         macro avg       0.98      0.98      0.98     11514
      weighted avg       0.98      0.98      0.98     11514
      cross validation score [0.97090499 0.97802675 0.98584332 0.97967692 0.98280354]

#### Discusion
- We observed that using upsampling improve detection of hate speech (recall of hate speech improve from 0.x to 0.y).

