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
      
  Results for upsampling resampling 
    
                    precision    recall  f1-score   support

       hate_speech       0.54      0.13      0.21       286
         offensive       0.93      0.96      0.94      3838
           neither       0.83      0.90      0.86       832

          accuracy                           0.90      4956
         macro avg       0.76      0.66      0.67      4956
      weighted avg       0.89      0.90      0.89      4956
      
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

                                 Macro                                 Micro

|DT   |  precision   | recall  |   f1-score  |  precision |  recall  |   f1-score |
|--------|---------|---------|-----------|-------------|--------|---------|
|LinearSVC |  0.74,     |   0.68,    |  0.69,     |  0.9,     |   0.9,   |     0.9 |
logistic regression  | 0.73, | 0.75,  |0.73, | 0.89, | 0.89,  |0.89 |
MultinomialNB | 0.72,| 0.64,| 0.67,| 0.86,| 0.86 | 0.86|
DecisionTreeClassifier |0.68,| 0.67, |0.67, |0.87, |0.87, |0.87|
RandomForestClassifier| 0.76,| 0.6,| 0.62,| 0.88,| 0.88,| 0.88


      
      

#### Discusion
- We observe that using LinearSVC with unbalanced dataset every score seems very high except scores for hate speech
- We observed that by using LinearSVC with under sampling, all scores seems reaonable, how ever upsampling should be a better choice due to small amount of data
- We observed that by applying upsampling before split, the results are overfitted, even crossvaladation cant figure out, the reason is that every minor class have too many duplicates so that many of the test cases are being seen in the training set already
- We observed that using upsampling improve detection of hate speech (recall of hate speech improve from 0.x to 0.y).
- We observee that by using LinearSVC with upsampling data before split, there is not much difference to unbalanced data

