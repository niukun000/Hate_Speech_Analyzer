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
1. Split wth StratifiedKfold (preserving the percentage of samples for each class.)
2. Remove tags, non word charactors    #word#  @kun, ().,
3. tokenize sentence in to words
4. Convert words into lists of lower case tokens
5. Removing Stop words  (!!!delete whole sentence with label if this make sentence empty) 
6. Convert all tokens to a dictionary of unique word with frequence of occurancy as values (Bag-Of-Word feature extraction)
7. Removing word occurenvy less than 5
8. Reassign weight to values of each word with tf-idf
    tf dif: term frequency–inverse document frequency,


#### Methods
- [SKLean SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html): I use default parameters with Linear kernel.
- [MultimonialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Descision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


#### Results

                                 Macro average                                 Weight average

|DT   |  precision   | recall  |   f1-score  |  precision |  recall  |   f1-score |
|--------|---------|---------|-----------|-------------|--------|---------|
|LinearSVC |  0.74,     |   0.68,    |  0.69,     |  0.88,     |   0.9,   |     0.88 |
logistic regression  | 0.73, | 0.75,  |0.73, | 0.89, | 0.90,  |0.89 |
MultinomialNB | 0.72,| 0.64,| 0.67,| 0.85,| 0.85 | 0.81|
DecisionTreeClassifier |0.68,| 0.67, |0.67, |0.87, |0.88, |0.87|
RandomForestClassifier| 0.76,| 0.6,| 0.62,| 0.89,| 0.90,| 0.89


      
      

#### Discusion
- We observe that using LinearSVC with unbalanced dataset every score seems very high except scores for hate speech
- We observed that by using LinearSVC with under sampling, all scores seems reaonable, how ever upsampling should be a better choice due to small amount of data
- We observed that by applying upsampling before split, the results are overfitted, even crossvaladation cant figure out, the reason is that every minor class have too many duplicates so that many of the test cases are being seen in the training set already
- We observed that using upsampling improve detection of hate speech (recall of hate speech improve from 0.x to 0.y).
- We observee that by using LinearSVC with upsampling data before split, there is not much difference to unbalanced data

