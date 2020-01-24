# Hate_Speech_Analyzer

Niu Kun's CMPT400 project.
kun edited this

File Name: imbalanced.py
methods: pipline, LinearSVC
results: results for unbalanced LinearSVC


### Dataset
1. [labeled_data.csv](https://github.com/t-davidson/hate-speech-and-offensive-language): From "Davidson T, Warmsley D, Macy MW, Weber I. Automated Hate Speech Detection and the                                                        Problem of Offensive Language. ICWSM. 2017;.". 
2. [Twitter_DT](https://www.google.com)

|Dataset | #Tweets | Classe (#tweets)| Targeting characteristics|
|--------|---------|-----------------|--------------------------|
|Davidson, T| |hatespeech  offenssivelanguage  neither|hatesppech and offenssivelanguage|

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
- MultimonialNB:(https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)
- Logistic Regression: (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Descision Tree: (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- Random Forest:(https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


#### Results
Results for upsampling vs downsampling experiment. The experiment is done using SVC.
- table 1

Overall results comparing different methods. All methods are using upsampling.
- table 2


#### Discusion
- We observed that using upsampling improve detection of hate speech (recall of hate speech improve from 0.x to 0.y).

