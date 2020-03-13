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
1. Split with StratifiedKfold (preserving the percentage of samples for each class.)
2. Remove tags, non-word characters    #word#  @kun, ().,
3. tokenize sentence into words
4. Convert words into lists of lower case tokens
5. Removing English Stop words  
6. Convert all tokens to a dictionary of unique word with frequency of occurrence as values (Bag-Of-Word feature extraction)
7. Removing word occurrence less than 5
8. Reassign weight to values of each word with tf-idf
    tf dif: term frequency-inverse document frequency,
9. Split data Use StratifiedKFold with 5 splits

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
RandomForestClassifier|  0.76,| 0.6,| 0.62,| 0.89,| 0.90,| 0.89|


      
      

#### Discusion
- We observe that RandomForestClassifier has highest macro Recall and Weight average for all precision recall and f1, and logistic regression has best overall scores


### Deep learning methods

#### Dara Preprocessing
1. Remove tags 
2. Replace newline and tabs with space
3. add missing space after full stops and commas
4. Tokenize sentence into individual words
5. Change words into lower cases
6. Remove punctuations none word charactors
7. Remove stop words
8. Change :nt" to "not"
9. lemmatize words
9. Embedding words
10. Split data Use StratifiedKFold with 5 splits

#### Methods
1, CNN
2, LSTM
3, CNN + LSTM
4, CNN+LSTM gloveEmbedding
5, LSTM + glove + attention
6, CNN + glove + attention

#### Results

                                 Macro average                                 Weight average

|DT   |  precision   | recall  |   f1-score  |  precision |  recall  |   f1-score |
|--------|---------|---------|-----------|-------------|--------|---------|
|LSTM| 0.67   |   0.68   |   0.67 | 0.87 |     0.86   |   0.86|
|CNN  |0.71   |   0.60   |   0.63| 0.85  |    0.85 |     0.85|
|CNN+LSTM  | 0.64| 0.68| 0.68| 0.86 |0.84|0.85|
|CNN+LSTM glove|0.62|0.53|0.56|0.81|0.82|0.80|

                                 Macro average                                 Weight average

|DT   |  precision   | recall  |   f1-score  |  precision |  recall  |   f1-score |
|--------|---------|---------|-----------|-------------|--------|---------|
|LSTM| 0.75   |   0.69   |   0.70 | 0.89 |     0.90   |   0.89|
|CNN  |0.77   |   0.67   |   0.69| 0.89  |    0.90 |     0.89|


