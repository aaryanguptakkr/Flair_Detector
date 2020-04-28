There are three major parts to this software :-

Data Collection

Pre-Processing &amp; EDA

Model building &amp; training

## Data Collection

There were two available APIs to access the reddit database.

1. **Praw** - Praw is the de-facto library used to access small amounts of data using a quick and easy setup. It doesn&#39;t provide access to more than 1000 records though.

1. **Pushshift** - Pushshift is another bulk api available to access redshift datawarehouse. Being a datawarehouse , it allows us to access a large amount of data at a time, but the data is not realtime and can be delayed by a few hours.

**Technique chosen - Pushshift**. (Since we could afford a delay of a few hours for data to be available we chose pushshift).

## Data Pre-Processing

The data from the reddit database was natural language based. This data required preliminary cleanup to reduce dimensionality of the model and provide accurate results.

We used the following techniques to reduce noise in inputs.

### Remove Stop Words

Our data had a large amount of words like &quot;is/are/am/being/the&quot; which don&#39;t provide any useful information. These words, if left in the data corpus reduce the efficiency of the model by inclining towards these words because of their high frequency. These words were simply discarded from our inputs using nltk library.

### Remove Punctuations/Numbers/Emojis

Our data is full of symbols/numbers/punctuations which does not provide useful information for classification and textual processing . These were simply discarded using the custom regex and nltk library.

### Reduction to Root Words

We had to reduce words to their root words to process meaningful information only.

1. Lemmatization - It reduces the word to it&#39;s root word which is always meaningful. This technique understands word semantics in addition to it&#39;s syntax. Example - Studies will be reduced to study which is a real dictionary word, Good/Better/Best will be reduced to Good.
2. Stemming - It reduces the word to its root word by removing the suffix. The root word thus generated may or may not be meaningful. Example - Studies will be reduced to &quot;Studi&quot; which is meaningless

**Technique Chosen - Lemmatization** ( Lemmatization replaces root words with synonyms in addition to stemming reducing my data dimensionality further. Moreover, the words returned from lemmatization were meaningful, and allowed me to further use Stanford&#39;s glove embeddings in modelling (Explained in modelling section)

## Exploratory Data Analysis

### Too Many Class Labels

Problem of too many class labels is faced when the dataset has a lot of output labels . If not handled , The Accuracy of the model can suffer as the model would have to consider a lot of labels. Hence making it complex.

I faced a similar problem here :-

The dataset had 66 classes initially which reduced the efficiency of my model.

I took only 10 most frequent classes &amp; dropped the rest &amp; then concatenated those classes.

### Biased Data

Problem of biased data is faced when the proportion of different outputs is not comparable to each other . If the data is left unbalanced , the output would always be leaning towards the prominent class.

I faced a similar problem here :-

Around 60% of the data consisted of only 4 classes i.e. Coronavirus , Politics , Non-Political , CAA-NRC.

#### Solution :-

I undersampled the prominent classes &amp; oversampled the less prominent ones. I randomly selected 2000 data values from each class &amp; concatenated them to form the dataset.

## Model Training

I considered following two approaches :-

**Machine Learning Approach**

Several Models in Machine Learning works very efficiently on classification problems.

The one i considered are :-

Naive Bayes :-I considered it as the base model for this classification as it uses probabilistic approach &amp; handled each feature independently.

Linear SVM :- I considered it for its accuracy.SVM&#39;s accuracy is comparable to Neural networks because of its geometric approach &amp; the fact that it considers the relative dependencies of features.

Random Forest Classifier : - I considered it for explainability of features. RFC model weights different words according to their features and their impact on label classes.

I used Scikit Learn library for the above.

#### Problem faced :-

The accuracies of the model was comparable to each other( varied from 65% - 70% accuracy) and the fact that each model worked better on certain classes which could not be ignored on the basis of their accuracies on the whole corpus . There was high possibility that SVM , NB would work better on certain classes compared to RFC.

#### Solution :-

I built a hybrid model which compares the probability of each model&#39;s final result &amp; compares them to decide the final output.

Basically , figuring out which of the model is more confident about it&#39;s output &amp; considering its output.

**Deep Learning Approach**

Deep Learning Language Models generally work better on text as NLP requires a lot of processing and specific attention to specific parts of text.

A lot of approaches are possible here :-

Attention models , Bidirectional models etc which gives us a chance to improve our model upto to great extent.

I built a LSTM language model on pytorch framework using glove embeddings.

The model was built by using hidden layers &amp; linear layers. I used ReLu and Softmax as activations &amp; performed some dropouts to avoid over computation.

Adam Optimizer was preferred over SGD optimizer because of the fact that Adam model is more sensitive towards hyper parameters compared to SGD.

For calculating loss, Cross Entropy Loss was selected .

This loss function tries to minimize the distance between the predicted and actual probability distributions. Therefore results in better classification of data..

#### Problem :-

The Embedding needs a lots of data to be trained which was not possible here.

As the APIs available had their limitations .

#### Solution:-

I used GLOVE pre trained embeddings which is trained by Stanford NLP on millions of data values &amp; the results were saved as GLOVE files.

#### Problem:-

The accuracy of this model was not upto the mark

#### Solutions Tried :-

Tweaking Model i.e varying the parameters , hidden layer&#39;s size , Optimizers , Loss Functions, Learning Rates etc and calculating the accuracy &amp; considering the best score.

The results were not upto the mark.

#### Possible Reason can be :-

Shortage of data as these models requires a lot of computation.
