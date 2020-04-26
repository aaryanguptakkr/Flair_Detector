# Flair_Detector
<h4> The model of comprises of mainly 4 components :-
  <p> Data Collection
  <p> Data Preprocessing
  <p> Exploratory Design Analysis
  <p> Model Training
 <h4> The repo is Comprised of 4 notebooks:-
   <h5> Reddit_Scrapper_Trainer :-
     <p> It comprises of all the other notebooks. Both of the approaches are trained and explained here.The following parts of the project are present here :- Data Collection , EDA , Preprocessing , Training of Models , Evaluating them.
   <h5> preprocessing :- 
     <p> All the functions used for preprocessing are defined here.
   <h5> rscaper
     <p> It loads the pretrained models & define the function to give the final output
   <h5> predictor :-
     <p> It takes the input & gives the ouput on the model trained so far , 

<h2>
  Data Collection
  
  <p>
  There were two available APIs to access the reddit database.

<h4>Praw-
  <p>Praw is the de-facto library used to access small amounts of data using a quick and easy setup. It doesn’t provide access to more than 1000 records though.
<h4>Pushshift
  <p>Pushshift is another bulk api available to access redshift datawarehouse. Being a datawarehouse , it allows us to access a large amount of data at a time, but the data is not realtime and can be delayed by a few hours.
    
<h4>
Technique chosen
  <p> Pushshift. (Since we could afford a delay of a few hours for data to be available we chose pushshift).

  <h2> Data Pre-Processing
  <p> The data from the reddit database was natural language based. This data required preliminary cleanup to reduce dimensionality of the model and provide accurate results.
We used the following techniques to reduce noise in inputs.

<h3> Remove Stop Words
  <p> Our data had a large amount of words like “is/are/am/being/the” which don’t provide any useful information. These words, if left in the data corpus reduce the efficiency of the model by inclining towards these words because of their high frequency. These words were simply discarded from our inputs using nltk library.

<h3>Remove Punctuations/Numbers/Emojis
  <p> Our data is full of symbols/numbers/punctuations which does not provide useful information for classification and textual processing . These were simply discarded using the custom regex and nltk library.
    
<h3>Reduction to Root Words
  <p> We had to reduce words to their root words to process meaningful information only.
  <h4> Lemmatization
    <p>  It reduces the word to it’s root word which is always meaningful. This technique understands word semantics in addition to it’s syntax. Example - Studies will be reduced to study which is a real dictionary word, Good/Better/Best will be reduced to Good.
  <h4> Stemming
    <p> It reduces the word to its root word by removing the suffix. The root word thus generated may or may not be meaningful. Example - Studies will be reduced to “Studi” which is meaningless

<h4> Technique Chosen - Lemmatization
  <p> Lemmatization replaces root words with synonyms in addition to stemming reducing my data dimensionality further. Moreover, the words returned from lemmatization were meaningful, and allowed me to further use Stanford’s glove embeddings in modelling (Explained in modelling section)
    
  <h2>Exploratory Data Analysis
  <h3>Too Many Class Labels
    <p>Problem of too many class labels is faced when the dataset has a lot of output labels . If not handled , The Accuracy of the model can suffer as the model would have to consider a lot of labels. Hence making it complex. 
I faced a similar problem here :-
The dataset had 66 classes initially which reduced the efficiency of my model.
  <h4> Solution
<p> I took only 10 most frequent classes & dropped the rest & then concatenated those classes.

<h3> Biased Data
  <p>Problem of biased data is faced when the proportion of different outputs is not comparable to each other . If the data is left unbalanced , the output would always be leaning towards the prominent class.
I faced a similar problem here :-
Around 60% of the data consisted of only 4 classes i.e. Coronavirus , Politics , Non-Political , CAA-NRC. 
<h4>Solution :- 
<p>I undersampled the prominent classes & oversampled the less prominent ones. I randomly selected 2000 data values from each class & concatenated them to form the dataset.
  
  <h2>Model Training
  <p>I considered following two approaches  :-
    <h3>Machine Learning Approach
      <p> Several Models in Machine Learning works very efficiently on classification problems.
The one i considered are :- 
Naive Bayes :-I considered it as the base model for this classification as it uses probabilistic approach & handled each feature independently.
Linear SVM :- I considered it for its accuracy.SVM’s accuracy is comparable to Neural networks because of its geometric approach & the fact that it considers the relative dependencies of features.
Random Forest Classifier : - I considered it for explainability of features. RFC model weights different words according to their features and their impact on label classes.
I used Scikit Learn library for the above.
<h5>Problem faced :-
  <p>The accuracies of the model was comparable to each other( varied from 65% - 70% accuracy) and the fact that each model worked better on certain classes which could not be ignored on the basis of their accuracies on the whole corpus . There was high possibility that SVM , NB would work better on certain classes compared to RFC.
<h5> Solution :-
  <p> I built a hybrid model which compares the probability of each model’s final result & compares them to decide the final output. Basically , figuring out which of the model is more confident about it’s output & considering its output.

<h3>Deep Learning Approach
  <p> Deep Learning Language Models generally work better on text as NLP requires a lot of processing and specific attention to specific parts of text.
A lot of approaches are possible here :-
Attention models , Bidirectional models etc which gives us a chance to improve our model upto to great extent. 
<p>I built a LSTM language model on pytorch framework using glove embeddings.
The model was built by using hidden layers & linear layers. I used ReLu and Softmax as activations & performed some dropouts to avoid over computation. 

<p>Adam Optimizer was preferred over SGD optimizer because of the fact that Adam model is more sensitive towards hyper parameters compared to SGD.

<p>For calculating loss, Cross Entropy Loss was selected . This loss function tries to minimize the distance between the predicted and actual probability distributions. Therefore results in better classification of data.
  
<h4> Problem :-
  <p>The Embedding needs a lots of data to be trained which was not possible here. As the APIs available had their limitations .
<h4> Solution:-
  <p> I used GLOVE pre trained embeddings which is trained by Stanford NLP on millions of data values & the results were saved as GLOVE files.
 <h4> Problem:-
   <p> The accuracy of this model was not upto the mark 
 <h4> Solutions Tried :-
   <p> Tweaking Model i.e  varying the parameters , hidden layer’s size , Optimizers , Loss Functions, Learning Rates etc  and calculating the accuracy & considering the best score.
The results were not upto the mark.
  <h4> Possible Reason can be :- 
    <p> Shortage of data as these models requires a lot of computation.


