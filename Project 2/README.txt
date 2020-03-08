Bayesian learning for classifying netnews text articles
------------------------------------------------------------------------------------------------------------------------------
Project Deliverables:
Naive Bayes classifiers are among the most successful known algorithms for learning to classify
text documents. We will provide a dataset containing 20,000 newsgroup messages drawn from
the 20 newsgroups. The dataset contains 1000 documents from each of the 20 newsgroups.

1. Please download the data from http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naivebayes.
html (Newsgroup Data)
2. Please use half data as training data, and the other half as testing data.
3. Please implement the Naive Bayes classifier by yourself.
--------------------------------------------------------------------------------------------------------------------------
In order to run the project, you will require to have the following files in the same folder:
(a) classifier_nb.py
(b) 20_newsgroups
(c) mini_newsgroups
(d) libraries.txt
(e) prior_probabilities.csv
(f) word_probabilities.csv
(g) actual_predicted_values.csv
------------------------------------------------------------------------------------------
Results are stored in csv files mentioned below
(a) prior_probabilities.csv: Comprises of probabilities of documents in each category of newsgroup.
(b) word_probabilities.csv: Stores probabilities of words in each class for each news group category.
(c) actual_predicted_values.csv: The final results of actual and predicted value classes.
------------------------------------------------------------------------------------------
Running the Program:

(1) Navigate to the folder in which the above mentioned files are present
(2) Download dependent libraries using:
	* [ pip install -r libraries.txt]
(3) Running the program using:
	*[ python classifier_nb.py]

Note that it will take a while to load all files and pre-process them.
