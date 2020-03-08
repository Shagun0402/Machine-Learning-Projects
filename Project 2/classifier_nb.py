"""
CSE 6363 Machine Learning Project 2: Naive Bayes Classification
Name: Shagun Paul
UTA ID: 1001557958
"""
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
import os
import time
import math
import nltk as nl
import numpy as np
import csv

class naive_bayes():
    def __init__(self):
        # Loads the labels of the categories
        self.categories = os.listdir('20_newsgroups')
        # Stores file paths for pre-processing data
        self.file_paths_dict = {}
        # Stores Prior Probabilities of text files in each class
        self.dict_prior_prob = {}
        # Stores occurences of words in sample files for each class
        self.common_words_freq = {}
        # Stores conditional probabilities of words in each class
        self.conditional_prob_words = {}

    # Compute prior probabilities
    def prior_probability(self, news_docu_number, document_total):
        for category, num_docs in news_docu_number.items():
            self.dict_prior_prob[category] = num_docs / document_total
        csv_open = csv.writer(open('prior_probability.csv', 'w'))
        csv_open.writerow(['Class', 'Probability of Docs for Class'])
        print("------------------------------------------------------------------------------------------------------")
        print("The Prior probabilities are : \n")
        for key,val in self.dict_prior_prob.items():
            print(key,":",val)
            csv_open.writerow([key, val])

    # Computes Prior Probabilities of Files based on their file path
    def load_files_prior_prob(self):
        tot_docs = 0
        news_doc_dict = {}
        #For number of files in each class of a category
        for k in range(len(self.categories)):
            self.file_paths_dict[self.categories[k]] = os.listdir('20_newsgroups/' + self.categories[k])
        # Calculating number of documents
        for category in self.categories:
            length_tmp= len(self.file_paths_dict[category])
            news_doc_dict[category] = length_tmp
            tot_docs += length_tmp
        print("------------------------------------------------------------------------------------------------------")
        print("Number of Documents in each category are: \n ")
        # Calculate the prior probability
        for key, value in news_doc_dict.items():
            print(key,":",value)
        self.prior_probability(news_doc_dict, tot_docs)

    # Fetches file to return text in it
    def fetch_file(self, category, fileName):
        path = '20_newsgroups/' + category + '/' + fileName
        fetched_file = open(path, 'rb')
        text = fetched_file.read()
        fetched_file.close()
        return str(text.decode("utf-8", 'ignore'))

    # Breaks down sentences to words
    def break_sentences(self, text):
        break_text = text.split("\n\n")
        sentence = "".join(break_text[1:])
        return sentence

    # Breaks down sentences to words and then removes stop words from sentences
    def text_cleaning(self, text):
        words = nl.word_tokenize(text)
        wor = [w for w in words if w.isalnum()]
        stop_words = set(nl.corpus.stopwords.words('english'))
        wor = [w.lower() for w in wor if not w in stop_words]
        return wor

    # Computes word occurences in a class sample for each category
    def compute_occurences(self, hist_word_list):
        frequency = {}
        for word_list in hist_word_list:
            for word in word_list:
                frequency[word] = frequency.get(word, 0) + 1
        return frequency

    # This function calculates the frequency of Tokens in each file according to each Class
    def calc_word_freq_cat(self, files_name_dict):
        word_dict = {}
        print("------------------------------------------------------------------------------------------------------")
        print(" Loading and Pre-Processing Data.... PLease Wait. \n \n")
        for category in self.categories:
            files = files_name_dict[category]
            collective_words = []
            """
            This fragment retrieves text from file and then breaks down the given text into sentences.
            The sentences are then broken dwn to words and the stop words are removed from the text.
            The retrieved word list is then stored according to each class in each news category.
            """
            for fileName in files:
                text = self.fetch_file(category, fileName)
                sentences = self.break_sentences(text)
                filtered_words = self.text_cleaning(sentences)
                collective_words.append(filtered_words)
            word_dict[category] = collective_words
        print("------------------------------------------------------------------------------------------------------")
        print(" Success! Pre-processing and Loading is now Completed! \n \n")
        return word_dict

   # Compute Word Occurences
    def calc_words_occurences(self, class_words_dict):
        for category in self.categories:
            self.common_words_freq[category] = self.compute_occurences(class_words_dict[category])

    # Calculate total number of words
    def calc_total_words(self):
        words_per_document = 0
        total_words = {}
        for key, val in self.common_words_freq.items():
            for words, freq in val.items():
                words_per_document += freq
            total_words[key] = words_per_document
            words_per_document = 0
        return total_words

    # Evaluates Conditional Probabilities for each word present in each class
    def compute_conditional_probab(self):
        total_words = self.calc_total_words()
        for category, words in self.common_words_freq.items():
            temp_word_dict = {}
            for word, freq in words.items():
                temp_word_dict[word] = freq / total_words[category]
            self.conditional_prob_words[category] = temp_word_dict

    def write_to_file(self):
        word_probabilities = csv.writer(open('word_probabilities.csv', 'w'))
        word_probabilities.writerow(['Category', 'Words', 'Probability'])
        for category, values in self.conditional_prob_words.items():
            word_probabilities.writerow([category])
            for word, prob in values.items():
                word_probabilities.writerow(['',word,prob])

    def trainClassifier(self):
        self.load_files_prior_prob()
        print("------------------------------------------------------------------------------------------------------")
        print(" \n The Categories found are: \n")
        print(self.categories)

        start_time = time.time()
        print("------------------------------------------------------------------------------------------------------")
        print("Loading training data. Please Wait...................")
        word_dict = self.calc_word_freq_cat(self.file_paths_dict)
        end_time = time.time()
        total_time = end_time - start_time
        print("Wait time to load is: %s" %(total_time / 60))
        # compute word frequency
        self.calc_words_occurences(word_dict)
        # compute conditional probability
        self.compute_conditional_probab()
        self.write_to_file()
        print("------------------------------------------------------------------------------------------------------")
        print("Success! The Model is trained using Training Data")

    # Loads Sample Data
    def load_sample_data(self):
        sample_dict = {}
        for k in range(len(self.categories)):
            sample_dict[self.categories[k]] = os.listdir('mini_newsgroups/' + self.categories[k])
        return self.calc_word_freq_cat(sample_dict)

    # Finds probability of each word in a document.Omits the probability of words that are not present in document.
    def predict_cond_class(self, document):
        probab_maximum = 0
        cat_predicted = ""
        for category, cat_prob in self.dict_prior_prob.items():
            total_cond_prob = abs(math.log(cat_prob))
            for word in document:
                prob = self.conditional_prob_words[category].get(word, 0)
                if prob != 0:
                    total_cond_prob += abs(math.log(prob))
                else:
                    total_cond_prob += prob
            if total_cond_prob > probab_maximum:
                probab_maximum = total_cond_prob
                cat_predicted = category
        return cat_predicted

    # Predicts test data
    def predict(self):
        sample_word_dict = self.load_sample_data()
        actual_predicted_values = []
        predicted_classes = []
        actual_classes = []

        for category, doc_list in sample_word_dict.items():
            for doc in doc_list:
                predicted_class = self.predict_cond_class(doc)
                predicted_classes.append(predicted_class)
                actual_classes.append(category)
                if predicted_class == category:
                    actual_predicted_values.append(0)
                else:
                    actual_predicted_values.append(1)
        csv_open = csv.writer(open('actual_predicted_values.csv', 'w'))
        csv_open.writerow(['Predicted Values','Actual Values'])
        for j in range(len(predicted_classes)):
            csv_open.writerow([predicted_classes[j], actual_classes[j]])
        print("------------------------------------------------------------------------------------------------------")
        print("Predicted Classes of Sample Data are: \n")
        print(predicted_classes)
        print("\n\n")
        print("The Actual Classes of Sample Data are: \n")
        print(actual_classes)
        return np.array(actual_predicted_values)

    def compute_accuracy(self):
        actual_predicted_values = self.predict()
        count_1s = np.count_nonzero(np.array(actual_predicted_values))
        count_0s = 2000 - count_1s
        print("------------------------------------------------------------------------------------------------------")
        print("Accuracy of Predicted Sample Data is: ", (count_0s / 2000) * 100, "%")

if __name__ == '__main__':
    classify = naive_bayes()
    # Trains sample data
    classify.trainClassifier()
    # Finds Accuracy of classifier
    classify.compute_accuracy()
