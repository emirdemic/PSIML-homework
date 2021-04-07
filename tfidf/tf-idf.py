import os 
import numpy as np
from collections import Counter, OrderedDict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer 
import sys
sys.stdout.reconfigure(encoding='utf-8')


def get_document_n(directory_path):
	'''
	Returns the number of txt files in all directories.
	
	Args:
		directory_path (str): path to the folder with subdirectories and txt files
	Returns:
		n (int): number of txt files in all subdirectories of root folder
	'''
	n = 0
	for root, directiories, files in os.walk(directory_path):
		for file in files:
			if file.endswith('txt'):
				n += 1 

	return n


def stem_words(file):
	'''
	Function tokenizes words from a string and takes gets their stems.
	Applied only on alphanumerical characters, other characters are skipped.

	Args:
		file (str)
	Returns:
		list_to_return (list): list with stem terms
	'''
	stemmer = SnowballStemmer('english')
	list_to_return = []
	tokenized = word_tokenize(file)
	for word in tokenized:
		if word.isalnum():
			list_to_return.append(stemmer.stem(word))

	return list_to_return 


def get_frequency(stemmed_list):
	'''
	Calculates the word frequency for all words in one document and
	stores them into a Counter object.

	Args:
		stemmed_file(list): a list of strings
	Returns:
		frequency(dict): dictionary with words as keys and number of occurrences as values
	'''
	frequency = Counter(stemmed_list)

	return frequency


def get_idf(frequencies, directory_path):
	'''
	Given the dictionary with word frequencies for each document,
	function calculates the inverse document frequency metric for 
	each word. Function also takes directory_path input to the directory with 
	files and passes it to get_document_n function in order to get the number
	of documents.

	Args:
		frequencies (dict)
		directory_path (str)
	Returns:
		idf (Counter)
	'''

	document_n = get_document_n(directory_path)
	words = []
	for document in frequencies:
		for word in frequencies[document]:
			words.append(word)
	idf = Counter(words)
	for key in idf:
		idf[key] = np.log(document_n / idf[key])

	return idf 


def get_tfidf(frequencies, idf, txt_path):
	'''
	Function calculates TF-IDF score for each word 
	in a specific document.

	Args:
		frequencies
		idf
		txt_path
	Returns:
		document_tfidf
	'''

	tfidf = frequencies.copy()
	document_tfidf = tfidf[txt_path.lower()]
	for word in document_tfidf:
		document_tfidf[word] = document_tfidf[word] * idf[word]

	return document_tfidf


def top_10_words(tfidf):
	'''
	Given TF-IDF scores of words in a particular document, function
	calculates top 10 words with the highest TF-IDF scores by descending order.
	If scores are tied, those words are ordered lexicographically.

	Args:
		tfidf (dict)
	Returns:
		top_words (list)
	'''
	final_file = tfidf.copy()

	if len(final_file) <= 10:
		top_words = sorted(final_file, key = lambda x: (-final_file[x], x))
		return top_words

	elif len(final_file) > 10:
		top_words = sorted(final_file, key = lambda x: (-final_file[x], x))[:10] 
		return top_words


def sentence_summary(txt_path, tfidf):
	'''
	Given a specific document, function determines top 5 sentences by calculating the sum of 
	TF-IDF scores of the words sentences were formed with. Function returns top 5 sentences in 
	the order of their appearance. Function also runs top_10_words function in order
	to get top 10 most important words.

	Args:
		txt_path (str): path to the specific document
		tfidf (dict): dictionary with words as keys and TF-IDF scores as values
	Returns:
		top_words (str): top 10 words, comma separated
		top5_sentences (str): top 5 sentences, separated by their original punctuation
	'''
	read_file = open(txt_path, 'r', encoding = 'utf-8').read()
	sentence_list = sent_tokenize(read_file)
	top_words = top_10_words(tfidf)

	if len(sentence_list) <= 5:
		top_sentences = ' '.join(sentence_list)
		top_words = ', '.join(top_words)
		return top_words, top_sentences

	else:
		sentence_tfidf = []
		for sentence in sentence_list:
			word_tfidf = []
			tokenized_words = stem_words(sentence)

			for word in tokenized_words:
				word_tfidf.append(tfidf[word])
			word_tfidf = np.array(word_tfidf)

			if len(word_tfidf) <= 10:
				sentence_tfidf.append(np.sum(word_tfidf))
			elif len(word_tfidf) > 10:
				sentence_tfidf.append(np.sum(np.sort(word_tfidf)[-10:]))

		sentence_tfidf = np.array(sentence_tfidf)
	top5_indexes = np.sort((-sentence_tfidf).argsort(kind='mergesort')[:5])
	top5_sentences = np.array(sentence_list)[top5_indexes]

	top_words = ', '.join(top_words)
	top5_sentences = ' '.join(top5_sentences)

	return top_words, top5_sentences


def run_program(directory_path, txt_path):
	'''
	Function which runs all functions in adequate order and returns 
	top 10 words and top 5 sentences.

	Args:
		directory_path (str): path to the folder with documents and subdirectories
		txt_path (str): path to the txt file which needs to be analyzed
	Returns:
		words (str): top 10 words, comma separated
		sentences (str): top 5 sentences separated by their original punctuation  
	'''
	file_frequencies = {}

	for root, directiories, files in os.walk(directory_path):
		for file in files:
			if file.endswith('.txt'):
				filename = os.path.join(root, file)
				read_file = open(filename, 'r', encoding = 'utf-8').read()

				stemmed = stem_words(read_file)
				frequency = get_frequency(stemmed)
				file_frequencies[filename.lower()] = frequency 

	idf = get_idf(file_frequencies, directory_path)
	tfidf = get_tfidf(file_frequencies, idf, txt_path)
	words, sentences = sentence_summary(txt_path, tfidf)
	return words, sentences 


if __name__ == '__main__':
	directory_path = input()
	txt_path = input()
	words, sentences = run_program(directory_path, txt_path)
	print(words)
	print(sentences)