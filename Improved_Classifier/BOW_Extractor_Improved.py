"""BOW_Extractor.py : Module that extracts the BOW features from the training files - improvements"""
from collections import OrderedDict
from nltk.stem import PorterStemmer # Porter Stemmer
from nltk.corpus import wordnet as wn
import lxml.html
from urllib2 import urlopen
import os
import re

uniqueTermIdDictionary = OrderedDict() # Dictionary that stores terms and corresponding ID
idEnumerator = 1 # Global enumerator to assign IDs to terms
stopwords = set() # Set with stopwords - O(1) search
porter = PorterStemmer()

def main():
    loadStopwords()
    importTweetFileToDictionary('tweets/Tweets.14cat.train') # Import training file and preprocess
    exportUniqueTerms('tc_out_improved/feats1.dic') # Export unique terms and corresponding IDs to file

def importTweetFileToDictionary(pathToFile):
    """Reads tweets train file and saves unique terms in dictionary structure with unique IDs

    Parameters
    ----------
    pathToFile : String type
        The path leading to the file
    """
    global uniqueTermIdDictionary
    global idEnumerator

    with open(pathToFile, 'r') as file:
        for line in file:
            if line == "\n": # Skip empty lines
                continue
            tweetID, tweet, category = line.strip().split("\t")
            for link in getLinks(tweet):
                linkTerms = filter(None, getLinkTitleTerms(link))
                for term in linkTerms:
                    if isNotAStopword(term) and term is not '' or term is not ' ':
                        stemmedTerm = stemWord(term)
                        if stemmedTerm not in uniqueTermIdDictionary:
                            uniqueTermIdDictionary[stemmedTerm] = idEnumerator
                            idEnumerator += 1
            tweet = removeLinks(tweet).lower() # Remove links from text and lower case
            termsList = filter(None, tokenize(tweet))
            for term in termsList:
                if isNotAStopword(term):
                    stemmedTerm = stemWord(term)
                    if stemmedTerm not in uniqueTermIdDictionary:
                        uniqueTermIdDictionary[stemmedTerm] = idEnumerator
                        idEnumerator += 1
                    synonyms = getSynonyms(term)
                    for synonym in synonyms:
                        if synonym not in uniqueTermIdDictionary:
                            uniqueTermIdDictionary[synonym] = idEnumerator
                            idEnumerator += 1

def tokenize(string):
    """Splits parameter 'string' on spaces and returns a list of the tokens.

    Parameters
    ----------
    string : String type
        A sentence to be split

    Returns
    -------
    tokens : List of strings
        A list containing all tokens
    """
    # return re.split(r'\s+', string) # Tokenize on any whitespace character - baseline tokenization
    return re.split(r'_|\W+', string) # Tokenize on any non alphabetical character

def removeLinks(text):
    """Removes links from given text - either http or https.

    Parameters
    ----------
    text : String type
        A text string which may contain links

    Returns
    -------
    withoutLinks : String type
        The given string without any links
    """
    return re.sub(r'https?:\/\/[^\s]+', '', text)

def getLinkTitleTerms(link):
    """Returns title list from given link - either http or https.

    Parameters
    ----------
    link : String type
        The website

    Returns
    -------
    titleTerms : List type
        Terms of webpage's titles
    """
    titleTerms = []
    try:
        tree = lxml.html.parse(urlopen(link, timeout = 1))
        titleText = tree.find('.//title').text
        titleTerms = tokenize(titleText.lower())
        return titleTerms
    except:
        return []

def getLinks(text):
    """Returns links from given text - either http or https.

    Parameters
    ----------
    text : String type
        A text string which may contain links

    Returns
    -------
    links : List type
        The links within the text
    """
    p = re.compile(r'https?:\/\/[^\s]+')
    links = p.findall(text)
    return links

def exportUniqueTerms(pathToFile):
    """Exports the unique terms with their id to a file

    Parameters
    ----------
    pathToFile : String type
        Path leading to the output file
    """
    global uniqueTermIdDictionary

    path = pathToFile.rsplit('/', 1)[0]
    if not os.path.exists(path): # Check whether the directory exists or not
        os.makedirs(path)
    # Write operations
    with open(pathToFile, 'w') as output:
        for term, id in uniqueTermIdDictionary.iteritems():
            output.write('{}\t{}\n'.format(term, id))

def stemWord(word):
    """Stems the given word using the Porter Stemmer library

    Parameters
    ----------
    word : String type
        A word to be stemmed

    Returns
    -------
    stemmedWord : String type
        The stemmed version of the given word
    """
    global porter
    return porter.stem(word)

def isNotAStopword(word):
    """Determines whether a word is a stopword

    Parameters
    ----------
    word : String type
        A word to be checked

    Returns
    -------
    isNotStopword : Boolean type
        Returns True if the given word is not a stopword, otherwise False
    """
    global stopwords
    if word in stopwords:
        return False
    return True

def loadStopwords():
    """Loads all stopword terms from file and saves them to a set structure
    """
    global stopwords
    with open('files/stopwords.txt') as stopWordFile:
        stopwords = set(stopWordFile.read().splitlines())

def getSynonyms(term):
    """Finds word synonyms and returns them

    Parameters
    ----------
    term : String type
        A word whose synonyms will be obtained

    Returns
    -------
    synonyms : List type
        List of synonyms
    """
    tempSet = set()
    for synset in wn.synsets(term):
        for word in synset.lemma_names():
            tempSet.add(word)
    return list(tempSet)

main()
