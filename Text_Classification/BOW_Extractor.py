"""BOW_Extractor.py : Module that extracts the BOW features from the training files"""
from collections import OrderedDict
import os
import re

uniqueTermIdDictionary = OrderedDict() # Dictionary that stores terms and corresponding ID
idEnumerator = 1 # Global enumerator to assign IDs to terms

def main():
    importTweetFileToDictionary('tweets/Tweets.14cat.train') # Import training file and preprocess
    exportUniqueTerms('tc_out/feats.dic') # Export unique terms and corresponding IDs to file

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
            tweet = removeLinks(tweet) # Remove links from text
            termsList = filter(None, tokenize(tweet))
            for term in termsList:
                if term not in uniqueTermIdDictionary:
                    uniqueTermIdDictionary[term] = idEnumerator
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
    return re.split(r'\W+', string) # Tokenize on any non alphabetical character

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

main()
