from collections import OrderedDict
from collections import defaultdict
import collections
import os
import re

uniqueTermIdDictionary = OrderedDict()
classDictionary = OrderedDict()

def main():
    importFeatsToDictionary('out/feats.dic')
    importCategoriesToDictionary('files/classIDs.txt')
    convertTweetEntries('tweets/Tweets.14cat.train', 'out/feats.train')
    convertTweetEntries('tweets/Tweets.14cat.test', 'out/feats.test')

def importCategoriesToDictionary(pathToFile):
    """Reads class file and imports class and corresponding id to dictionary

    Parameters
    ----------
    pathToFile : String type
        The path leading to the file
    """
    with open(pathToFile, 'r') as file:
        for line in file:
            if line == "\n": # Skip empty lines
                continue
            category, catID = line.strip().split('\t')
            classDictionary[category] = catID

def convertTweetEntries(pathToInputFile, pathToOutputFile):
    """Reads file and converts it to termID format

    Parameters
    ----------
    pathToInputFile : String type
        The path leading to the input file
    pathToOutputFile : String type
        The path leading to the output file
    """
    global uniqueTermIdDictionary
    global classDictionary

    output = open(pathToOutputFile, 'w')

    with open(pathToInputFile, 'r') as file:
        for line in file:
            if line == "\n": # Skip empty lines
                continue
            tempString = ""
            tempSet = set()
            tweetID, tweet, category = line.strip().split("\t")
            tweet = removeLinks(tweet) # Remove links
            for term in tokenize(tweet):
                if term != '' and term != ' ':
                    if term not in uniqueTermIdDictionary: # If the term is not in the dictionary neglect it (test set)
                        continue
                    else:
                        tempSet.add(int(uniqueTermIdDictionary[term]))
            sortedSet = sorted(tempSet)
            output.write(str(classDictionary[category]) + ' ')
            for termID in sortedSet:
                output.write(str(termID) + ':1 ') # ':1 '.join(map(str, aset))
            output.write('#' + str(tweetID) + '\n')
    output.close()

def importFeatsToDictionary(pathToFile):
    """Reads the features dictionary from a file

    Parameters
    ----------
    pathToFile : String type
        The path leading to the file
    """
    global uniqueTermIdDictionary
    with open(pathToFile, 'r') as file:
        for line in file:
            if line == "\n": # Skip empty lines
                continue
            term, termID = line.strip().split("\t")
            uniqueTermIdDictionary[term] = termID

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
    # return re.split(r' ', string) # r stands for raw expression
    return re.split(r'(?!\&\b)\W+', string)

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

    with open(pathToFile, 'w') as output:
        for term, id in uniqueTermIdDictionary.iteritems():
            output.write('{}\t{}\n'.format(term, id))

main()
