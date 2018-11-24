"""Feature_Converter.py : Module that converts both tweet train & test files to feats files"""
from BOW_Extractor_Improved import tokenize, removeLinks, isNotAStopword, stemWord
from collections import OrderedDict

uniqueTermIdDictionary = OrderedDict() # Dictionary that stores terms and corresponding ID
classDictionary = OrderedDict() # Dictionary that stores class name and corresponding ID

def main():
    importFeatsToDictionary('tc_out_improved/feats1.dic') # Import terms and corresponding IDs created in BOW_Extraction module
    importCategoriesToDictionary('files/classIDs.txt') # Import classes and corresponding IDs
    convertTweetEntries('tweets/Tweets.14cat.train', 'tc_out_improved/feats1.train')
    convertTweetEntries('tweets/Tweets.14cat.test', 'tc_out_improved/feats1.test')

def convertTweetEntries(pathToInputFile, pathToOutputFile):
    """Reads file and converts it to termID format - feats file

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
            tempString = "" # String used to generate the format for the classifier
            tempSet = set() # Set used to: 1) keep unique IDs within entry 2) sort IDs in ascending order
            tweetID, tweet, category = line.strip().split("\t")
            tweet = removeLinks(tweet).lower() # Remove links
            termList = filter(None, tokenize(tweet))
            for term in termList:
                if isNotAStopword(term):
                    stemmedTerm = stemWord(term)
                    if stemmedTerm not in uniqueTermIdDictionary: # If the term is not in the dictionary neglect it (test set)
                        continue
                    else:
                        tempSet.add(int(uniqueTermIdDictionary[stemmedTerm])) # Otherwise, if it is in the term list add its corresponding ID
            sortedSet = sorted(tempSet)
            output.write(str(classDictionary[category]) + ' ')
            for termID in sortedSet:
                output.write(str(termID) + ':1 ') # ':1 '.join(map(str, aset)) - second way though it needs a small tweak to print well
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

main()
