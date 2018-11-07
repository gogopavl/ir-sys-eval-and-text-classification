from collections import OrderedDict
from collections import defaultdict
import collections
import os
import re

testClassesList = []
predictionClassesList = []

def main():
    global testClassesList, predictionClassesList
    testClassesList = importClasses('out/feats.test')
    predictionClassesList = importClasses('svm_windows/pred.out')
    print('Accuracy = {}'.format(calculateAccuracy()))

def calculateAccuracy():
    """Calculates accuracy of classifier

    Returns
    -------
    accuracy : Float type
        The accuracy of the classifier in terms of correctly classified IDs
    """
    global testClassesList, predictionClassesList
    correctPredictions = 0
    numberOfIDs = len(testClassesList)
    for i in range(0, numberOfIDs):
        if testClassesList[i] == predictionClassesList[i]:
            correctPredictions += 1
    return float(correctPredictions)/float(numberOfIDs)

def importClasses(pathToFile):
    """Reads classes from test file

    Parameters
    ----------
    pathToFile : String type
        The path leading to the file

    Returns
    -------
    classesList : List of Integers
        The list with all class IDs
    """
    classesList = []

    with open(pathToFile, 'r') as file:
        for line in file:
            if line == "\n": # Skip empty lines
                continue
            category = line.strip().split(' ')[0]
            classesList.append(category)
    return classesList

main()
