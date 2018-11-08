from collections import OrderedDict
from collections import defaultdict
import os
import re

distinctClasses = set() # Set containg the unique class IDs
testClassesList = [] # List containing all the test file class ids
predictionClassesList = [] # List containing all the prediction file class ids
classMeasures = OrderedDict() # Dictionary where key = class ID and value = (precision, recall, f1) triplet

def main():
    global testClassesList, predictionClassesList, distinctClasses
    testClassesList = importClasses('out/feats.test')
    predictionClassesList = importClasses('svm_windows/pred.out')

    calculateClassMeasures()
    print('Accuracy = {:.3f}'.format(calculateSystemAccuracy()))
    print('Macro-F1 = {:.3f}'.format(calculateSystemFMeasure()))
    print('Results per class:')
    for classID in distinctClasses:
        print('{}: P={:.3f} R={:.3f} F={:.3f}'.format(classID, classMeasures[classID][0], classMeasures[classID][1], classMeasures[classID][2]))

    matrix = calculateConfusionMatrix()
    printConfusionMatrix(matrix)

def printConfusionMatrix(matrix):
    """Prints confusion matrix in a specific format

    Parameters
    ----------
    matrix : Dictionary type
        Confusion matrix - Dictionary of Dictionaries and frequencies
    """
    for key, value in matrix.iteritems():
        for k, v in value.iteritems():
            print("{:>3}".format(str(v))),
        print("")

def calculateConfusionMatrix():
    """Calculates the confusion matrix of the classifier

    Returns
    -------
    confusionMatrix : Dictionary type
        Confusion matrix - Dictionary of Dictionaries and frequencies
    """
    global distinctClasses, testClassesList, predictionClassesList

    confusionMatrix = OrderedDict()
    for index in distinctClasses:
        confusionMatrix[index] = OrderedDict()
        for innerIndex in distinctClasses:
            confusionMatrix[index][innerIndex] = 0

    totalPredictions = len(testClassesList)
    for i in range(0, totalPredictions):
        confusionMatrix[int(testClassesList[i])][int(predictionClassesList[i])] += 1

    return confusionMatrix

def calculateSystemFMeasure():
    """Calculates precision, recall and F1 measures of the classifier for the given class

    Returns
    -------
    F1 : Float type
        The mean F1 of the classifier
    """
    global classMeasures
    F1Sum = 0.0
    for classID, classScores in classMeasures.iteritems():
        F1Sum += classScores[2]

    return float(F1Sum) / float(len(classMeasures))

def calculateClassMeasures():
    """Calculates precision, recall and F1 measures of the classifier for the given class
    """
    global testClassesList, predictionClassesList, classMeasures
    correctPredictions = numberOfPredictions = classTotal = 0

    totalPredictions = len(testClassesList)
    for classID in distinctClasses:
        for i in range(0, totalPredictions):

            if int(predictionClassesList[i]) == classID:
                numberOfPredictions += 1 #
                if testClassesList[i] == predictionClassesList[i]:
                    correctPredictions += 1

            if int(testClassesList[i]) == classID:
                classTotal += 1

        precision = float(correctPredictions) / float(numberOfPredictions)
        recall = float(correctPredictions) / float(classTotal)
        f1 = (2 * precision * recall) / (precision + recall)
        classMeasures[classID] = (precision, recall, f1)

def calculateSystemAccuracy():
    """Calculates accuracy of classifier

    Returns
    -------
    accuracy : Float type
        The accuracy of the classifier in terms of correctly classified IDs
    """
    global testClassesList, predictionClassesList
    correctPredictions = 0
    totalPredictions = len(testClassesList)
    for i in range(0, totalPredictions):
        if testClassesList[i] == predictionClassesList[i]:
            correctPredictions += 1
    return float(correctPredictions)/float(totalPredictions)

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
    global distinctClasses
    classesList = []

    with open(pathToFile, 'r') as file:
        for line in file:
            if line == "\n": # Skip empty lines
                continue
            category = line.strip().split(' ')[0]
            classesList.append(category)
            distinctClasses.add(int(category))
    return classesList

main()
