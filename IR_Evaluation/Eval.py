from collections import OrderedDict
from collections import defaultdict
import numpy as np
import glob
import os
import re

systemResults = OrderedDict() # Dictionary where key = systemID and value = Dictionary, where key = queryID and value = [(docID, rank, score)]
queryRelevantDocuments = OrderedDict() # Dictionary where key = queryID and value = [(documentID, relevance value)]

def main():
    importResultsFiles('systems/')
    importRelevantDocuments('systems/qrels.txt')
    calculateMeasures()

def calculateMeasures():
    """Calculates the measures for each system and writes them to file

    Returns
    -------
    measures : Dictionary type
        Results for each system
    """
    global systemResults, queryRelevantDocuments

    folder = "eval_out/"
    if not os.path.exists(folder): # Check whether the directory exists or not
        os.makedirs(folder)

    with open(folder+'All.eval', 'w') as allFile:
        allFile.write('\tP@10\tR@50\tr-Precision\tAP\tnDCG@10\tnDCG@20\n')
        for systemID, results in systemResults.iteritems():

            systemPrecision = precision(results, 10)
            systemRecall = recall(results, 50)
            systemRPrecision = r_precision(results)
            average_precision = averagePrecision(results)
            nDCG10 = nDCG(results, 10)
            nDCG20 = nDCG(results, 20)

            filename = "S"+str(systemID)+".eval"
            with open(folder+filename, 'w') as output:
                output.write('\tP@10\tR@50\tr-Precision\tAP\tnDCG@10\tnDCG@20\n')
                for queryID in queryRelevantDocuments:
                    output.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(queryID, systemPrecision[queryID], systemRecall[queryID], systemRPrecision[queryID], average_precision[queryID], nDCG10[queryID], nDCG20[queryID]))
                output.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format("mean", np.mean(systemPrecision.values()), np.mean(systemRecall.values()), np.mean(systemRPrecision.values()), np.mean(average_precision.values()), np.mean(nDCG10.values()), np.mean(nDCG20.values())))

            allFile.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format("S"+str(systemID), np.mean(systemPrecision.values()), np.mean(systemRecall.values()), np.mean(systemRPrecision.values()), np.mean(average_precision.values()), np.mean(nDCG10.values()), np.mean(nDCG20.values())))

def nDCG(systemResults, k):
    """Calculates the nDCG for the system results

    Parameters
    ----------
    systemResults : Dictionary type
        Dictionary containing system results, where key = queryID and value = [(docID, rank, score)]
    k : Integer type
        The cutoff - number of top k results

    Returns
    -------
    nDCG : Dictionary type
        The nDCG of the system for each query
    """
    global queryRelevantDocuments

    nDCG = OrderedDict()

    for queryID, results in systemResults.iteritems():
        numberOfRelevantDocumentsRetrieved = 0
        numberOfRelevantDocuments = len(queryRelevantDocuments[queryID])
        sum = 0
        idealSum = 0
        numberOfDocs = 0
        flag = True
        for counter, entry in enumerate(results):
            documentID = entry[0]
            if flag: # Only for the first entry - no discount
                sum = getDocumentRelevanceValue(queryID, documentID)
                flag = False
            else: # For every entry from i=2 onwards
                sum += float(getDocumentRelevanceValue(queryID, documentID))/float(np.log2(counter + 1))
            if counter + 1 == k:
                break

        iDCG = getIDCG(queryID, k)
        nDCG[queryID] = float(sum)/float(iDCG)
    return nDCG

def getIDCG(queryID, k):
    """Calculates the iDCG for a given query at cutoff k

    Parameters
    ----------
    queryID : Integer type
        The query ID
    k : Integer type
        The cutoff - number of top k results

    Returns
    -------
    iDCG : Float type
        The iDCG value at k
    """
    global queryRelevantDocuments
    relevanceValuesSorted = sorted(queryRelevantDocuments[queryID].items(), key = lambda (k,v): v, reverse = True)
    iDCG = 0
    flag = True
    for iteration, (documentID, value) in enumerate(relevanceValuesSorted):
        if flag: # Only for the first entry - no discount
            iDCG = value
            flag = False
        else: # For every entry from i=2 onwards
            iDCG += float(value)/float(np.log2(iteration + 1))
        if iteration + 1 == k:
            break
    return iDCG

def getDocumentRelevanceValue(queryID, documentID):
    """Returns the document gain value for a given query and document

    Parameters
    ----------
    queryID : Integer type
        The query ID
    documentID: Integer type
        The document ID

    Returns
    -------
    gain : Integer type
        The relevance value of the specified document
    """
    global queryRelevantDocuments
    if documentID in queryRelevantDocuments[queryID]:
        return queryRelevantDocuments[queryID][documentID]
    else:
        return 0

def averagePrecision(systemResults):
    """Calculates the average precision for the system results

    Parameters
    ----------
    systemResults : Dictionary type
        Dictionary containing system results, where key = queryID and value = [(docID, rank, score)]

    Returns
    -------
    average_precision : Dictionary type
        The average precision of the system for each query
    """
    global queryRelevantDocuments

    average_precision = OrderedDict()

    for queryID, results in systemResults.iteritems():
        numberOfRelevantDocumentsRetrieved = 0
        numberOfRelevantDocuments = len(queryRelevantDocuments[queryID])
        sum = 0.0
        numberOfDocs = 0
        for entry in results:
            documentID = entry[0]
            numberOfDocs += 1
            if documentID in queryRelevantDocuments[queryID]:
                numberOfRelevantDocumentsRetrieved += 1
                sum += float(numberOfRelevantDocumentsRetrieved)/float(numberOfDocs)
        average_precision[queryID] = float(sum)/float(numberOfRelevantDocuments)
    return average_precision

def r_precision(systemResults):
    """Calculates the r-precision for the system results

    Parameters
    ----------
    systemResults : Dictionary type
        Dictionary containing system results, where key = queryID and value = [(docID, rank, score)]

    Returns
    -------
    r_precision : Dictionary type
        The r-precision of the system for each query
    """
    global queryRelevantDocuments

    query_rPrecision = OrderedDict()

    for queryID, results in systemResults.iteritems():
        numberOfRelevantDocumentsRetrieved = 0
        numberOfRelevantDocuments = len(queryRelevantDocuments[queryID])
        for counter, entry in enumerate(results):
            documentID = entry[0]
            if documentID in queryRelevantDocuments[queryID]:
                numberOfRelevantDocumentsRetrieved += 1
            if counter + 1 == numberOfRelevantDocuments:
                break
        query_rPrecision[queryID] = float(numberOfRelevantDocumentsRetrieved)/float(numberOfRelevantDocuments)
    return query_rPrecision

def recall(systemResults, k):
    """Calculates the recall for the first k results

    Parameters
    ----------
    systemResults : Dictionary type
        Dictionary containing system results, where key = queryID and value = [(docID, rank, score)]
    k : Integer type
        The top k number of results

    Returns
    -------
    recall : Dictionary type
        The recall of the system for each query
    """
    global queryRelevantDocuments

    queryRecall = OrderedDict()

    for queryID, results in systemResults.iteritems():
        numberOfRelevantDocumentsRetrieved = 0
        for counter, entry in enumerate(results):
            documentID = entry[0]
            if documentID in queryRelevantDocuments[queryID]:
                numberOfRelevantDocumentsRetrieved += 1
            if counter + 1 == k:
                break
        queryRecall[queryID] = float(numberOfRelevantDocumentsRetrieved)/float(len(queryRelevantDocuments[queryID]))
    return queryRecall

def precision(systemResults, k):
    """Calculates the precision for the first k results

    Parameters
    ----------
    systemResults : Dictionary type
        Dictionary containing system results, where key = queryID and value = [(docID, rank, score)]
    k : Integer type
        The top k number of results

    Returns
    -------
    precision : Dictionary type
        The precision of the system for each query
    """
    global queryRelevantDocuments

    queryPrecision = OrderedDict()

    for queryID, results in systemResults.iteritems():
        numberOfRelevantDocumentsRetrieved = 0
        numberOfRetrievedDocuments = k
        for counter, entry in enumerate(results):
            documentID = entry[0]
            if documentID in queryRelevantDocuments[queryID]:
                numberOfRelevantDocumentsRetrieved += 1
            if counter + 1 == k:
                break
        queryPrecision[queryID] = float(numberOfRelevantDocumentsRetrieved)/float(numberOfRetrievedDocuments)
    return queryPrecision

def importResultsFiles(directory):
    """Reads ranked results from given results file and stores them into memory

    Parameters
    ----------
    directory : String type
        The directory leading to the results files
    """
    global systemResults

    for resultsFile in sorted(glob.glob(directory+'*.results')):
        with open(resultsFile, 'r') as file:
            systemID = int(resultsFile.split(".")[0][-1])
            systemResults[systemID] = OrderedDict()
            for line in file:
                if line == "\n": # Skip empty lines
                    continue
                lineParts = line.strip().split(" ")
                lineParts = lineParts[0:1] + lineParts[2:-1]
                queryID, documentID, rank, score = lineParts
                if int(queryID) not in systemResults[systemID]:
                    systemResults[systemID][int(queryID)] = []
                systemResults[systemID][int(queryID)].append((int(documentID), int(rank), float(score)))

def importRelevantDocuments(pathToFile):
    """Reads the relevant documents for each query and stores them into memory

    Parameters
    ----------
    pathToFile : String type
        The path leading to the file
    """
    global queryRelevantDocuments

    with open(pathToFile, 'r') as file:
        for line in file:
            if line == "\n": # Skip empty lines
                continue
            lineParts = line.strip().split(":")
            queryID = int(lineParts[0])
            if queryID not in queryRelevantDocuments:
                queryRelevantDocuments[queryID] = OrderedDict()

            docIDRelevanceList = re.split(r'(?!\,)\W+', lineParts[1].strip())
            docIDRelevanceList = filter(None, docIDRelevanceList)

            for pair in docIDRelevanceList:
                documentID, value = pair.strip().split(",")
                queryRelevantDocuments[queryID][int(documentID)] = int(value)

main()
