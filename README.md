# Information Retrieval System Evaluation and Text Classification

## Dependencies

The assignment was completed using Python 2.7 on Anaconda, since that's the version in the DICE machines.

Libraries and packages used (in alphabetical order):
- collections
- glob
- lxml.html
- nltk.corpus
- nltk.stem
- numpy
- os
- re
- urllib2

## Folder contents

### IR_Evaluation/

- eval_out/ : Folder in which the IR evaluator's output is stored (*.eval)
- format_check_scripts/ : Folder containing given perl scripts to check output format
- systems/ : Folder containing the qrels.txt file and *.results files (evaluator's input)
- Eval.py : Python module that implements the IR system evaluator module

### Text_Classification/

- files/ : Folder containing input files for the module - class IDs and stopword file
- Improved_Classifier/ : Folder containing all files and folders used for the improved classifier (see notes!)
- svm_linux/ : Folder containing the SVM classifier executable files, the model and the prediction output
- tc_out/ : Folder in which the text classification's module output is stored
- tweets/ : Folder containing the tweet train and test files
- BOW_Extractor.py : Python module that implements the BOW extraction from the tweets train file
- Feature_Converter.py : Python module that converts the tweets train and test files to the appropriate format for the classifier
- Classifier_Evaluator.py : Python module that evaluates the classifier's perfomance
- autorun.sh : Shell script that invokes all necessary modules and executables to complete the task

## Running IR Evaluation module

From a shell in the IR_Evaluation directory run the Eval script ("python .\Eval.py")

## Running Text Classification module

From a shell in the Text_Classification directory run the autorun script (".\autorun.sh")

### Notes

- The Improved_Classifier/ directory follows the exact same structure as the Text_Classification/ folder, but all file and folders' names have "_improved" appended to them.
- The improved text classification module retrieves all webpage title text from all links within tweets, so running it may take a while depending on system.
- The only difference between the baseline and the improved module is the features added to the feats.dic


