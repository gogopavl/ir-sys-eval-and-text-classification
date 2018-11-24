python BOW_Extractor.py
python Feature_Converter.py
svm_linux/svm_multiclass_learn -c 1000 tc_out/feats.train svm_linux/model
svm_linux/svm_multiclass_classify tc_out/feats.test svm_linux/model svm_linux/pred.out
python Classifier_Evaluator.py
