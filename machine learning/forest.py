import tensorflow as tf
from numpy import argmax
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.metrics import f1_score as f1s
from DNNhelpers import load, splitter, filter, taxoRank, xysplit, process, get_model, evalmetrics, cv_summary, roc_plot
import pandas
import numpy as np
from time import sleep
from sklearn.ensemble import RandomForestClassifier





def main():
    
    df = load("genus.tsv")
    scoreslist = []
    predictionslist = []
    tf.get_logger().setLevel('ERROR')
    
    folds = splitter(df, 10, is_LOSO=False)
    for fold in folds:

        train = fold[0]
        test = fold[1]
        foldname = fold[2]
        train = filter(train, is_train=True)
        test = filter(test, is_train=False)
        order, lengths = taxoRank(train)
        xtrain, ytrain = xysplit(train, order)
        xtest, ytest = xysplit(test, order)
        xtrain, ytrain = process(xtrain, ytrain, lengths, is_train=True, rf=True)
        xtest, ytest = process(xtest, ytest, lengths, is_train=False, rf=True)
        model = RandomForestClassifier(n_estimators=250,
            max_depth=None,
            max_features="log2",
            min_samples_split=6,
            min_samples_leaf=20,
            bootstrap=True,
            n_jobs=-1)
        history = model.fit(xtrain, ytrain)

        scores = [0,0]
        yhat = model.predict_proba(xtest)
        scores = evalmetrics(ytest, yhat, scores, foldname)
        predictionslist.append((ytest, yhat))
        scoreslist.append((scores))

    cv_summary(scoreslist)
    roc_plot(predictionslist)




if __name__=="__main__":
    main()
