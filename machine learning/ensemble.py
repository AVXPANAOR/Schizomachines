import tensorflow as tf
from numpy import argmax
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.metrics import f1_score as f1s
from DNNhelpers import load, splitter, filter, taxoRank, xysplit, process, get_model, evalmetrics, cv_summary, roc_plot
import pandas
import numpy as np
from time import sleep




def main():
    
    df = load("genus.tsv")
    scoreslist = []
    predictionslist = []
    tf.get_logger().setLevel('ERROR')
    
    folds = splitter(df, 10, is_LOSO=True)
    for fold in folds:

        train = fold[0]
        test = fold[1]
        foldname = fold[2]
        train = filter(train, is_train=True)
        test = filter(test, is_train=False)
        order, lengths = taxoRank(train)
        xtrain, ytrain = xysplit(train, order)
        xtest, ytest = xysplit(test, order)
        xtrain, ytrain = process(xtrain, ytrain, lengths, is_train=True)
        xtest, ytest = process(xtest, ytest, lengths, is_train=False)
        model = get_model(lengths)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(xtrain, ytrain, epochs=100, validation_split=0.2, callbacks=[early_stopping], verbose=1)
        
        scores = model.evaluate(xtest, ytest, verbose=1)
        yhat = model.predict(xtest)
        scores = evalmetrics(ytest, yhat, scores, foldname)
        predictionslist.append((ytest, yhat))
        scoreslist.append((scores))

    cv_summary(scoreslist)
    roc_plot(predictionslist)


if __name__=="__main__":
    main()