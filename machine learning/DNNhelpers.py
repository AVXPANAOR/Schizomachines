import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from imblearn.over_sampling import SVMSMOTE
from sklearn.preprocessing import LabelBinarizer as lb
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve, matthews_corrcoef, balanced_accuracy_score, f1_score
from itertools import combinations



def load(tsvpath):

    df = pd.read_csv(tsvpath, sep="\t", header=0, index_col=0)
    df = df.filter(like="Bacteria", axis=0)
    taxon = pd.Series([f"{taxon.split(';')[1].lstrip('p__')}-{taxon.split(';')[-2].lstrip('f__')}-{taxon.split(';')[-1].lstrip('g__')}" 
                      for taxon in df.index], index=df.index, name='taxon')
    df = pd.concat([df, taxon], axis=1)
    df = df.drop(df[df["taxon"].str.endswith("uncultured-uncultured")].index)
    df = df.drop(df[df["taxon"].str.endswith("-")].index)
    df = df.drop(df[df["taxon"].str.endswith("Mitochondria")].index)
    df = df.drop(df[df["taxon"].str.endswith("Chloroplast")].index)
    df = df.set_index("taxon")
    df = df.T
    df["disease"] = [sample[0] for sample in df.index]
    df["study"] = [sample.split("-")[1].upper() for sample in df.index]
    
    return df



def splitter(df, K=None, is_LOSO=False):
   
    df = pd.DataFrame.sample(df, frac=1)
    folds = []

    if is_LOSO:
        groups = df.groupby("study")
        for name, group in groups:
            test = groups.get_group("BDMDD65")
            test = pd.concat([test, groups.get_group("BDMDD11"), groups.get_group("NSCZ21")])
            if not (name in ["BDMDD65", "BDMDD11", "NSCZ21", "BDMDD105"]):
                test = pd.concat([test, groups.get_group(name)])
                train = df.drop(test.index)
                folds.append((train, test, name))
    else:
        for i in range(K):
            start = round(i*df.shape[0]/K)
            end = round((i+1)*df.shape[0]/K)
            test = df.iloc[start:end,:]
            train = df.copy().drop(test.index)
            folds.append((train, test, i))

    return folds



def filter(df, is_train=False): 
    #removes taxa that are found in less than 1% of samples.
    #removes taxa with maximum frequency of less than 1%
    df = df.select_dtypes(exclude="str").div(df.sum(axis=1, numeric_only=True),axis=0).round(6).combine_first(df)
    if is_train:
        df = df.loc[:,np.count_nonzero(df, axis=0,) > (df.shape[0] * 0.01)]
        df = df.select_dtypes(exclude="str").loc[:,df.max(axis=0, numeric_only=True) > 0.01].combine_first(df.select_dtypes(include="str"))

    return df



def taxoRank(df): #ToDo
    #calcultes spearman's rank correlation matrix among taxa of each clade.
    #sorts taxa based on TaxoNN_corr algorithm.
    #see Sharma et al. TaxoNN: ensemble of neural networks on stratified microbiome data for disease prediction(2020)
    firmicutes = df.filter(like="Firmicutes", axis=1)
    proteobacteria = df.filter(like="Proteobacteria", axis=1)
    bacteroidota = df.filter(like="Bacteroidota", axis=1)
    other = df.drop(firmicutes, axis=1).drop(proteobacteria, axis=1).drop(bacteroidota, axis=1).select_dtypes(exclude="str")
   
    order = np.array([])
    lengths = []
    for phylum in [firmicutes, proteobacteria, bacteroidota, other]:
        phylum = phylum.corr(method="spearman").abs()
        phylum["rank"] = phylum.prod(axis="columns").pow(1/phylum.shape[0])
        phylum = phylum.sort_values(by="rank", ascending=False)
        order = np.append(order, phylum.index.array)
        lengths.append(phylum.shape[0])
    
    return order, lengths

    

def xysplit(df, order):

    features = np.array(df[order])
    lables = np.array(df["disease"].map({"h":0, "m":1, "b":2, "s":3}))

    return features, lables



def augment(x_train, y_train):

    #augments trainset using SVM-SMOTE algorithm, to handle class imbalance.
    smote = SVMSMOTE(sampling_strategy='auto')
    x_train, y_train = smote.fit_resample(x_train, y_train)

    return x_train, y_train



def normalizer(dataset):

    #normalizes data using standard scaler.
    scaler = sklearn.preprocessing.StandardScaler()
    dataset = scaler.fit_transform(dataset)
    dataset = np.clip(dataset, -5, 5)

    return dataset



def splicer(dataset, lengths):

    #splices dataset features to 4 clades.
    spliced = dict()
    dataset = np.expand_dims(dataset, axis=2)
    spliced["clade1"], \
    spliced["clade2"], \
    spliced["clade3"], \
    spliced["clade4"] = tf.split(dataset, lengths, axis=1)

    return spliced



def process(features, labels, lengths, is_train=False, rf=False):

    features = normalizer(features)
    if is_train:
        features, labels = augment(features, labels)
    if rf==True:
        return features, labels
    featureset = splicer(features, lengths)

    return featureset, labels



def get_submodel(inputshape):

    #builds a convolutional neural network to be used in the final ensemble model.
    inputlayer = tf.keras.Input(inputshape)
    convlayer = tf.keras.layers.Conv1D(32, kernel_size=2, strides=1, activation="relu")
    x = convlayer(inputlayer)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.44)(x)

    x = tf.keras.layers.Conv1D(64, kernel_size=2, strides=1, activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.29)(x)

    outputlayer = tf.keras.layers.Flatten()(x)
    outputlayer = tf.keras.layers.BatchNormalization()(outputlayer)
    outputlayer = tf.keras.layers.Dropout(0.14)(outputlayer)

    submodel = tf.keras.Model(inputs=inputlayer, outputs=outputlayer)
    return submodel



def get_model(lengths):

    #builds an ensemble of 4 CNNs, each recieving a single clade as an input.
    #CNN outputs are eventually concatenated into a single tensor wich is fed to a dense layer.
    submodel1 = get_submodel((lengths[0],1,)) 
    submodel2 = get_submodel((lengths[1],1,))
    submodel3 = get_submodel((lengths[2],1,))
    submodel4 = get_submodel((lengths[3],1,))

    input1 = tf.keras.Input((lengths[0],1,), name="clade1")
    input2 = tf.keras.Input((lengths[1],1,), name="clade2")
    input3 = tf.keras.Input((lengths[2],1,), name="clade3")
    input4 = tf.keras.Input((lengths[3],1,), name="clade4")
    
    y1 = submodel1(input1)
    y2 = submodel2(input2)
    y3 = submodel3(input3)
    y4 = submodel4(input4)

    x = tf.keras.layers.Concatenate()([y1, y2, y3, y4])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.18)(x)

    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.46)(x)
    output = tf.keras.layers.Dense(4, activation="softmax")(x)

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    model = tf.keras.Model(inputs=[input1, input2, input3, input4], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0097909),
                     loss="sparse_categorical_crossentropy", metrics=metrics)

    return model

def evalmetrics(ytest, yhat, scores, name):

    yhatmax = np.argmax(yhat, axis=1)
    scores.append(matthews_corrcoef(ytest, yhatmax))
    scores.append(balanced_accuracy_score(ytest, yhatmax))
    scores.append(f1_score(ytest, yhatmax, average="macro"))
    scores.append(roc_auc_score(ytest, yhat, average="macro", multi_class="ovr"))
    scores.append(roc_auc_score(ytest, yhat, average="macro", multi_class="ovo"))
    scores.append(name)
    return scores



def OvRplot(predictions, labelnames):

    #plots One-vs-Rest ROC-curve averaged over all folds.
    tprs = {i:[] for i in [0,1,2,3,"average"]}
    aucs = {i:[] for i in [0,1,2,3,"average"]}
    mean_fpr = np.linspace(0, 1, 1000)

    for y_test, y_hat in predictions:

        binarizer = lb().fit(y_test)
        y_test = binarizer.transform(y_test)
    
        for i in [0, 1, 2, 3]:
         
            fpr, tpr, _ = roc_curve(y_test[:,i], y_hat[:,i])
            auc_score = auc(fpr, tpr)
            aucs[i].append(auc_score)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs[i].append(interp_tpr)

        fpr, tpr, _ = roc_curve(y_test.ravel(), y_hat.ravel())
        auc_score = auc(fpr, tpr)
        aucs["average"].append(auc_score)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs["average"].append(interp_tpr)
    
    fig, ax = plt.subplots(figsize=(6, 6))

    for ovr in tprs:
   
        mean_tpr = np.mean(tprs[ovr], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.round(auc(mean_fpr, mean_tpr), 3)
        std_auc = np.round(np.std(aucs[ovr]), 3)

        plt.plot(mean_fpr, mean_tpr,
                 label=(f"{labelnames[ovr]} vs the rest (AUC = {mean_auc} ± {std_auc})"
                         if ovr != "average"
                        else f"micro-average (AUC = {mean_auc} ± {std_auc})"))
        
    plt.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    plt.show()


def OvOplot(predictions, labelnames):

    #plots One-vs-One ROC-curve averaged over all folds.
    pairlist = list(combinations([0, 1, 2, 3], 2))
    mean_fpr = np.linspace(0.0, 1.0, 1000)
    tprs = dict()
    tprs["average"] = []
    aucs = dict()
    aucs["average"] = []

    for y_test, y_hat in predictions:
        
        ovo_tpr = np.zeros_like(mean_fpr)
    
        for ix, (label_a, label_b) in enumerate(pairlist):

            a_mask = y_test == label_a
            b_mask = y_test == label_b
            ab_mask = np.logical_or(a_mask, b_mask)

            a_true = a_mask[ab_mask]
            b_true = b_mask[ab_mask]
        
            fpr_a, tpr_a, _ = roc_curve(a_true, y_hat[ab_mask, label_a])
            fpr_b, tpr_b, _ = roc_curve(b_true, y_hat[ab_mask, label_b])

            mean_tpr = np.zeros_like(mean_fpr)
            mean_tpr += np.interp(mean_fpr, fpr_a, tpr_a)
            mean_tpr += np.interp(mean_fpr, fpr_b, tpr_b)
            mean_tpr /= 2
            mean_tpr[0] = 0.0
            mean_auc = auc(mean_fpr, mean_tpr)
            ovo_tpr += mean_tpr

            if ix not in tprs:

                tprs[ix] = []
                aucs[ix] = []

            tprs[ix].append(mean_tpr)
            aucs[ix].append(mean_auc)
    
        ovo_tpr /= sum(1 for pair in enumerate(pairlist))
        binarizer = lb().fit(y_test)
        y_test = binarizer.transform(y_test)
        macro_roc_auc_ovo = roc_auc_score(y_test, y_hat, multi_class="ovo", average="macro")
        tprs["average"].append(ovo_tpr)
        aucs["average"].append(macro_roc_auc_ovo)
        
    fig, ax = plt.subplots(figsize=(6, 6))

    for ix, (label_a, label_b) in enumerate(pairlist):
        mean_tpr = np.mean(tprs[ix], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.round(auc(mean_fpr, mean_tpr), 3)
        std_auc = np.round(np.std(aucs[ix]), 3)

        plt.plot(mean_fpr, mean_tpr,
                 label=f"Mean {labelnames[label_a]} vs {labelnames[label_b]} (AUC = {mean_auc}) ± {std_auc})")
    
    mean_tpr = np.mean(tprs["average"], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.round(auc(mean_fpr, mean_tpr), 3)
    std_auc = np.std(aucs["average"]).round(3)
    plt.plot(mean_fpr, mean_tpr,
                 label=f"One-vs-One macro-average (AUC = {mean_auc}) ± {std_auc})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-One multiclass")
    plt.legend()
    plt.show()


def roc_plot(predictions):

    labelnames = ["Healthy", "MDD", "BD", "SCZ"]
    
    OvOplot(predictions, labelnames)
    OvRplot(predictions, labelnames)
    


    print("-" * 75)

def cv_summary(scoreslist):

    #prints model evaluation summaries, averaged over all folds.
    i = 1
    print("-" * 75)
    
    for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist:
        print("-" * 75)
        print(f"fold {name}:   loss: {loss} - acc: {acc} - mcc: {mcc} - b-acc: {bacc} - F1: {f1s} - AUC(OVR): {ovr} - AUC(OVO): {ovo}")
        i += 1

    print("-" * 75)
    print(f"mean acc: {np.mean([acc for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])} +- {np.std([acc for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])}")
    print(f"mean loss: {np.mean([loss for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])} +- {np.std([loss for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])}")
    print(f"mean mcc: {np.mean([mcc for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])} +- {np.std([mcc for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])}")
    print(f"mean b-acc: {np.mean([bacc for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])} +- {np.std([bacc for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])}")
    print(f"mean F1: {np.mean([f1s for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])} +- {np.std([f1s for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])}")
    print(f"mean AUC(OVR): {np.mean([ovr for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])} +- {np.std([ovr for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])}")
    print(f"mean AUC(OVO): {np.mean([ovo for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])} +- {np.std([ovo for (loss, acc, mcc, bacc, f1s, ovr, ovo, name) in scoreslist])}")



    print("-" * 75)