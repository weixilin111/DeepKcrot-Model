import os, sys
from itertools import cycle
import keras.backend.tensorflow_backend as KTF
from keras.models import Sequential
from keras.layers import Dropout, Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib as mpl
import numpy as np
import tensorflow as tf
from keras.models import Model

mpl.use('Agg')
import matplotlib.pyplot as plt


def pep1(path):
    seqs = open(path).readlines()
    cut = 0
    X = [[AA.index(res.upper()) if res.upper() in AA else 0
          for res in (seq.split()[0][cut:-cut] if cut != 0 else seq.split()[0])]
         for seq in seqs if seq.strip() != '']
    y = [int(seq.split()[1]) for seq in seqs if seq.strip() != '']

    return np.array(X), np.array(y)

def plot_roc_cv(data, out, label_column=0, score_column=1):
    tprs = []
    aucs = []
    fprArray = []
    tprArray = []
    thresholdsArray = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(len(data)):
        fpr, tpr, thresholds = roc_curve(data[i][:, label_column], data[i][:, score_column])
        fprArray.append(fpr)
        tprArray.append(tpr)
        thresholdsArray.append(thresholds)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blueviolet', 'deeppink', 'cyan'])
    ## ROC plot for CV
    fig = plt.figure(0)
    for i, color in zip(range(len(fprArray)), colors):
        plt.plot(fprArray[i], tprArray[i], lw=1, alpha=0.7, color=color,
                 label='ROC fold %d (AUC = %0.5f)' % (i + 1, aucs[i]))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.5f $\pm$ %0.5f)' % (mean_auc, std_auc),
             lw=2, alpha=.9)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(out,dpi=600)
    plt.close(0)
    return mean_auc

def plot_prc_CV(data, out, label_column=0, score_column=1):
    precisions = []
    aucs = []
    recall_array = []
    precision_array = []
    mean_recall = np.linspace(0, 1, 100)

    for i in range(len(data)):
        precision, recall, _ = precision_recall_curve(data[i][:, label_column], data[i][:, score_column])
        recall_array.append(recall)
        precision_array.append(precision)
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1])[::-1])
        roc_auc = auc(recall, precision)
        aucs.append(roc_auc)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blueviolet', 'deeppink', 'cyan'])
    ## ROC plot for CV
    fig = plt.figure(0)
    for i, color in zip(range(len(recall_array)), colors):
        plt.plot(recall_array[i], precision_array[i], lw=1, alpha=0.7, color=color,
                 label='PRC fold %d (AUPRC = %0.5f)' % (i + 1, aucs[i]))
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = mean_recall[::-1]
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(aucs)

    plt.plot(mean_recall, mean_precision, color='blue',
             label=r'Mean PRC (AUPRC = %0.5f $\pm$ %0.5f)' % (mean_auc, std_auc),
             lw=2, alpha=.9)
    std_precision = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    plt.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.savefig(out,dpi=600)
    plt.close(0)
    return mean_auc

def plot_prc_ind(data, out, label_column=0, score_column=1):
    precision, recall, _ = precision_recall_curve(data[:, label_column], data[:, score_column])
    ind_auc = auc(recall, precision)
    fig = plt.figure(0)
    plt.plot(recall, precision, lw=2, alpha=0.7, color='red',
             label='PRC curve (area = %0.3f)' % ind_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.savefig(out)
    plt.close(0)
    return ind_auc

def plot_roc_ind(data, out, label_column=0, score_column=1):
    fprIndep, tprIndep, thresholdsIndep = roc_curve(data[:, label_column], data[:, score_column])
    ind_auc = auc(fprIndep, tprIndep)
    fig = plt.figure(0)
    plt.plot(fprIndep, tprIndep, lw=2, alpha=0.7, color='red',
             label='ROC curve (area = %0.3f)' % ind_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(out)
    plt.close(0)
    return ind_auc

def save_CV_result_binary(data, out, info=None):
    with open(out, 'w') as f:
        if info:
            f.write('%s\n' % info)
        for i in range(len(data)):
            f.write('# result for fold %d\n' % (i + 1))
            for j in range(len(data[i])):
                f.write('%d\t%s\n' % (data[i][j][0], data[i][j][1]))
    return None

def save_IND_result_binary(data, out, info=None):
    with open(out, 'w') as f:
        if info:
            f.write('%s\n' % info)
        for i in data:
            f.write('%d\t%s\n' % (i[0], i[1]))
    return None

def calculate_metrics(labels, scores, cutoff=0.5, po_label=1):
    my_metrics = {
        'SN': 'NA',
        'SP': 'NA',
        'ACC': 'NA',
        'MCC': 'NA',
        'Recall': 'NA',
        'Precision': 'NA',
        'F1-score': 'NA',
        'Cutoff': cutoff,
    }

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(scores)):
        if labels[i] == po_label:
            if scores[i] >= cutoff:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if scores[i] < cutoff:
                tn = tn + 1
            else:
                fp = fp + 1

    my_metrics['SN'] = tp / (tp + fn) if (tp + fn) != 0 else 'NA'
    my_metrics['SP'] = tn / (fp + tn) if (fp + tn) != 0 else 'NA'
    my_metrics['ACC'] = (tp + tn) / (tp + fn + tn + fp)
    my_metrics['MCC'] = (tp * tn - fp * fn) / np.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (
                                                                                                                         tp + fp) * (
                                                                                                                     tp + fn) * (
                                                                                                                         tn + fp) * (
                                                                                                                         tn + fn) != 0 else 'NA'
    my_metrics['Precision'] = tp / (tp + fp) if (tp + fp) != 0 else 'NA'
    my_metrics['Recall'] = my_metrics['SN']
    my_metrics['F1-score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 'NA'
    return my_metrics
def calculate_metrics_cv(cv_res, label_column=0, score_column=1, cutoff=0.5, po_label=1):
    metrics_list = []
    for i in cv_res:
        metrics_list.append(calculate_metrics(i[:, label_column], i[:, score_column], cutoff=cutoff, po_label=po_label))
    return metrics_list
def save_prediction_metrics_cv(m_list, out):
    with open(out, 'w') as f:
        f.write('Fold')
        for key in m_list[0]:
            f.write('\t%s' % key)
        f.write('\n')
        for i in range(len(m_list)):
            f.write('%d' % (i + 1))
            for key in m_list[i]:
                f.write('\t%s' % m_list[i][key])
            f.write('\n')
    return None
def save_prediction_metrics_ind(m_dict, out):
    with open(out, 'w') as f:
        f.write(' ')
        for key in m_dict:
            f.write('\t%s' % key)
        f.write('\n')
        f.write('Indep')
        for key in m_dict:
            f.write('\t%s' % m_dict[key])
        f.write('\n')
    return None


def CNN(embed_input_dim,embed_output_dim,input_length):

    model = Sequential()
    model.add(Embedding(embed_input_dim, embed_output_dim, input_length=input_length))

    model.add(Conv1D(128, 4, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 4, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 4, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model

def evaluate(X, y, model, params, out, indep=None, batch_size=256, epochs=1000):
    classes = sorted(list(set(y)))
    prediction_result_cv = []
    prediction_result_ind = []
    folds = StratifiedKFold(10,shuffle=True).split(X, y)
    if indep:
        inds = np.zeros((len(indep[1]), 11))
    for i, (trained, valided) in enumerate(folds):
        X_train, y_train = X[trained], y[trained]
        X_valid, y_valid = X[valided], y[valided]
        instance = model(*params)
        if not os.path.exists('%s.%d.h5' % (out, i)):
            best_saving = ModelCheckpoint(filepath='%s.%d.h5' % (out, i), monitor='val_loss',
                                          verbose=1, save_best_only=True, save_weights_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', patience=50,mode='auto' )
            instance.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), verbose=2,
                         callbacks=[best_saving, early_stopping], batch_size=batch_size)
        instance.load_weights('%s.%d.h5' % (out, i))
        print("Validation test:", instance.evaluate(X_valid, y_valid, batch_size=batch_size))
        tmp_result = np.zeros((len(y_valid), len(classes)))
        tmp_result[:, 0], tmp_result[:, 1] = y_valid, instance.predict(X_valid, batch_size=batch_size)[:, 0]
        prediction_result_cv.append(tmp_result)
        if indep:
            print("Independent test:", instance.evaluate(indep[0], indep[1], batch_size=batch_size))
            tmp_result1 = np.zeros((len(indep[1]), len(classes)))
            tmp_result1[:, 0], tmp_result1[:, 1] = indep[1], instance.predict(indep[0], batch_size=batch_size)[:, 0]
            prediction_result_ind.append(tmp_result1)
            inds[:, 0] = indep[1]
            inds[:, i + 1] += instance.predict(indep[0], batch_size=batch_size)[:, 0]
        np.savetxt(out + '.ind.txt', inds, fmt='%f', delimiter='\t')

    return prediction_result_cv, prediction_result_ind
def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    dst = sys.argv[1] if len(sys.argv) > 1 else 'data'
    X, y = pep1('Deep-kcr_Train.txt' )
    indep = pep1('Deep-kcr_Test.txt') if os.path.exists('Deep-kcr_Test.txt') else None
    params = [21,5, X.shape[1]]
    print(X.shape[1])
    cv_res, ind_res = evaluate(X, y, CNN, params, indep=indep, out='Deep_kcrot_pre_WE_%s' % (dst), epochs=500,batch_size=256)
    classes = sorted(list(set(y)))

    if len(classes) == 2:
        save_CV_result_binary(cv_res, r'dataTest_CV.txt')
        plot_roc_cv(cv_res, r'dataTest_ROC_CV.png', label_column=0,score_column=1)
        plot_prc_CV(cv_res, r'dataTest_PRC_CV.png', label_column=0,score_column=1)
        cv_metrics = calculate_metrics_cv(cv_res, label_column=0, score_column=1, cutoff=0.5, po_label=1)
        save_prediction_metrics_cv(cv_metrics, r'dataTest_metrics_CV.txt')
        if indep:
            save_CV_result_binary(ind_res, r'dataTest_IND.txt')
            plot_roc_cv(ind_res, r'dataTest_ROC_IND.png', label_column=0,score_column=1)
            plot_prc_CV(ind_res, r'dataTest_PRC_IND.png', label_column=0,score_column=1)
            ind_metrics = calculate_metrics_cv(ind_res, label_column=0, score_column=1, cutoff=0.5, po_label=1)
            save_prediction_metrics_cv(ind_metrics, r'dataTest_metrics_IND.txt')


if __name__ == '__main__':
    AA = 'GAVLIFWYDNEKQMSTCPHR-'
    main()
