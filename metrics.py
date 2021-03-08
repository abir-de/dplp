import networkx as nx # will be required later
from util import *

def compute_precision_acc(labels, predictions):
    sorted_predictions = sorted(((e, i) for i, e in enumerate(predictions)), reverse=True)
    precision = 0.0
    r = int(sum(labels))
    j = 1
    for i in range(len(sorted_predictions)):
        if labels[sorted_predictions[i][1]] == 1.0:
            precision += 1.0 * j / (i+1) / r
            j += 1

    return precision

def reciprocal_rank(labels, predictions):
    sorted_predictions = sorted(((e, i) for i, e in enumerate(predictions)), reverse=True)
    rr = 0
    for i in range(len(sorted_predictions)):
        if labels[sorted_predictions[i][1]] == 1.0:
            rr = 1.0 / (i+1)
            break
    return rr


def precision_at_k(labels, predictions, k):
    sorted_predictions = sorted(((e, i) for i, e in enumerate(predictions)), reverse=True)
    assert(k <= len(labels))
    tp = 0
    for i in range(k):
        if labels[sorted_predictions[i][1]] == 1.0:
            tp += 1

    return 1.0*tp/k


def precision_at_k_list(labels, predictions):
    list = []
    for k in range(len(labels)):
        list.append((precision_at_k(labels, predictions, k+1)))
    return list

def auc(labels, predictions):
    ss=0
    mm=0
    for k in range(len(labels)):
        for j in range(k+1, len(labels)):
            ss=ss+np.sign((labels[k]-labels[j]) * (predictions[k]-predictions[j]))
            mm = mm+ np.abs(np.sign(labels[k]-labels[j]))
    return ss/mm


