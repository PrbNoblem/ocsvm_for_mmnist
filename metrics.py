import numpy as np 


def calculate_precision(y_p, y_n):
    true_pos_found = sum([x for x in y_p if x==1])
    false_pos_found = sum([x for x in y_n if x==1])
    if true_pos_found == 0:
        return 0
    else:
        return true_pos_found/(true_pos_found + false_pos_found)


def calculate_f_score(y_p, y_n):
    p = calculate_precision(y_p, y_n)
    r = calculate_recall(y_p)
    if p + r == 0:
        return 0
    else:
        return 2*(p*r/(p+r))

def calculate_recall(y_p):
    return sum([x for x in y_p if x==1])/len(y_p)

def calculate_accuracy(y_p, y_n):
    corr_pos = sum([i for i in y_p if i == 1])
    incorr_pos = sum([i for i in y_n if i == 1])
    return (corr_pos + (len(y_p) - incorr_pos))/ (2 * len(y_p))