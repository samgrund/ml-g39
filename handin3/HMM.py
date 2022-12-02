import sys
import numpy as np


def Create_Pi(ann):
    pi = [0, 0, 0]
    for i in ann:
        # print(i[0])
        # break
        # print(i)
        if i[0] == 'C':
            pi[0] += 1
        elif i[0] == 'R':
            pi[1] += 1
        else:
            pi[2] += 1
    pi[0] /= 5
    pi[1] /= 5
    pi[2] /= 5
    return pi

def Count_Hidden(str,a,b):
    ############### Count num of (x[i-1] == am x[i] == b) ##############3

    cnt = 0
    # print(a,b)
    # print(str[0],str[1])
    for i in range(1,len(str)):
        if str[i-1] == a and str[i] == b:
            cnt += 1

    return cnt

def Create_Transition(ann):
    hidden_stat = ['C', 'R', 'N']
    trans = np.ones(shape=(3, 3))
    # print(hidden_stat[2])
    for str in ann:
        # print(str[0],str[1])
        for i in range(3):
            for j in range(3):
                a = Count_Hidden(str, hidden_stat[i], hidden_stat[j])
                trans[i,j] += a
                # print(a,i,j)
    # print(trans)
    trans = np.divide(trans, trans.sum(axis=1, keepdims=True),
                      where=trans.sum(axis=1, keepdims=True) != 0)

    return trans

def Create_Emission(gen,ann):
    hidden_stat_mapping = {'C': 0, 'R': 1, 'N': 2}
    observ_stat_mapping = {'A':0, 'T':1, 'C':2, 'G':3}

    emis = np.ones(shape=(3, 4))
    for j in range(len(ann)):
        for i in range(len(ann[j])):
            emis[hidden_stat_mapping[ann[j][i]]][observ_stat_mapping[gen[j][i]]] += 1
    # print(emis)

    emis = np.divide(emis, emis.sum(axis=1, keepdims=True),
                      where=emis.sum(axis=1, keepdims=True) != 0)
    return emis

def Init(gen,pred,ann):
    # print(ann[0][364])
    ###################### Initial Pi #################################
    pi = Create_Pi(ann)
    # print(pi)

    ###################### Initial A ##################################
    # Transition
    A = Create_Transition(ann)
    # print(A)

    ###################### Initial B ##################################
    # Emission
    B = Create_Emission(gen,ann)
    # print(B)

    return pi, A, B