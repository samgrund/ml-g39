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

def New_Trans(old):
    # Noncoding, coding start, coding, coding end, reverse end, reversing, reverse start
    new = np.zeros(shape=(7,7))
    for i in range(1,6):
        new[i,i+1] = 1 # CS -> Coding is a must-be
    new[0,0] = old[2,2] # N,N
    new[0,1] = old[2,0] # N,CS
    new[0,4] = old[2,1] # N,RE
    new[2,2] = old[0,0] # C,C
    new[2,3] = old[0,1] + old[0,2] # C,CE
    new[3,0] = old[0,2] # CE,N
    new[3,4] = old[0,1] # CE,RS
    new[5,5] = old[1,1] # R,R
    new[5,6] = old[1,0] + old[1,2] # R,RS
    new[6,0] = old[1,2] # RS,N
    new[6,1] = old[1,0] # RS,CS
    # print(new)

    return new

def New_Emiss(trans, emit):
    # N CCC C CCC RRR R RRR
    new_trans = np.zeros(shape=(15, 15))
    new_emit = np.zeros(shape=(4, 15))
    for i in range(14):
        new_trans[i, i+1] = 1
    new_trans[0,0] = trans[2,2] # N,N
    new_trans[0,1] = trans[2,0]  # N,CS
    new_trans[0,8] = trans[2,1]  # N,RE
    new_trans[4,4] = trans[0,0]  # C,C
    new_trans[4,5] = trans[0,1] + trans[0, 2]  # C,CE
    new_trans[7,0] = trans[0,2] # CE,N
    new_trans[7,8] = trans[0,1]  # CE,RE
    new_trans[11,11] = trans[1,1]  # R,R
    new_trans[11,12] = trans[1,0] + trans[1, 2]  # R,RS
    new_trans[14,0] = trans[1,2] # RS,N
    new_trans[14,1] = trans[1,0] # RS,CS

    new_emit[0,] = [0,1] + [0] * 4 + [1]*2 + [0] * 2 + [1] + [0] * 2 + [1,0]
    new_emit[1,] = [0]*2 + [1] + [0]*2 + [1] + [0]*2 + [1]*2 + [0]*4 + [1]
    new_emit[2,] = [0]*12 + [1] + [0,0]
    new_emit[3,] = [0]*3 + [1] + [0]*11
    new_emit[:,0] = emit[:,0]
    new_emit[:,4] = emit[:,1]
    new_emit[:,11] = emit[:,2]

    return new_trans,new_emit

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


    new_A = New_Trans(A) # 7*7
    new_A_large,new_B_large = New_Emiss(A,B.transpose())
    print("7*7 transmission: ",new_A)
    print("15*15 transmission: ",new_A_large)
    print("15*15 emission: ",new_B_large)
    # print(pi)
    return pi, new_A, new_B_large