import sys
import numpy as np
import FileReader as FR
import HMM
import EM
import utility

gen = []
hidden = []
if __name__ == "__main__":

    gen1 = FR.read_fasta_file('./data-handin3/genome1.fa')
    gen2 = FR.read_fasta_file('./data-handin3/genome2.fa')
    gen3 = FR.read_fasta_file('./data-handin3/genome3.fa')
    gen4 = FR.read_fasta_file('./data-handin3/genome4.fa')
    gen5 = FR.read_fasta_file('./data-handin3/genome5.fa')
    gen6 = FR.read_fasta_file('./data-handin3/genome6.fa')
    gen7 = FR.read_fasta_file('./data-handin3/genome7.fa')
    gen8 = FR.read_fasta_file('./data-handin3/genome8.fa')
    gen9 = FR.read_fasta_file('./data-handin3/genome9.fa')
    gen10 = FR.read_fasta_file('./data-handin3/genome10.fa')

    ann1 = FR.read_fasta_file('./data-handin3/true-ann1.fa')
    ann2 = FR.read_fasta_file('./data-handin3/true-ann2.fa')
    ann3 = FR.read_fasta_file('./data-handin3/true-ann3.fa')
    ann4 = FR.read_fasta_file('./data-handin3/true-ann4.fa')
    ann5 = FR.read_fasta_file('./data-handin3/true-ann5.fa')

    gen = [gen1['genome1'], gen2['genome2'], gen3['genome3'], gen4['genome4'], gen5['genome5']]
    pred = [gen6['genome6'], gen7['genome7'], gen8['genome8'], gen9['genome9'], gen10['genome10']]
    ann = [ann1['true-ann1'], ann2['true-ann2'], ann3['true-ann3'], ann4['true-ann4'], ann5['true-ann5']]


    pi, A, B = HMM.Init(gen,pred,ann)

    # print("Pi: ",pi)
    # for i in pi:
    #     print(i)
    # print("A: ",A)

    # print("B: ",B)



    # print(A,B,pi)
    # A, B = EM.em_learning(gen[0], pi, A, B, n_iter=100)


