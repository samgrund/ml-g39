import sys
import numpy as np
import FileReader as FR
import HMM

gen = []
hidden = []
if __name__ == "__main__":
    """
    for i in range(0,5):
        filename = sys.argv[i]
        gen[i] = FR.read_fasta_file(filename)
    for i in range(5,10):
        filename = sys.argv[i]
        hidden[i-5] = FR.read_fasta_file(filename)
    """

    full_path = "D:/Aarhus/Courses/MachineLearning/Exercise/ML22/handins/handin3/"

    gen1 = FR.read_fasta_file(full_path + 'data-handin3/genome1.fa')
    ann1 = FR.read_fasta_file(full_path + 'data-handin3/true-ann1.fa')

    gen2 = FR.read_fasta_file(full_path + 'data-handin3/genome2.fa')
    ann2 = FR.read_fasta_file(full_path + 'data-handin3/true-ann2.fa')

    gen3 = FR.read_fasta_file(full_path + 'data-handin3/genome3.fa')
    ann3 = FR.read_fasta_file(full_path + 'data-handin3/true-ann3.fa')

    gen4 = FR.read_fasta_file(full_path + 'data-handin3/genome4.fa')
    ann4 = FR.read_fasta_file(full_path + 'data-handin3/true-ann4.fa')

    gen5 = FR.read_fasta_file(full_path + 'data-handin3/genome5.fa')
    ann5 = FR.read_fasta_file(full_path + 'data-handin3/true-ann5.fa')

    gen6 = FR.read_fasta_file(full_path + 'data-handin3/genome6.fa')
    gen7 = FR.read_fasta_file(full_path + 'data-handin3/genome7.fa')
    gen8 = FR.read_fasta_file(full_path + 'data-handin3/genome8.fa')
    gen9 = FR.read_fasta_file(full_path + 'data-handin3/genome9.fa')
    gen10 = FR.read_fasta_file(full_path + 'data-handin3/genome10.fa')

    gen = [gen1['genome1'], gen2['genome2'], gen3['genome3'], gen4['genome4'], gen5['genome5']]
    pred = [gen6['genome6'], gen7['genome7'], gen8['genome8'], gen9['genome9'], gen10['genome10']]
    ann = [ann1['true-ann1'], ann2['true-ann2'], ann3['true-ann3'], ann4['true-ann4'], ann5['true-ann5']]

    HMM.Init(gen,pred,ann)


