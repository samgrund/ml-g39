def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences

if __name__ == "__main__":
    """
    s = '''
    ;Example FASTA file
    >genomeA
    CGATTAAAGA
    TAGAAATACA
    >annotationA
    CCCCCCNNNN
    NNNNRRRRRR
    '''
    with open('test.fa', 'w') as fp:
        fp.write(s)
    """
    x = read_fasta_file('test.fa')
    print(x['genomeA'])