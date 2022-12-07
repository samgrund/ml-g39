def write_fasta(dictionary, filename):
    """
    Takes a dictionary and writes it to a fasta file
    Must specify the filename when caling the function
    """

    import textwrap
    with open(filename, "w") as outfile:
        for key, value in dictionary.items():
            outfile.write(">" + key + "\n")
            outfile.write("\n".join(textwrap.wrap(value, 60)))
            outfile.write("\n")
