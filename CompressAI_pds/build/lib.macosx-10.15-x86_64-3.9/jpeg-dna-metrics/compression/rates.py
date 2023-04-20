"""Compression performance metrics"""
import argparse
from skimage import io

def open_fasta(fasta_path):
    """Open fasta file and get the list of strands from it"""
    with open(fasta_path, 'r', encoding="UTF-8") as f:
        lines = f.readlines()
    strands = []
    for line in lines:
        # if line[0] in ['A', 'T', 'C', 'G']:
        strands.append(line[:-1])
    return strands

def compression_rate(img_path, fasta_path):
    """Compression rate definition"""
    img = io.imread(img_path)
    n_bits = img.nbytes
    strands = open_fasta(fasta_path)
    length = 0
    for strand in strands:
        length += len(strand)
    return n_bits/length

def nucleotide_rate(img_path, fasta_path):
    """Nucleotide rate definition"""
    img = io.imread(img_path)
    n_pixels = img.shape[0] * img.shape[1]
    strands = open_fasta(fasta_path)
    length = 0
    for strand in strands:
        length += len(strand)
    return length/n_pixels

def main(img_path, fasta_path):
    """Main script"""
    print(nucleotide_rate(img_path, fasta_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('IMGPATH',
                        type=str,
                        help='Image path')
    parser.add_argument('FASTAPATH',
                        type=str,
                        help='Fasta file path')

    args = parser.parse_args()

    img_path = args.IMGPATH
    fasta_path = args.FASTAPATH
    main(img_path, fasta_path)
