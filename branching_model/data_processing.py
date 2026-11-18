import numpy as np
from collections import Counter

def parse_fasta(file_path):
    """
    Generator that yields (header, sequence) from a FASTA file.
    """
    with open(file_path, 'r') as f:
        name, seq = None, []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if name: yield name, ''.join(seq)
                name, seq = line, []
            else:
                seq.append(line)
        if name: yield name, ''.join(seq)

def get_empirical_clusters(fasta_path, ignore_ambiguous=True):
    """
    Parses FASTA and counts identical sequences to form clusters.
    
    Args:
        fasta_path (str): Path to the fasta file.
        ignore_ambiguous (bool): If True, removes 'N' and '-' characters before comparison.
        
    Returns:
        np.array: Array of cluster sizes.
    """
    print(f"--- Processing FASTA: {fasta_path} ---")
    counts = Counter()
    total_seqs = 0
    
    for header, sequence in parse_fasta(fasta_path):
        total_seqs += 1
        # Normalize to upper case
        seq_clean = sequence.upper()
        
        if ignore_ambiguous:
            # Remove Ns (ambiguous) and dashes (gaps) to ensure 
            # we are comparing the actual biological information available.
            seq_clean = seq_clean.replace('N', '').replace('-', '')
        
        # Count identical strings
        counts[seq_clean] += 1
        
    cluster_sizes = np.array(list(counts.values()))
    
    print(f"Total Sequences: {total_seqs}")
    print(f"Unique Clusters: {len(cluster_sizes)}")
    print(f"Max Cluster Size: {np.max(cluster_sizes)}")
    
    return cluster_sizes