import os
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import subprocess
import pickle
from tqdm import tqdm


def write_fasta(sequences, output_file):
    """Write sequences to FASTA format"""
    with open(output_file, 'w') as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n{seq}\n")


def run_mmseqs_clustering(fasta_file, output_dir, min_seq_id=0.3):
    """Run MMseqs2 clustering on sequences"""
    # Create MMseqs2 database
    db_name = os.path.join(output_dir, "seqDB")
    subprocess.run([
        "mmseqs", "createdb",
        fasta_file,
        db_name
    ], check=True)

    # Run clustering
    cluster_name = os.path.join(output_dir, "clusters")
    subprocess.run([
        "mmseqs", "cluster",
        db_name,
        cluster_name,
        os.path.join(output_dir, "tmp"),
        f"--min-seq-id", str(min_seq_id),
        "-c", "0.8",
        "--cov-mode", "0"
    ], check=True)

    # Create TSV file
    tsv_name = os.path.join(output_dir, "clusters.tsv")
    subprocess.run([
        "mmseqs", "createtsv",
        db_name,
        db_name,
        cluster_name,
        tsv_name
    ], check=True)

    return tsv_name


def create_splits(clusters_tsv, split_ratios=(0.8, 0.1, 0.1)):
    """Create train/val/test splits from clustering results"""
    # Read clustering results
    clusters = {}
    with open(clusters_tsv, 'r') as f:
        for line in f:
            rep, member = line.strip().split('\t')
            if rep not in clusters:
                clusters[rep] = []
            clusters[rep].append(member)

    # Randomly assign clusters to splits
    cluster_reps = list(clusters.keys())
    np.random.shuffle(cluster_reps)

    n_clusters = len(cluster_reps)
    n_train = int(n_clusters * split_ratios[0])
    n_val = int(n_clusters * split_ratios[1])

    train_clusters = cluster_reps[:n_train]
    val_clusters = cluster_reps[n_train:n_train + n_val]
    test_clusters = cluster_reps[n_train + n_val:]

    # Create splits dictionary
    splits = {
        'train': [],
        'val': [],
        'test': []
    }

    # Assign sequences to splits based on their cluster
    for rep in train_clusters:
        splits['train'].extend(clusters[rep])
    for rep in val_clusters:
        splits['val'].extend(clusters[rep])
    for rep in test_clusters:
        splits['test'].extend(clusters[rep])

    return splits


def main():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    # Define input/output paths
    input_csv = os.path.join(root_dir, 'data', 'ph', '4_ph_consolidated.csv')
    splits_dir = os.path.join(root_dir, 'data', 'ph', 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # Load consolidated pH dataset
    df = pd.read_csv(input_csv)
    print('Dataset shape:', df.shape)

    # Extract unique sequences and their IDs
    sequences = {}
    for _, row in df.iterrows():
        if row['pdb_id_corrected'] not in sequences:
            sequences[row['pdb_id_corrected']] = row['pdb_sequence']

    print('Number of unique sequences:', len(sequences))

    # Write sequences to FASTA
    fasta_file = os.path.join(splits_dir, 'sequences.fasta')
    write_fasta(sequences, fasta_file)

    # Run MMseqs2 clustering
    print('Running MMseqs2 clustering...')
    clusters_tsv = run_mmseqs_clustering(fasta_file, splits_dir)

    # Create train/val/test splits
    print('Creating splits...')
    splits = create_splits(clusters_tsv)

    # Save splits
    splits_file = os.path.join(splits_dir, 'ph_clusters.pkl')
    with open(splits_file, 'wb') as f:
        pickle.dump(splits, f)

    print('Split sizes:')
    for split, seqs in splits.items():
        print(f'{split}: {len(seqs)} sequences')


if __name__ == "__main__":
    main() 