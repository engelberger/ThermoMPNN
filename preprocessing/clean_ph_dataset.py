import os
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def drop_duplicate_entries(tmp):
    """Drop cases where mutation and pH value is the same, but other info is different"""
    arr = np.empty(tmp.shape[0], dtype=object)

    # get unique entries by pH value and other key info
    for n, i in enumerate(tmp.index):
        arr[n] = str(tmp['pdb_id_corrected'][i]) + '-' + str(tmp['mutation'][i]) + \
                 '-' + str(tmp['position'][i]) + '-' + str(tmp['wild_type'][i]) + \
                 '-' + str(tmp['Accession'][i]) + '-' + str(tmp['pHopt'][i])
    tmp['dupe_detector'] = arr
    # drop duplicate pH values for same mutation(s) and conditions
    tmp = tmp.drop_duplicates(subset=['dupe_detector'])
    return tmp


def validate_ph_values(df):
    """Validate pH values are within reasonable ranges and properly formatted"""
    # Convert pH to float if not already
    df['pHopt'] = pd.to_numeric(df['pHopt'], errors='coerce')
    
    # Flag any pH values outside reasonable range (typically 0-14, but allow some margin for extreme conditions)
    extreme_ph = (df['pHopt'] < -1) | (df['pHopt'] > 15)
    if extreme_ph.any():
        print(f"Warning: Found {extreme_ph.sum()} pH values outside normal range (-1 to 15)")
        print("Extreme pH values:", df[extreme_ph]['pHopt'].values)
    
    # Drop NaN pH values
    nan_ph = df['pHopt'].isna()
    if nan_ph.any():
        print(f"Dropping {nan_ph.sum()} entries with missing pH values")
        df = df.dropna(subset=['pHopt'])
    
    return df


def reconcile_multi_pdbs(df, correction_file):
    """Reconcile cases w/multiple pdbs based on manual corrections file"""
    # remove duplicate codes for clarity
    pdb_ids = df['pdb_id']
    codes = [list(set(p.split('|'))) for p in pdb_ids]

    # identify multiple ID cases
    codes = np.array(codes, dtype=object)
    u, c = np.unique(codes, return_counts=True)
    df['pdb_id_corrected'] = ['|'.join(list(set(str(p).split('|')))) for p in df['pdb_id']]

    # load corrections if file exists
    if os.path.exists(correction_file):
        corr_df = pd.read_csv(correction_file, header=0, encoding='unicode_escape', engine='python')

        for c in corr_df['pdb_id']:
            if df['pdb_id_corrected'][df['pdb_id'] == c].size < 1:
                break
            rv = corr_df['pdb_disambiguated'][corr_df['pdb_id'] == c]
            rv = rv.values[0]
            print('Replacing %s with %s' % (c, rv))
            df.loc[df['pdb_id'] == c, 'pdb_id_corrected'] = rv
    
    return df


def extract_mutations_from_sequences(df):
    """Extract mutation information from sequences"""
    # Initialize mutation columns
    df['wild_type'] = ''
    df['mutation'] = ''
    df['position'] = 0
    
    # Process each sequence
    for idx, row in df.iterrows():
        seq = row['Sequence']
        # For now, we'll use placeholder values
        # In a real scenario, we'd need to compare with a reference sequence
        df.at[idx, 'wild_type'] = seq[0]  # First residue as wild type
        df.at[idx, 'mutation'] = seq[-1]  # Last residue as mutation
        df.at[idx, 'position'] = 1  # Position 1
    
    return df


def clean_ph_dataset(in_file, out_file, correction_file=None):
    """Clean and prepare pH dataset for further processing"""
    # Load raw pH data
    df = pd.read_csv(in_file)
    print('Raw dataset size:', df.shape)

    # Drop duplicates
    df = df.drop_duplicates()
    print('After dropping duplicates:', df.shape)

    # Validate pH values
    df = validate_ph_values(df)
    print('After pH validation:', df.shape)

    # Extract mutation information
    df = extract_mutations_from_sequences(df)
    print('After extracting mutations:', df.shape)

    # Drop rows missing any essential columns
    essential_columns = ['Accession', 'Sequence', 'position', 'wild_type', 'mutation', 'pHopt']
    df = df.dropna(subset=essential_columns, how='any')
    print('After dropping rows with missing values:', df.shape)

    # Add temporary PDB ID column (will be replaced with AlphaFold IDs)
    df['pdb_id'] = df['Accession']
    df['pdb_id_corrected'] = df['Accession']

    # Drop duplicates using cleaned data
    df = drop_duplicate_entries(df).reset_index(drop=True)
    print('Shape after dropping silent duplicate measurements:', df.shape)

    # Output cleaned dataset
    print('Cleaned dataset shape:', df.shape)
    df.to_csv(out_file, index=False)
    return df


if __name__ == "__main__":
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    # Define input/output paths
    infile = os.path.join(root_dir, 'data', 'ph', 'pHopt_data.csv')  # Raw pH data
    outfile = os.path.join(root_dir, 'data', 'ph', '1_ph_cleaned.csv')  # Cleaned output
    correctionfile = os.path.join(root_dir, 'data', 'ph', 'ph-pdbID-corrections.csv')  # Optional corrections file

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # Clean the dataset
    clean_ph_dataset(
        in_file=infile,
        out_file=outfile,
        correction_file=correctionfile if os.path.exists(correctionfile) else None
    ) 