import os
from Bio.PDB import PDBParser
import pandas as pd
from tqdm import tqdm


def add_structure_metadata(df, pdb_loc):
    """Add structure-related metadata from PDB files"""
    pdb_ids = df['pdb_id_corrected'].unique()
    
    # Initialize metadata columns
    df['oligomeric_state'] = ''
    df['structure_method'] = ''
    df['resolution'] = 0
    
    for pid in tqdm(pdb_ids):
        # get PDB metadata
        fname = os.path.join(pdb_loc, pid + '.pdb')
        parser = PDBParser(PERMISSIVE=True, QUIET=True)
        struct = parser.get_structure('tmp', fname)
        header = struct.header
        
        # add metadata to CSV dataframe
        df.loc[df['pdb_id_corrected'] == pid, 'structure_method'] = header['structure_method']
        df.loc[df['pdb_id_corrected'] == pid, 'resolution'] = header['resolution']
    
    return df


def add_ph_metadata(df):
    """Add pH-specific metadata"""
    # Add temperature column if available
    if 'temperature' not in df.columns and 'Sample Weight' in df.columns:
        df['temperature'] = 298.15  # Assume room temperature if not specified
    
    # Add EC number as enzyme classification if available
    if 'EC Number' in df.columns:
        df['enzyme_class'] = df['EC Number']
    
    # Add organism information
    if 'Organism' in df.columns:
        df['organism'] = df['Organism']
    
    # Add experimental method if available
    if 'method' not in df.columns:
        df['method'] = 'not specified'
    
    # Add pH measurement uncertainty if available
    if 'pH_uncertainty' not in df.columns:
        df['pH_uncertainty'] = 0.1  # Default uncertainty of 0.1 pH units
    
    return df


def main(csv_loc, meta_loc, pdb_loc):
    """Main function to add both structure and pH-specific metadata"""
    # Load validated csv
    df = pd.read_csv(csv_loc)
    
    # Add structure metadata
    df = add_structure_metadata(df, pdb_loc)
    
    # Add pH-specific metadata
    df = add_ph_metadata(df)
    
    # Load and add oligomeric state if metadata file exists
    if os.path.exists(meta_loc):
        odf = pd.read_csv(meta_loc)
        for pid in df['pdb_id_corrected'].unique():
            try:
                df.loc[df['pdb_id_corrected'] == pid, 'oligomeric_state'] = \
                    odf[odf['pdb_id'] == pid]['oligomeric_state'].values[0]
            except (IndexError, KeyError):
                # Handle special cases or missing entries
                df.loc[df['pdb_id_corrected'] == pid, 'oligomeric_state'] = 'unknown'
    
    return df


if __name__ == "__main__":
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    # Define input/output paths
    csv_loc = os.path.join(root_dir, 'data', 'ph', '2_ph_aligned.csv')
    meta_loc = os.path.join(root_dir, 'data', 'ph', 'ph-metadata.csv')  # Optional metadata file
    pdb_loc = os.path.join(root_dir, 'data', 'ph', 'pdbs', 'alphafold')
    out_loc = os.path.join(root_dir, 'data', 'ph', '3_ph_wmetadata.csv')

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_loc), exist_ok=True)

    # Add metadata and save
    df = main(csv_loc, meta_loc, pdb_loc)
    df.to_csv(out_loc, index=False) 