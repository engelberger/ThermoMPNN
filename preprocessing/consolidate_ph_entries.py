import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def drop_duplicate_entries(tmp):
    """Drop cases where mutation and pH measurements are duplicates"""
    arr = np.empty(tmp.shape[0], dtype=object)

    # get unique entries by pH value and key info
    for n, i in enumerate(tmp.index):
        arr[n] = str(tmp['pdb_id_corrected'][i]) + '-' + str(tmp['mutation'][i]) + \
                 '-' + str(tmp['pdb_position'][i]) + '-' + str(tmp['wild_type'][i]) + \
                 '-' + str(tmp['uniprot_id'][i]) + '-' + str(tmp['pHopt'][i])
    tmp['dupe_detector'] = arr
    
    # drop exact duplicates
    tmp = tmp.drop_duplicates(subset=['dupe_detector'])
    print('Shape after dropping explicit duplicate measurements:', tmp.shape)
    
    # handle cases with multiple measurements
    row_list = []
    
    # get unique entries by mutation only
    arr = np.empty(tmp.shape[0], dtype=object)
    for n, i in enumerate(tmp.index):
        arr[n] = str(tmp['pdb_id_corrected'][i]) + '-' + str(tmp['mutation'][i]) + \
                 '-' + str(tmp['pdb_position'][i]) + '-' + str(tmp['wild_type'][i]) + \
                 '-' + str(tmp['uniprot_id'][i])
    tmp['mut_detector'] = arr

    for mut_id in tqdm(tmp['mut_detector'].unique()):
        rows = tmp[tmp['mut_detector'] == mut_id]
        if rows.shape[0] > 1:
            # Multiple measurements for same mutation
            # Check if measurements are within uncertainty range
            ph_values = rows['pHopt'].values
            uncertainties = rows['pH_uncertainty'].values if 'pH_uncertainty' in rows.columns else np.full_like(ph_values, 0.1)
            
            # Calculate mean and standard deviation
            mean_ph = np.mean(ph_values)
            std_ph = np.std(ph_values)
            
            if std_ph <= max(uncertainties):
                # Measurements are consistent within uncertainties
                # Keep the measurement with lowest uncertainty, or mean if uncertainties are equal
                best_row = rows.iloc[np.argmin(uncertainties)]
                best_row['pHopt'] = mean_ph
                best_row['pH_uncertainty'] = np.sqrt(np.mean(uncertainties**2))  # Combined uncertainty
                row_list.append(pd.DataFrame([best_row]))
            else:
                # Measurements are inconsistent
                print(f"Warning: Inconsistent pH measurements for {mut_id}: {ph_values}")
                # Keep all measurements but flag them
                rows['pH_measurement_conflict'] = True
                row_list.append(rows)
        else:
            # Single measurement
            row_list.append(rows)
    
    print('Rows after pH consolidation:', len(row_list))
    return pd.concat(row_list, axis=0)


def split_monomers_oligomers(df):
    """Split dataset into monomers and oligomers based on oligomeric state or structure source"""
    # For AlphaFold structures, treat all as monomers since they are single-chain predictions
    if 'structure_source' in df.columns and df['structure_source'].str.contains('alphafold').any():
        print('Using AlphaFold structures - treating all as monomers')
        monomers = df.copy()
        oligomers = pd.DataFrame(columns=df.columns)
    else:
        # For experimental structures, use oligomeric state
        monomers = df[df['oligomeric_state'] == 'monomer']
        oligomers = df[df['oligomeric_state'] != 'monomer']
    
    print('Split results:')
    print(f'Monomers: {monomers.shape[0]} entries')
    print(f'Oligomers: {oligomers.shape[0]} entries')
    return monomers, oligomers


if __name__ == "__main__":
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    # Define input/output paths
    input_csv = os.path.join(root_dir, 'data', 'ph', '3_ph_wmetadata.csv')
    splits_dir = os.path.join(root_dir, 'data', 'ph', 'splits')
    out_monomers = os.path.join(splits_dir, '4A_ph_monomers.csv')
    out_oligomers = os.path.join(splits_dir, '4B_ph_oligomers.csv')
    out_consolidated = os.path.join(root_dir, 'data', 'ph', '4_ph_consolidated.csv')

    # Load the dataset with metadata
    df = pd.read_csv(input_csv)
    print('Initial dataset shape:', df.shape)

    # Split into monomers and oligomers
    monomers, oligomers = split_monomers_oligomers(df)
    print('Monomers:', monomers.shape)
    print('Oligomers:', oligomers.shape)

    # Create splits directory if it doesn't exist
    os.makedirs(splits_dir, exist_ok=True)

    # Save split datasets
    monomers.to_csv(out_monomers, index=False)
    oligomers.to_csv(out_oligomers, index=False)

    # Consolidate pH entries for monomers
    final_cleaned = drop_duplicate_entries(monomers)
    
    # Remove temporary columns and save
    final_cleaned = final_cleaned.loc[:, ~final_cleaned.columns.str.contains('^(dupe_detector|mut_detector)$')]
    final_cleaned.to_csv(out_consolidated, index=False)
    print('Final consolidated dataset shape:', final_cleaned.shape) 