import os
import logging
import requests
import pandas as pd
from tqdm import tqdm
import time
import concurrent.futures
from functools import partial

# Set testing mode
TESTING = False
MAX_TEST_DOWNLOADS = 50
MAX_WORKERS = 10  # Number of parallel downloads

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_unique_uniprot_ids(csv_path, testing=TESTING):
    df = pd.read_csv(csv_path)
    uniprot_ids = df['Accession'].unique()
    if testing:
        uniprot_ids = uniprot_ids[:MAX_TEST_DOWNLOADS]
        logging.info(f"TESTING MODE: Limited to {MAX_TEST_DOWNLOADS} UniProt IDs")
    logging.info(f"Found {len(uniprot_ids)} unique UniProt IDs")
    return uniprot_ids

def construct_alphafold_url(uniprot_id):
    return f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

def verify_existing_pdbs(output_dir):
    valid_files = []
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.pdb'):
                file_path = os.path.join(output_dir, file)
                if os.path.getsize(file_path) > 0:
                    valid_files.append(file)
    logging.info(f"Found {len(valid_files)} existing valid PDB files")
    return valid_files

def load_mapping_file(mapping_file):
    if os.path.exists(mapping_file):
        df = pd.read_csv(mapping_file)
        return {
            'uniprot_id': df['uniprot_id'].tolist(),
            'pdb_file': df['pdb_file'].tolist(),
            'status': df['status'].tolist()
        }
    return {'uniprot_id': [], 'pdb_file': [], 'status': []}

def download_pdb_with_retry(url, output_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                f.write(response.content)
            if os.path.getsize(output_path) < 1000:  # Basic size check
                raise ValueError("Downloaded file is too small")
            return True
        except (requests.exceptions.RequestException, ValueError) as e:
            logging.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # Exponential backoff
            continue
    return False

def download_single_structure(uniprot_id, output_dir):
    url = construct_alphafold_url(uniprot_id)
    output_path = os.path.join(output_dir, f"{uniprot_id}.pdb")
    
    if download_pdb_with_retry(url, output_path):
        logging.info(f"Successfully downloaded {uniprot_id}")
        return (uniprot_id, f"{uniprot_id}.pdb", 'success')
    else:
        logging.warning(f"Failed to download {uniprot_id} after all retries")
        if os.path.exists(output_path):
            os.remove(output_path)
        return (uniprot_id, '', 'failed')

def download_alphafold_structures(uniprot_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mapping_file = os.path.join(output_dir, 'uniprot_pdb_mapping.csv')
    mapping_data = load_mapping_file(mapping_file)
    
    # Skip already processed IDs
    processed_ids = set(mapping_data['uniprot_id'])
    uniprot_ids = [uid for uid in uniprot_ids if uid not in processed_ids]
    
    if len(uniprot_ids) == 0:
        logging.info("All structures have been processed already")
        return
    
    logging.info(f"Downloading {len(uniprot_ids)} new structures using {MAX_WORKERS} workers")
    success_count = 0
    fail_count = 0
    
    # Create a partial function with fixed output_dir
    download_fn = partial(download_single_structure, output_dir=output_dir)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all download tasks
        future_to_id = {executor.submit(download_fn, uid): uid for uid in uniprot_ids}
        
        # Process completed downloads with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_id), 
                         total=len(future_to_id),
                         desc="Downloading structures"):
            try:
                uniprot_id, pdb_file, status = future.result()
                mapping_data['uniprot_id'].append(uniprot_id)
                mapping_data['pdb_file'].append(pdb_file)
                mapping_data['status'].append(status)
                
                if status == 'success':
                    success_count += 1
                else:
                    fail_count += 1
                
                # Update mapping file periodically (every 10 structures)
                if (success_count + fail_count) % 10 == 0:
                    pd.DataFrame(mapping_data).to_csv(mapping_file, index=False)
                
            except Exception as e:
                logging.error(f"Error processing {future_to_id[future]}: {str(e)}")
                fail_count += 1
    
    # Final update to mapping file
    pd.DataFrame(mapping_data).to_csv(mapping_file, index=False)
    logging.info(f"Download complete. Success: {success_count}, Failed: {fail_count}")

def main():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    input_csv = os.path.join(root_dir, 'data', 'ph', 'pHopt_data.csv')
    output_dir = os.path.join(root_dir, 'data', 'ph', 'pdbs', 'alphafold')
    
    if TESTING:
        logging.info("Running in TESTING mode - limited to 50 downloads")
    
    uniprot_ids = get_unique_uniprot_ids(input_csv)
    existing_pdbs = verify_existing_pdbs(output_dir)
    download_alphafold_structures(uniprot_ids, output_dir)

if __name__ == "__main__":
    main() 