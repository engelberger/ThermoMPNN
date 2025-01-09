import sys
import os
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import pairwise2
from protein_mpnn_utils import parse_PDB
import concurrent.futures
from functools import partial


def load_csv_data(csv_loc, mapping_file=None):
    """Load CSV data and key column unique IDs, with optional AlphaFold mapping"""
    df = pd.read_csv(csv_loc)

    # Load AlphaFold mapping if provided
    if mapping_file and os.path.exists(mapping_file):
        af_mapping = pd.read_csv(mapping_file)
        # Merge with main dataset using Accession/uniprot_id
        df = df.merge(
            af_mapping[["uniprot_id", "pdb_file", "status"]],
            left_on="Accession",
            right_on="uniprot_id",
            how="left",
        )
        # Only keep entries with successfully downloaded structures
        df = df[df["status"] == "success"]
        print(f"Entries with valid AlphaFold structures: {len(df)}")

    protein_names = df["Organism"].unique()
    # Use AlphaFold PDB files if available, otherwise use original PDB IDs
    if "pdb_file" in df.columns:
        pdb_ids = df["pdb_file"].unique()
    else:
        pdb_ids = df["pdb_id_corrected"].unique()

    sequences = df["Sequence"]
    return df, protein_names, pdb_ids, sequences


def get_df_slice(pdb_id, df):
    """Get relevant data slice for a given PDB ID"""
    if "pdb_file" in df.columns:
        # Using AlphaFold structures
        data_idx = df["pdb_file"] == pdb_id
    else:
        # Using original PDB IDs
        pdb_id = pdb_id.split("_")[0]
        data_idx = df["pdb_id_corrected"].str.contains(pdb_id)

    df_slice = {
        "wt_AA": df["wild_type"][data_idx],
        "mut_AA": df["mutation"][data_idx],
        "pos": df["position"][data_idx],
        "seq": df["Sequence"][data_idx].unique()[0],
        "pH": df["pHopt"][data_idx],  # Include pH values in the slice
    }
    return df_slice, data_idx


def seq1_index_to_seq2_index(align, index):
    """Map index from sequence 1 to sequence 2 using alignment"""
    cur_seq1_index = 0

    # first find the aligned index
    for aln_idx, char in enumerate(align.seqA):
        if char != "-":
            cur_seq1_index += 1
        if cur_seq1_index > index:
            break

    # now the index in seq 2 corresponding to aligned index
    if align.seqB[aln_idx] == "-":
        return None

    seq2_to_idx = align.seqB[: aln_idx + 1]
    seq2_idx = aln_idx
    for char in seq2_to_idx:
        if char == "-":
            seq2_idx -= 1

    if seq2_idx < 0:
        return None

    return seq2_idx


def process_single_structure(pdb, df, pdb_loc, using_alphafold):
    """Process a single structure for alignment"""
    results = {
        "pdb_positions": {},
        "pdb_sequences": {},
        "structure_sources": {},
        "successful": set(),
        "mutations": 0,
    }

    if using_alphafold:
        # For AlphaFold structures, use the full path from mapping
        pdb_path = os.path.join(pdb_loc, os.path.basename(pdb))
    else:
        # For original PDBs
        pdb_fname = f"{pdb}.pdb"
        pdb_path = os.path.join(pdb_loc, pdb_fname)
        if not os.path.exists(pdb_path):
            return results

    # retrieve PDB data
    try:
        pdb_data = parse_PDB(pdb_path)
        pdb_seq = pdb_data[0]["seq"].replace("-", "X")
        del pdb_data  # Free memory
        gc.collect()  # Force garbage collection
    except Exception as e:
        print(f"Error parsing PDB {pdb}: {str(e)}")
        return results

    # retrieve CSV data including pH values
    df_slice, data_idx = get_df_slice(pdb, df)
    align, *rest = pairwise2.align.globalxx(df_slice["seq"], pdb_seq)
    del rest  # Free memory
    iterrows = data_idx[data_idx == True].index

    # iterate over mutations and search through each one
    for idx in iterrows:
        csv_pos = df_slice["pos"][idx] - 1
        if (
            df_slice["seq"][csv_pos] != df_slice["wt_AA"][idx]
        ):  # CSV should be internally consistent
            continue

        new_position = seq1_index_to_seq2_index(align, csv_pos)
        if new_position is not None:
            results["mutations"] += 1
            # check if offset is correct and log if so
            if pdb_seq[new_position] == df_slice["seq"][csv_pos]:
                results["successful"].add(idx)
                # add successful chain/position to list
                results["pdb_positions"][idx] = new_position
                results["structure_sources"][idx] = (
                    "alphafold" if using_alphafold else "pdb"
                )
                results["pdb_sequences"][idx] = pdb_seq.replace("-", "X")

    del align, df_slice  # Free memory
    gc.collect()  # Force garbage collection
    return results


def load_checkpoint(checkpoint_file):
    """Load progress from checkpoint file"""
    if os.path.exists(checkpoint_file):
        return pd.read_csv(checkpoint_file)
    return None


def save_checkpoint(df, checkpoint_file):
    """Save current progress to checkpoint file"""
    df.to_csv(checkpoint_file, index=False)


class CustomAlignment:
    """Class to handle sequence alignment and pH data preservation"""

    def __init__(
        self, csv_loc, out_loc, pdb_loc, af_mapping=None, max_workers=4, batch_size=100
    ):
        # load CSV data with optional AlphaFold mapping
        self.df, self.protein_names, self.pdb_ids, self.sequences = load_csv_data(
            csv_loc, af_mapping
        )

        # Check for checkpoint
        checkpoint_file = out_loc + ".checkpoint"
        checkpoint_df = load_checkpoint(checkpoint_file)

        if checkpoint_df is not None:
            print("Resuming from checkpoint...")
            processed_pdbs = set(checkpoint_df["pdb_file"].unique())
            self.pdb_ids = [pid for pid in self.pdb_ids if pid not in processed_pdbs]
            self.df = checkpoint_df
        else:
            # Initialize result containers
            self.df["pdb_position"] = [[] for _ in range(self.df.shape[0])]
            self.df["pdb_sequence"] = [[] for _ in range(self.df.shape[0])]
            self.df["structure_source"] = [[] for _ in range(self.df.shape[0])]

        # Determine if we're using AlphaFold structures
        using_alphafold = "pdb_file" in self.df.columns

        # Process structures in batches
        for i in range(0, len(self.pdb_ids), batch_size):
            batch_pdbs = self.pdb_ids[i : i + batch_size]
            print(
                f"\nProcessing batch {i//batch_size + 1} of {len(self.pdb_ids)//batch_size + 1}"
            )

            # Create a partial function with fixed arguments
            process_fn = partial(
                process_single_structure,
                df=self.df,
                pdb_loc=pdb_loc,
                using_alphafold=using_alphafold,
            )

            successful_count = 0
            mutation_count = 0

            # Process batch in parallel
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = {executor.submit(process_fn, pdb): pdb for pdb in batch_pdbs}

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Processing structures",
                ):
                    results = future.result()
                    mutation_count += results["mutations"]
                    successful_count += len(results["successful"])

                    # Update successful alignments
                    for idx in results["successful"]:
                        self.df.at[idx, "pdb_position"] = results["pdb_positions"][idx]
                        self.df.at[idx, "pdb_sequence"] = results["pdb_sequences"][idx]
                        self.df.at[idx, "structure_source"] = results[
                            "structure_sources"
                        ][idx]

                print(
                    f"Batch processed {mutation_count} mutations, {successful_count} successful alignments"
                )

                # Save checkpoint after each batch
                save_checkpoint(self.df, checkpoint_file)
                gc.collect()  # Force garbage collection

        # Final cleanup and save
        self.df = self.df.dropna(subset=["pdb_position"]).reset_index(drop=True)
        print("Dataset size after alignment and quality check:", self.df.shape)
        self.df.to_csv(out_loc, index=False)

        # Remove checkpoint file after successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)


if __name__ == "__main__":
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    # Configuration
    csv_loc = os.path.join(root_dir, "data", "ph", "1_ph_cleaned.csv")
    out_loc = os.path.join(root_dir, "data", "ph", "2_ph_aligned.csv")
    pdb_loc = os.path.join(
        root_dir, "data", "ph", "pdbs", "alphafold"
    )  # Use AlphaFold directory
    af_mapping = os.path.join(
        root_dir, "data", "ph", "pdbs", "alphafold", "uniprot_pdb_mapping.csv"
    )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_loc), exist_ok=True)

    # Perform alignment with AlphaFold structures (using 8 worker processes and batch size of 100)
    a = CustomAlignment(
        csv_loc, out_loc, pdb_loc, af_mapping, max_workers=18, batch_size=100
    )
