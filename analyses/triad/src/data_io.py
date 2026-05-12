import os
import pandas as pd
import glob
from tqdm import tqdm

def parse_acquisition_metadata(acq):
    '''
    Parse acquisition name into metadata fields.
    Expected format: {date}-{time}_{MMF/MFF}-triad{number}_{species}_...

    Arguments:
        acq -- acquisition name string

    Returns:
        dict with keys: triad_type, species
    '''
    fields = [f for f in acq.split('_') if f]  # filter empty strings from double __
    triad_type = fields[1].split('-')[0]        # 'MMF-triad1' -> 'MMF'
    species = fields[2]                          # 'Dyak'
    return {'triad_type': triad_type, 'species': species}


def save_processed_df(df, acq, calib, processedmat_dir):
    '''
    Save processed df for a single acquisition as parquet.

    Arguments:
        df              -- processed tracks dataframe
        acq             -- acquisition name
        calib           -- calibration dict (for potential future use, e.g. adding PPM to df)
        processedmat_dir -- directory to save parquet files
    '''
    savepath = os.path.join(processedmat_dir, f'{acq}.parquet')
    df.to_parquet(savepath, index=False)
    print(f"Saved processed df to {savepath}")


def load_all_processed_dfs(processedmat_dir, with_courtship_only=True, verbose=True):
    '''
    Load all processed parquet files from processedmat_dir,
    parse acquisition metadata, and concatenate into a single df.

    Arguments:
        processedmat_dir -- directory containing processed parquet files

    Keyword Arguments:
        with_courtship_only -- if True, skip acquisitions without courtship column (default: True)
        verbose             -- if True, print per-file loaded/warning messages (default: True)

    Returns:
        combined_df -- concatenated df with additional columns:
                       acquisition, triad_type, species, assay_type
    '''
    parquet_files = sorted(glob.glob(os.path.join(processedmat_dir, '*.parquet')))
    if len(parquet_files) == 0:
        print(f"No parquet files found in {processedmat_dir}")
        return None

    print(f"Found {len(parquet_files)} processed acquisitions")

    dfs = []
    no_courtship_count = 0
    for fp in tqdm(parquet_files, desc='Loading acquisitions'):
        acq = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_parquet(fp)

        meta = parse_acquisition_metadata(acq)
        df['acquisition'] = acq
        df['triad_type'] = meta['triad_type']
        df['species'] = meta['species']
        df['assay_type'] = f"{meta['species']}_{meta['triad_type']}"

        if with_courtship_only:
            if 'courtship' not in df.columns:
                if verbose:
                    print(f"Warning: 'courtship' column not found in {acq}. Skipping.")
                no_courtship_count += 1
                continue

        dfs.append(df)
        if verbose:
            print(f"  Loaded {acq}: {meta['species']} {meta['triad_type']} ({len(df)} rows)")

    combined_df = pd.concat(dfs, ignore_index=True)
    if with_courtship_only and no_courtship_count > 0:
        print(f"Note: {no_courtship_count} acquisitions skipped (no 'courtship' column).")
    print(f"\nCombined df: {len(combined_df)} rows, "
          f"{combined_df['acquisition'].nunique()} acquisitions, "
          f"assay types: {combined_df['assay_type'].unique().tolist()}")

    return combined_df


def get_assay_dfs(combined_df):
    '''
    Split combined df into per-assay-type dfs.

    Arguments:
        combined_df -- concatenated df with assay_type column

    Returns:
        assay_dfs -- dict mapping assay_type string to df,
                     e.g. {'Dyak_MMF': df, 'Dyak_MFF': df}
    '''
    assay_dfs = {assay: group.copy()
                 for assay, group in combined_df.groupby('assay_type')}
    for assay, df in assay_dfs.items():
        print(f"  {assay}: {df['acquisition'].nunique()} acquisitions, {len(df)} rows")
    return assay_dfs