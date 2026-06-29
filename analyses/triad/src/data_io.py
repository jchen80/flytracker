import os
import re
import pandas as pd
import glob
from tqdm import tqdm

_KNOWN_TRIAD_TYPES = {'MMF', 'MFF', 'MMMF'}
_SPECIES_MAP = {
    'dmel': 'Dmel', 'mel': 'Dmel',
    'dyak': 'Dyak', 'yak': 'Dyak',
}

def parse_acquisition_metadata(acq, ch_idx=None):
    '''
    Parse acquisition name into metadata fields.

    For single-chamber acquisitions the expected format is:
        {date}_{MMF/MFF}-triad{N}_{species}_...
    For multi-chamber acquisitions both triad types and both species appear in
    the filename in chamber order, e.g.:
        {date}_{MMF}-triad{N}_{MFF}-triad{M}_{species1}_{species2}_...
    Multi-chamber parquet keys carry a _ch{N} suffix which is stripped before
    parsing; the chamber index is inferred from that suffix when ch_idx is None.

    Arguments:
        acq    -- acquisition name string (may include _ch{N} suffix)

    Keyword Arguments:
        ch_idx -- which chamber's metadata to return (0-based); if None,
                  inferred from a trailing _ch{N} suffix, defaulting to 0

    Returns:
        dict with keys: triad_type, species, ch_idx
    '''
    # Strip _ch{N} suffix and infer ch_idx from it if not provided
    ch_match = re.search(r'_ch(\d+)$', acq)
    if ch_match:
        if ch_idx is None:
            ch_idx = int(ch_match.group(1))
        acq = acq[:ch_match.start()]
    if ch_idx is None:
        ch_idx = 0

    fields = [f for f in acq.split('_') if f]

    triad_tokens = [f.split('-')[0] for f in fields if f.split('-')[0] in _KNOWN_TRIAD_TYPES]
    species_tokens = [_SPECIES_MAP[f.lower()] for f in fields if f.lower() in _SPECIES_MAP]

    if not triad_tokens:
        raise ValueError(f"No known triad type ({_KNOWN_TRIAD_TYPES}) found in '{acq}'")
    if not species_tokens:
        raise ValueError(f"No known species token found in '{acq}'")

    triad_type = triad_tokens[ch_idx] if ch_idx < len(triad_tokens) else triad_tokens[0]
    species = species_tokens[ch_idx] if ch_idx < len(species_tokens) else species_tokens[0]

    return {'triad_type': triad_type, 'species': species, 'ch_idx': ch_idx}


def save_processed_df(df, acq, calib, processedmat_dir):
    '''
    Save processed df for a single acquisition as parquet.

    Bakes acquisition metadata (acquisition, triad_type, species, assay_type)
    into the parquet so downstream loaders don't need to re-parse the filename.

    Arguments:
        df              -- processed tracks dataframe
        acq             -- acquisition name
        calib           -- calibration dict (for potential future use, e.g. adding PPM to df)
        processedmat_dir -- directory to save parquet files
    '''
    df = df.copy()
    df['acquisition'] = acq
    try:
        meta = parse_acquisition_metadata(acq)
        df['triad_type'] = meta['triad_type']
        df['species']    = meta['species']
        df['assay_type'] = f"{meta['species']}_{meta['triad_type']}"
    except ValueError as e:
        print(f"Warning: could not parse metadata for '{acq}': {e}")
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

        # Use metadata baked into parquet; fall back to filename parsing for old files
        if 'triad_type' not in df.columns or 'species' not in df.columns:
            meta = parse_acquisition_metadata(acq)
            df['acquisition'] = acq
            df['triad_type'] = meta['triad_type']
            df['species']    = meta['species']
            df['assay_type'] = f"{meta['species']}_{meta['triad_type']}"
        elif 'acquisition' not in df.columns:
            df['acquisition'] = acq
        if 'assay_type' not in df.columns:
            df['assay_type'] = df['species'] + '_' + df['triad_type']

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