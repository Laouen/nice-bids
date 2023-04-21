import json
from glob import glob
import pandas as pd
import os
from argparse import ArgumentParser
from tqdm import tqdm
import math

def main(path: str):

    anonymize_sidecars(path)
    anonymize_participants(path)


def anonymize_sidecars(path: str):
    print('Anonymizing BIDS dataset in', path)

    participants_dates = ['dod', 'birthdate']
    sidecar_dates = ['recording_date'] + [f'crs_{i}_date' for i in range(1, 8)]
    sidecard_blacklist = ['erp_exam', 'erp_predict', 'mcs_recovery_comment', 'note']

    participants = pd.read_csv(
        os.path.join(path,'participants.tsv'),
        sep='\t', parse_dates=participants_dates
    )
    participants.set_index('participant_id', inplace=True)

    print('Anonymizing sidecar files')
    # anonymizing the participants sidecar information
    
    sidecards = glob(os.path.join(path,'sub-*/**','*.json'), recursive=True)
    pbar = tqdm(sidecards)
    for sidecar_path in pbar:

        pbar.set_description(sidecar_path)

        sub = 'sub-' + sidecar_path.replace(f'{path}/sub-', '').split('/')[0]
        sub_etiology_date = participants.loc[sub, 'etiology_date']
        sidecar = json.load(open(sidecar_path, 'r'))
        
        # Remove sensitive information
        for key in sidecard_blacklist:
            if key in sidecar:
                sidecar[key] = 'anonymized'

        # Make datetimes relative to the etiology date in days
        for key in sidecar_dates:
            if key in sidecar and not isinstance(sidecar[key], int):
                absolute_date = pd.to_datetime(sidecar[key])
                sidecar[f'{key}_days_from_etiology'] = (absolute_date-sub_etiology_date).days

                if math.isnan(sidecar[f'{key}_days_from_etiology']):
                    sidecar[f'{key}_days_from_etiology'] = 'n/a'

                del sidecar[key]

        json.dump(sidecar, open(sidecar_path, 'w'), indent=4, sort_keys=False)


def anonymize_participants(path: str):
    print('Anonymizing participants.tsv')

    participants_dates = ['dod', 'birthdate']
    participants_blacklist = ['etiology_date', 'ipp']

    participants = pd.read_csv(
        os.path.join(path,'participants.tsv'),
        sep='\t', parse_dates=participants_dates + ['etiology_date']
    )

    # Making participants datetimes relative to the etiology date in days
    etiology_date_col = participants['etiology_date']
    for date_col in participants_dates:
        participants[f'{date_col}_days_from_etiology'] = (
            participants[date_col] - etiology_date_col
        ).dt.days
        participants.drop(date_col, axis=1, inplace=True)

    # Anonymizinf sensitive columns from participants.tsv
    participants[participants_blacklist] = 'anonymized'
    
    # save the annonymized participants tsv
    participants.to_csv(
        os.path.join(path,'participants.tsv'),
        sep='\t', index=False, encoding='utf8'
    )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path', help='path to bids dataset to anonymize')
    args = parser.parse_args()
    main(args.path)