from multiprocessing import cpu_count, Pool
import pandas as pd
import json
import os
from glob import glob
from functools import partial
from tqdm.contrib.concurrent import process_map

from nice_bids.utils import (
    _correct_bids_filepath,
    _parse_bids_filename,
    query_filter
)
from nice_bids.paths import EEGPath, DerivativePath

def load_path(filepath, root, participants):
    return EEGPath(root=root,
                **{
                    k:v 
                    for k,v in _parse_bids_filename(filepath).items()
                    if k != 'suffix'
                },
                participants=participants
            )

class NICEBIDS:

    def __init__(self, root:str) -> None:

        self.root = root
        self.participants_descriptions = None
        self.participants = None

        self._read_participants()
        self._read_files()
        self._read_derivatives()
        self._create_metadata()


    def _read_participants(self):
        self.participants = pd.read_csv(
            os.path.join(self.root, 'participants.tsv'),
            sep='\t',
            index_col='participant_id',
            encoding='utf8'
        )

        participants_json = os.path.join(self.root, 'participants.json')
        if os.path.exists(participants_json):
            with open(participants_json, 'r') as json_file:
                self.participants_descriptions = json.load(json_file)

    def _read_files(self):
        
        self.files = glob(os.path.join(
            self.root,
            'sub-*',
            'ses-*',
            'eeg',
            'sub-*_task-*_eeg.*'
        ))

        # filter incorrect recording filenames
        self.files = list(filter(_correct_bids_filepath, self.files))
        
        load_file_func = partial(
            load_path,
            root=self.root,
            participants=self.participants
        )

        # Parallel loading with a progress bar
        self.files = process_map(
            load_file_func,
            self.files,
            max_workers=cpu_count(),
            chunksize=1
        ) 

    def _create_metadata(self):
        self.metadata = pd.DataFrame(
            [file.metadata for file in self.files]
        )

        grouping_cols = ['participant_id', 'ses', 'task', 'acq']

        for rec, group in self.metadata.groupby(grouping_cols):
            if len(group) == 1 and group['run'].values[0] != 1:
                raise ValueError(f'Single files metadata incorrect runs != 1 {rec}')

            t_dup = group.duplicated(subset=[c for c in group.columns if c != 'run'], keep=False).sum()
            if t_dup != len(group) and len(group) > 1:
                raise ValueError(f'Files metadata inconsistent across runs {rec}')

        self.metadata.drop_duplicates(
            subset=grouping_cols,
            keep='first',
            ignore_index=True,
            inplace=True
        )

        self.metadata = self.metadata.reindex(
            [*grouping_cols, *[c for c in self.metadata.columns if c not in grouping_cols]],
            axis=1
        )

        for c in self.metadata.columns:
            if 'date' in c:
                self.metadata[c] = pd.to_datetime(
                    self.metadata[c],
                    errors='ignore'
                )

    def __repr__(self) -> str:
        return f'Subjects: {len(self.participants)}, files: {len(self.files)}'

    def get(self, sub:str=None, task:str=None, ext:str=None, ses:str=None,
                  acq:int=None, run:int=None, suffix:str=None):
        
        query = partial(
            query_filter,
            sub=sub, task=task, ext=ext, ses=ses,
            acq=acq, run=run, suffix=suffix
        )
        
        return list(filter(query, self.files))
    
    def to_df(self):
        return self.metadata

    def _read_derivatives(self):

        self.derivative_files = []

        derivative_root = os.path.join(self.root, 'derivatives')
        derivative_folders = filter(
            lambda folder: os.path.isdir(os.path.join(derivative_root, folder)),
            os.listdir(derivative_root)
        )
        for folder in derivative_folders:
            files = glob(os.path.join(
                derivative_root,
                folder,
                'sub-*',
                'ses-*',
                'eeg',
                f'sub-*_task-*_*.*'
            ))

            derivative_files = filter(
                lambda f: _correct_bids_filepath(f, 'derivative'),
                files
            )

            self.derivative_files += [
                DerivativePath(
                    root=self.root,
                    derivative=folder,
                    **_parse_bids_filename(filepath, 'derivative'),
                    participants=self.participants
                )
                for filepath in derivative_files
            ]

    def get_derivatives(self, derivative:str, 
                        suffix:str=None, ext:str=None,
                        sub:str=None, ses:str=None,
                        task:str=None, acq:int=None, run:int=None):

        query = partial(
            query_filter,
            sub=sub, task=task, ext=ext, ses=ses,
            acq=acq, run=run, suffix=suffix, derivative=derivative
        )

        return list(filter(query, self.derivative_files))
