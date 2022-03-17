from multiprocessing import cpu_count, Pool
import pandas as pd
import json
import os
from glob import glob
from functools import partial

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from nice_bids.paths import BIDSPath, EEGPath, DerivativePath

def load_path(filepath, root, participants):
    return EEGPath(
        os.path.join(root, filepath),
        participants=participants
    )

class NICEBIDS:

    def __init__(self, root:str, 
                 sub:str=None, ses:str=None,
                 task:str=None, acq:int=None, rjust:int=2) -> None:

        self.subset = {
            'sub': sub,
            'ses': ses,
            'task': task,
            'acq': acq
        }
        self.subset = {k:v for k,v in self.subset.items() if v is not None}
        self.rjust = rjust
        self.root = root
        self.participants_descriptions = None
        self.participants = None

        self._read_participants()
        self._read_files()
        self._read_derivatives()
        self._create_metadata()

    def _read_participants(self):
        print('Reading participants metadata')
        self.participants = pd.read_csv(
            os.path.join(self.root, 'participants.tsv'),
            sep='\t',
            index_col='participant_id',
            encoding='utf8'
        )

        # Remove participants not selected
        if 'sub' in self.subset:
            fileter_mask = self.participants.index == self.subset['sub']
            self.participants = self.participants[fileter_mask]

        participants_json = os.path.join(self.root, 'participants.json')
        if os.path.exists(participants_json):
            with open(participants_json, 'r') as json_file:
                self.participants_descriptions = json.load(json_file)

    def _read_files(self):
        print('Loading data')
        
        sub = self.subset['sub'] if 'sub' in self.subset else '*'
        ses = self.subset['ses'] if 'ses' in self.subset else '*'
        task = self.subset['task'] if 'task' in self.subset else '*'
        acq = self.subset['acq'] if 'acq' in self.subset else '*'

        if acq != '*':
            acq = str(acq).rjust(self.rjust, '0')
        
        self.files = glob(os.path.join(
            self.root,
            f'sub-{sub}',
            f'ses-{ses}',
            'eeg',
            f'sub-{sub}_ses-{ses}_task-{task}_acq-{acq}*_eeg.*'
        ))

        # Filter incorrect recording filenames
        self.files = list(filter(BIDSPath.correct_filepath, self.files))
        
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
            chunksize=1,
            leave=False
        ) 

    def _create_metadata(self):
        print('Creating metadata')
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
            NICEBIDS.query_filter,
            sub=sub, task=task, ext=ext, ses=ses,
            acq=acq, run=run, suffix=suffix
        )
        
        return list(filter(query, self.files))
    
    def to_df(self, sub:str=None, task:str=None, ses:str=None,
                    acq:int=None, run:int=None):

        res = self.metadata.copy()

        fields_to_filter = {
            'participant_id': sub,
            'task': task,
            'ses': ses,
            'acq': acq,
            'run': run
        }

        for (field_name, val) in fields_to_filter.items():
            if val is not None:
                res = res[res[field_name] == val]
        
        return res

    def _read_derivatives(self):
        print('Reading derivatives')

        self.derivative_files = []

        sub = self.subset['sub'] if 'sub' in self.subset else '*'
        ses = self.subset['ses'] if 'ses' in self.subset else '*'
        task = self.subset['task'] if 'task' in self.subset else '*'
        acq = self.subset['acq'] if 'acq' in self.subset else '*'

        derivative_root = os.path.join(self.root, 'derivatives')
        derivative_folders = filter(
            lambda folder: os.path.isdir(os.path.join(derivative_root, folder)),
            os.listdir(derivative_root)
        )
        for folder in derivative_folders:
            print('\tFolder:', folder)
            files = glob(os.path.join(
                derivative_root,
                folder,
                f'sub-{sub}',
                f'ses-{ses}',
                'eeg',
                f'sub-{sub}_ses-{ses}_task-{task}_acq-{acq}*_*.*'
            ))

            derivative_files = filter(
                lambda f: BIDSPath.correct_filepath(f, 'derivative'),
                files
            )

            self.derivative_files += [
                DerivativePath(
                    root=os.path.join(self.root, folder, filepath),
                    participants=self.participants
                )
                for filepath in tqdm(derivative_files, leave=False)
            ]

    def get_derivatives(self, derivative:str, 
                        suffix:str=None, ext:str=None,
                        sub:str=None, ses:str=None,
                        task:str=None, acq:int=None, run:int=None):
        
        query = partial(
            NICEBIDS.query_filter,
            sub=sub, task=task, ext=ext, ses=ses,
            acq=acq, run=run, suffix=suffix, derivative=derivative
        )

        return list(filter(query, self.derivative_files))

    def __iter__(self):
        return self.files
    
    def __getitem__(self, idx:int):
        return self.files

    @staticmethod
    def query_filter(
        file, sub:str, task:str, ext:str,
        ses:str, acq:str, run:str, suffix:str, derivative:str=None):

        zipped_fields = zip(
            ['sub', 'task', 'ses', 'acq', 'run'],
            [sub, task, ses, acq, run]
        )

        correct_fields = all([
            v is None or f'{k}-{v}' in str(file)
            for k, v in zipped_fields
        ])

        correct_suffix = suffix is None or f'_{suffix}.' in str(file)

        correct_ext = ext is None or str(file).endswith(f'.{ext}')

        correct_derivative = file.derivative == derivative 

        return (
            correct_fields
            and correct_suffix
            and correct_ext
            and correct_derivative
        )
