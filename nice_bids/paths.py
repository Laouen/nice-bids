from __future__ import annotations
from typing import Union

from pathlib import Path
import copy
import os
import re
import pandas as pd
import json
from glob import glob

_EXTS = ['raw', 'mff', 'vmrk', 'vhdr', 'eeg', 'bdf', 'edf', 'EDF']
_EXTS_REGEX_PATTERN = '(' + '|'.join(_EXTS) + ')'
_FILENAME_REGEX_PATTERN = f'^sub-[0-9a-zA-Z]+(_ses-[0-9]+)?_task-[0-9a-zA-Z]+(_acq-[0-9]+)?(_run-[0-9]+)?_[0-9a-zA-Z]+\.'
_EEG_FILENAME_REGEX_PATTERN = f'{_FILENAME_REGEX_PATTERN}{_EXTS_REGEX_PATTERN}$'
_DERIVATIVE_FILENAME_REGEX_PATTERN = f'{_FILENAME_REGEX_PATTERN}.*$'
_DIRPATH_REGEX_PATTERN = 'sub-[0-9a-zA-Z]+/(ses-[0-9]+/)?eeg$'
_FILEPATH_HUMAN_PATTERN = '<root_path>/sub-<label>[/ses-<label>]/eeg/sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>]_<suffix>.<extension>'
_FILENAME_REGEX_DICT = {
    'eeg': _EEG_FILENAME_REGEX_PATTERN,
    'derivative': _DERIVATIVE_FILENAME_REGEX_PATTERN
}


class BIDSPath:

    def __init__(self, 
                 root: Union[str,BIDSPath], sub:str=None, task:str=None, ext:str=None,
                 suffix:str=None, derivative:str=None, ses:str=None, acq:int=None, run:int=None,
                 rjust:int=2, metadata:dict=None) -> None:

        if isinstance(root, BIDSPath):
            self._copy_constructor(root)
            return
        
        elif all([f is None for f in [sub, task, suffix, ext]]):
            self._filepath_parser_constructor(root, metadata)
        
        else:
            self._default_constructor(
                root, sub, task, ext, suffix, derivative,
                ses, acq, run, rjust, metadata
            )

    def _copy_constructor(self, other:BIDSPath):
        self._set(**other.fields)
        self.path = Path(other.path)
        self.metadata = copy.deepcopy(other.metadata)

    def _filepath_parser_constructor(self, 
                                     filepath:str,
                                     metadata:dict=None):

        # Extract root and derivative from filepath
        idx = re.search('/sub-[0-9a-zA-Z]+/(ses-[0-9]+/)?eeg', filepath)
        if idx is None:
            raise ValueError(
                'Filepath is not a valid BIDS path:\n'
                f'{filepath}\n'
                f'Valid BID paths follow the pattern:\n'
                f'{_FILEPATH_HUMAN_PATTERN}'
            )
        idx = idx.start()
        root = filepath[:idx]
        root_dirs = root.split('/')
        if len(root_dirs) > 1 and root_dirs[-2] == 'derivatives':
            root = str(Path(root).parent.parent)
            derivative = root_dirs[-1]
        else:
            derivative = None

        self._default_constructor(
            root=root,
            metadata=metadata,
            derivative=derivative,
            **BIDSPath.parse_filepath(filepath, filename_regex='derivative')
        )

    def _default_constructor(self,
        root: str, sub:str=None, task:str=None, ext:str=None,
        suffix:str=None, derivative:str=None, ses:str=None, acq:int=None,
        run:int=None, rjust:int=2, metadata:dict=None):

        if any([f is None for f in [sub, task, suffix, ext]]):
            print([sub, task, suffix, ext])
            raise ValueError('parameters sub, task, suffix, ext '
                             'must be all different than None')
        
        if any([not re.match('^[0-9a-zA-Z]+$', f) for f in [sub, task, suffix, ext]]):
            raise ValueError('parameters sub, task, suffix, ext '
                             'must be all be non empty alphanumeric strings')
        
        if any([f is not None and not isinstance(f, int) for f in [run, acq]]):
            raise ValueError('opional parameters run and acq must be integer numbers')

        self._set(root, sub, task, ext, suffix, derivative, ses, acq, run, rjust)
        self.path = self._build_path()
        self.metadata = self._get_metadata(metadata)

    def _set(self, root:str, sub:str, task:str, ext:str,
             suffix:str, derivative:str, ses:str,
             acq:int, run:int, rjust:int):

        self.root = root
        self.derivative = derivative
        self.derivative_path = f'derivatives/{derivative}' if derivative is not None else ''
        self.sub = sub
        self.task = task
        self.ses = ses
        self.acq = acq
        self.run = run
        self.ext = ext

        self.rjust = rjust
        self.suffix = suffix

        self.fields = {
            'root': root,
            'sub': sub,
            'task': task,
            'ext': ext,
            'suffix': suffix,
            'derivative': derivative,
            'ses': ses,
            'acq': acq,
            'run': run,
            'rjust': rjust
        }

    def _get_str_fields(self) -> str:
        fields = [
            ['sub', self.sub],
            ['ses', self.ses],
            ['task', self.task],
            ['acq', str(self.acq).rjust(self.rjust, '0') if isinstance(self.acq, int) else self.acq],
            ['run', str(self.run).rjust(self.rjust, '0') if isinstance(self.run, int) else self.run]
        ]

        # Remove not specified fields (i.e. None fields)
        fields = [f for f in fields if f[1] is not None]
        return "_".join(["-".join(f) for f in fields])
    
    def _build_path(self) -> Path:

        str_fields = self._get_str_fields()

        return Path(os.path.join(
            self.root,
            self.derivative_path, # if derivative_path is '' then this is not included in the path
            f'sub-{self.sub}',
            f'ses-{self.ses}' if self.ses is not None else '',
            'eeg',
            f'{str_fields}_{self.suffix}.{self.ext}'
        ))
    
    def _build_sidecar_path(self) -> Path:

        fields = [
            ['sub',self.sub],
            ['ses',self.ses],
            ['task',self.task],
            ['acq', str(self.acq).rjust(self.rjust, '0') if isinstance(self.acq, int) else self.acq]
        ]

        # Remove not specified fields (i.e. None fields)
        fields = [f for f in fields if f[1] is not None]
        filename = "_".join(["-".join(f) for f in fields])

        # NOTE: sidecar_patu is always the one in the subject directory and not
        # in the derivatives folder because of that, I don't add derivative root
        # to the siedcar path
        return Path(os.path.join(
            self.root,
            f'sub-{self.sub}',
            f'ses-{self.ses}' if self.ses is not None else '',
            'eeg',
            f'{filename}_eeg.json'
        ))
    
    # TODO: extend the suffixes allowed for sidecars or add a raw data asociation
    # to the derivative in order to know which sidecar to consider as associated
    # if we have other neuroimages than eeg 
    def _get_associated_sidecar_paths(self) -> Path:

        paths = []
        pattern = ''
        for field in ['sub', 'ses', 'task', 'acq', 'run']:
            field_value = self.fields.get(field, None)
            if field_value is None:
                continue
            pattern += f'{field}-{field_value}_'
            paths += glob(f'{self.root}/**/{pattern}eeg.json', recursive=True)
        
        paths += glob(f'{self.root}/**/{pattern}*eeg.json', recursive=True)
        
        # Return all the paths but not the ones in derivatives folder as they are
        # not considered sidecards with metadata but derivatives
        return [
            Path(p) for p in set(paths) # remove duplicates if there are
            if not p.startswith(f'{self.root}/derivatives')
        ]

    def __str__(self) -> str:
        return str(self.path)
    
    def __repr__(self) -> str:
        return str(self.path)

    def _get_metadata(self, metadata:dict=None) -> dict:
        res = metadata if metadata is not None else {}

        '''
        #NOTE: The following code is not used as we are not using sidecars in
        other levels that is not the recording level and looking for sidecars
        at all levels lows down dataset loading time.

        # Update metadata with all the avaiable not derivatives sidecards
        for file in self._get_associated_sidecar_paths():
            with open(file, 'r', encoding='utf8') as sidecar_file:
                res.update(json.load(sidecar_file))
        '''

        # Update metadata with the specific sidecar file (derivative or not) if exist
        json_file = self._build_sidecar_path()
        if os.path.exists(json_file) and os.path.isfile(json_file):
            with open(json_file, 'r', encoding='utf8') as sidecar_file:
                res.update(json.load(sidecar_file))

        res.update({
            'participant_id': self.sub,
            'ses': self.ses if self.ses is not None else 'n/a',
            'task': self.task,
            'acq': self.acq,
            'run': self.run
        })

        return res
    
    @staticmethod
    def parse_filepath(filepath, filename_regex='eeg'):
    
        if not BIDSPath.correct_filepath(filepath, filename_regex):
            raise ValueError(f'filepath has incorrect bids structure: {filepath}')

        filename = os.path.basename(filepath)
        fields = filename.split('.')[0].split('_')

        # Check fields
        params = {
            k:v 
            for k,v in [
                f.split('-')
                for f in fields[:-1] # Ignore suffix
            ]
        }

        # Cast numeric fields to integer
        for field in ['run', 'acq']:
            if field in params.keys():
                params[field] = int(params[field])

        # Parse non field params
        params['suffix'] = fields[-1]
        params['ext'] = '.'.join(filename.split('.')[1:])

        return params
    
    @staticmethod
    def correct_filepath(
        filepath: str,
        filename_regex='eeg') -> bool:

        # Check that is a file or a mff folder
        if not os.path.isfile(filepath) and not filepath.endswith('.mff'):
            return False

        # Check directory path has correct BIDS format inside the root
        dirpath = os.path.dirname(filepath)
        if not re.search(_DIRPATH_REGEX_PATTERN, dirpath):
            return False

        # Check filename
        filename = os.path.basename(filepath)
        if not re.match(_FILENAME_REGEX_DICT[filename_regex], filename):
            return False

        return True


class EEGPath(BIDSPath):

    def __init__(self, 
                 root:Union[str,EEGPath], sub:str=None, task:str=None, 
                 ext:str=None, ses:str=None, acq:int=None, run:int=None,
                 rjust:int=2, metadata:dict=None) -> None:

        
        suffix = None if sub is None else 'eeg'

        super(EEGPath, self).__init__(
            root=root, sub=sub, task=task, ext=ext, 
            suffix=suffix, ses=ses, acq=acq, run=run,
            rjust=rjust, metadata=metadata
        )


class DerivativePath(BIDSPath):

    def __init__(self, 
                 root:Union[str,DerivativePath], derivative:str=None,
                 sub:str=None, task:str=None, suffix:str=None, ext:str=None,
                 ses:str=None, acq:int=None, run:int=None,
                 rjust:int=2, metadata:dict=None) -> None:
        
        super(DerivativePath, self).__init__(
            root=root, derivative=derivative, 
            sub=sub, task=task, suffix=suffix, ext=ext, 
            ses=ses, acq=acq, run=run,
            rjust=rjust, metadata=metadata
        )