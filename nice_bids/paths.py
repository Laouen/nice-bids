from __future__ import annotations
from typing import Union

from pathlib import Path
import copy
import os
import re
import pandas as pd
import json
from glob import glob

from scipy.misc import derivative


class BIDSPath:

    def __init__(self, 
                 root: Union[str,BIDSPath], sub:str=None, task:str=None, ext:str=None,
                 suffix:str=None, derivative:str=None, ses:str=None, acq:int=None, run:int=None,
                 rjust:int=2, participants:pd.DataFrame=None) -> None:

        if isinstance(root, BIDSPath):
            self._copy_constructor(root)
            return

        if any([f is None for f in [sub, task, suffix, ext, rjust]]):
            raise ValueError('parameters sub, task, suffix, ext, rjust '
                             'must be all different than None')
        
        if any([not re.match('^[0-9a-zA-Z]+$', f) for f in [sub, task, suffix, ext]]):
            raise ValueError('parameters sub, task, suffix, ext '
                             'must be all be non empty alphanumeric strings')
        
        if any([f is not None and not isinstance(f, int) for f in [run, acq]]):
            raise ValueError('opional parameters run and acq must be integer numbers')

        self._set(root, sub, task, ext, suffix, derivative, ses, acq, run, rjust)
        self.path = self._build_path()
        self.metadata = self._get_metadata(participants)

    def _copy_constructor(self, other:BIDSPath):
        self._set(**other.fields)
        self.path = Path(other.path)
        self.metadata = copy.deepcopy(other.metadata)

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

        return Path(os.path.join(
            self.root,
            self.derivative_path, # if derivative_path is '' then this is not included in the path
            f'sub-{self.sub}',
            f'ses-{self.ses}' if self.ses is not None else '',
            'eeg',
            f'{filename}_{self.suffix}.json'
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

    def _get_metadata(self, participants):
        if participants is None:
            participants = pd.read_csv(
                os.path.join(self.root, 'participants.tsv'),
                sep='\t',
                index_col='participant_id',
                encoding='utf8'
            )

        res = participants.loc[self.sub].to_dict()

        # Update metadata with all the avaiable not derivatives sidecards
        for file in self._get_associated_sidecar_paths():
            with open(file, 'r', encoding='utf8') as sidecar_file:
                res.update(json.load(sidecar_file))

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




class EEGPath(BIDSPath):

    def __init__(self, 
                 root:Union[str,EEGPath], sub:str=None, task:str=None, 
                 ext:str=None, ses:str=None, acq:int=None, run:int=None,
                 rjust:int=2, participants=None) -> None:
        
        super(EEGPath, self).__init__(
            root=root, sub=sub, task=task, ext=ext, 
            suffix='eeg', ses=ses, acq=acq, run=run,
            rjust=rjust, participants=participants
        )



class DerivativePath(BIDSPath):

    def __init__(self, 
                 root:Union[str,DerivativePath], derivative:str=None,
                 sub:str=None, task:str=None, suffix:str=None, ext:str=None,
                 ses:str=None, acq:int=None, run:int=None,
                 rjust:int=2, participants=None) -> None:
        
        super(DerivativePath, self).__init__(
            root=root, derivative=derivative, 
            sub=sub, task=task, suffix=suffix, ext=ext, 
            ses=ses, acq=acq, run=run,
            rjust=rjust, participants=participants
        )