from pathlib import Path
import os
import re

# TODO: Reparametrize EEG from general structures and separate in a base NBPath 
# class and a derived EEGPath class
class EEGPath:

    def __init__(self, 
                 root: str, sub: str, task: str, ext:str,
                 ses:str=None, acq:int=None, run:int=None,
                 suffix:str='eeg', rjust:int=2) -> None:
        
        if any([f is None for f in [sub, task, suffix, ext, rjust]]):
            raise ValueError('parameters sub, task, suffix, ext, rjust '
                             'must be all different than None')
        
        if any([not re.match('^[0-9a-zA-Z]+$', f) for f in [sub, task, suffix, ext]]):
            raise ValueError('parameters sub, task, suffix, ext '
                             'must be all be non empty alphanumeric strings')
        
        if any([f is not None and not isinstance(f, int) for f in [run, acq]]):
            raise ValueError('parameters run and acq must be int or None ')
                            
        
        self.root = root
        self.sub = sub
        self.task = task
        self.ses = ses
        self.acq = acq
        self.run = run
        self.ext = ext

        self.rjust = rjust
        self.suffix = suffix

        self.path = self._build_path()


    def _build_path(self) -> Path:

        fields = [
            ['sub',self.sub],
            ['ses',self.ses],
            ['task',self.task],
            ['acq', str(self.acq).rjust(self.rjust, '0') if isinstance(self.acq, int) else self.acq],
            ['run', str(self.run).rjust(self.rjust, '0') if isinstance(self.run, int) else self.run]
        ]

        # Remove not specified fields (i.e. None fields)
        fields = [f for f in fields if f[1] is not None]
        filename = "_".join(["-".join(f) for f in fields])

        return Path(os.path.join(
            self.root,
            f'sub-{self.sub}',
            f'ses-{self.ses}' if self.ses is not None else '',
            'eeg',
            f'{filename}_{self.suffix}.{self.ext}'
        ))

    def __str__(self) -> str:
        return str(self.path)
    
    def __repr__(self) -> str:
        return str(self.path)

    # TODO: implement
    def get_metadata(self):
        pass