import re
import os

# TODO: add more extensions (brainvision, biosemi)
_EXTS = ['raw', 'mff']
_EXTS_REGEX_PATTERN = '(' + '|'.join(_EXTS) + ')'
_FILENAME_REGEX_PATTERN = f'^sub-[0-9a-zA-Z]+(_ses-[0-9]+)?_task-[0-9a-zA-Z]+(_acq-[0-9]+)?(_run-[0-9]+)?_[0-9a-zA-Z]+\.'
_EEG_FILENAME_REGEX_PATTERN = f'{_FILENAME_REGEX_PATTERN}{_EXTS_REGEX_PATTERN}$'
_DERIVATIVE_FILENAME_REGEX_PATTERN = f'{_FILENAME_REGEX_PATTERN}.*$'
_DIRPATH_REGEX_PATTERN = 'sub-[0-9a-zA-Z]+/(ses-[0-9]+/)?eeg$'
_FILENAME_HUMAN_PATTERN = 'sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>]_<suffix>.<extension>'
_FILENAME_REGEX_DICT = {
    'eeg': _EEG_FILENAME_REGEX_PATTERN,
    'derivative': _DERIVATIVE_FILENAME_REGEX_PATTERN
}

def _parse_bids_filename(filepath, filename_regex='eeg'):
    
    if not _correct_bids_filepath(filepath, filename_regex):
        raise ValueError('filepath has incorrect bids structure')

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

def _correct_bids_filepath(
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

def query_filter(file, sub:str, task:str, ext:str,
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