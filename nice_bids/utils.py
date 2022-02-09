import re

# TODO: Check this pattern works
_FILENAME_REGEX_PATTERN = '^sub-[0-9a-zA-Z]*(_ses-[0-9]*)_task-[0-9a-zA-Z]*(_acq-[0-9]*)(_run-[0-9]*)_[0-9a-zA-Z]*\..*$'
_FILENAME_HUMAN_PATTERN = 'sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>]_<suffix>.<extension>'
_DIRPATH_REGEX_PATTERN = 'sub-[0-9a-zA-Z]*/(ses-[0-9]*/)?eeg$'

# TODO: This function can be generalized to other brain images by 
# using the correct image filepath checker 
def _parse_eeg_bids_filename(filepath):
    
    if not _correct_eeg_bids_filepath(filepath):
        raise ValueError('filepath has incorrect bids structure')

    filename = filepath.split('/')[-1]
    fields = filename.split('.')[0].split('_')

    # Check fields
    params = {
        k:v 
        for f in fields[:-1] # ignore the suffix
        for k,v in f.split('-')
    }

    # Cast numeric fields to integer
    for field in ['run', 'acq']:
        if field in params.keys():
            params[field] = int(params[field])

    # Parse non field params
    params['suffix'] = fields[-1]
    params['ext'] = '.'.join(filename.split('.')[1:])

    return params

# TODO: This function can be generalized to other brain images by having 
# the _FILENAME_REGEX_PATTERN to use as parameter
def _correct_eeg_bids_filepath(filepath):

    # Check that is a file
    if not os.path.isfile(filepath):
        return False

    # Check directory path has correct BIDS format inside the root
    dirpath = os.path.dirname(filepath)
    if not re.search(_DIRPATH_REGEX_PATTERN, dirpath):
        return False

    # Check filename
    filename = filepath.split('/')[-1]
    if not re.match(_FILENAME_REGEX_PATTERN, filename):
        return False

    return True

