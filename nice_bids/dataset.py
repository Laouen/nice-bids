import pandas as pd
import json
import os
from glob import glob

from nice_bids.utils import _correct_eeg_bids_filepath, _parse_eeg_bids_filename
from nice_bids.paths import EEGPath


class NICEBIDS:

    def __init__(self, root) -> None:

        self.root = root
        self.participants_descriptions = None
        self.participants = None

        self._read_metadata()
        self._read_existent_recordings()


    def _read_metadata(self):
        self.participants = pd.read_csv(
            os.path.join(self.root, 'participants.tsv'),
            sep='\t',
            header=1,
            index_col='participant_id',
            encoding='utf8'
        )

        participants_json = os.path.join(self.root, 'participants.json')
        if os.path.exists(participants_json):
            with open(participants_json, 'r') as json_file:
                self.participants_descriptions = json.load(json_file)
    
    def _read_existent_recordings(self):
        
        recordings = glob(os.path.join(
            self.root,
            'sub-*',
            'ses-*',
            'eeg',
            'sub-*_task-*_eeg.*'
        ))

        # filter incorrect recording filenames
        recordings = filter(_correct_eeg_bids_filepath, recordings)

        self.recordings = [
            EEGPath(
                root=self.root,
                **_parse_eeg_bids_filename(filepath)
            )
            for filepath in recordings
        ]
