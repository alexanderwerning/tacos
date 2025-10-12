import pandas as pd
import os
from src.datasets.dataset import BaseDataset
from aac_datasets import Clotho, WavCaps, AudioCaps


class AACWrapper(BaseDataset):

    def __init__(self, ds, data_path='data'):
        self.ds = ds
        if hasattr(self.ds, "subset"):
            self.subset = ds.subset
        else:
            self.subset = "default"
        super().__init__()


    def __getmetadata__(self, index):

        metadata = self.ds[index]

        # metadata["similarities"] = [[3] * len(metadata["captions"])] * len(metadata["captions"])
        # metadata["onsets"] = [0] * len(metadata["captions"])
        # metadata["offsets"] = [metadata['durations']] * len(metadata["captions"])
        return metadata

    def at(self, index, column):
        if type(column) == list:
            return pd.DataFrame(self.ds.at(index, column))
        return self.ds.at(index, column)

    def __len__(self):
        return len(self.ds)

    def __str__(self):
        return f'AACWrapper_{self.ds.__class__.__name__}_{self.subset}'


if __name__ == '__main__':
    ds = Clotho(subset="eval", root="data", flat_captions=True) # Clotho(subset="dev", root="data", flat_captions=True)

    tc = AACWrapper(ds)
    # tc.preload_audios()
    print(len(tc[0]))