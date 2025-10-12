import os
import h5py
import torch
import shutil
import multiprocessing
from tqdm import tqdm
from src.datasets.audio_loading import _custom_load_audio_mp3, custom_transform



class BaseDataset(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()

        self._memory_cache = {}
        self.transform = custom_transform

    def preload_audios(self, n_workers=4, show_progress=True, reset_cache=False, cache_path="/tmp/paul"):

        os.makedirs(cache_path, exist_ok=True)
        cache_path = os.path.join(cache_path, str(self)+'.h5py')

        if os.path.exists(cache_path) and reset_cache:
            os.remove(cache_path)

        # don't precompute if file exits in data...
        if not os.path.exists(cache_path) and os.path.exists(os.path.join('data', str(self) + '.h5py')):
            print(f"Loading audio cache from data to cache...")
            shutil.copyfile(os.path.join('data', str(self) + '.h5py'), cache_path)

        files = self.at(slice(None), ["fpath", "fname"]).drop_duplicates("fname")
        args = [(row['fname'], row['fpath']) for _, row in files.iterrows()]

        if os.path.exists(cache_path):
            print(f"Loading audio cache from {cache_path} into memory...")
            with h5py.File(cache_path, 'r') as h5file:
                for fname in tqdm(h5file.keys(), desc="Loading into RAM", disable=not show_progress):
                    self._memory_cache[fname] = torch.from_numpy(h5file[fname][:])
            return self

        # Otherwise, create the HDF5 file and preload into memory
        print("Building audio cache:", cache_path)
        with h5py.File(cache_path, 'w') as h5file:
            with multiprocessing.Pool(n_workers) as pool:
                imap_iter = pool.imap(_custom_load_audio_mp3, args)
                if show_progress:
                    imap_iter = tqdm(imap_iter, total=len(args), desc="Building cache")
                for fname, audio in imap_iter:
                    h5file.create_dataset(fname, data=audio, compression="gzip")
                    self._memory_cache[fname] = torch.from_numpy(audio)  # Also keep in memory

        print(f"Audio cache created and loaded into RAM from {cache_path}")
        return self

    def _load_audio(self, index):
        fname = self.at(index, "fname")
        fpath = self.at(index, "fpath")
        if fname not in self._memory_cache:
            return torch.from_numpy(_custom_load_audio_mp3((fname, fpath))[1])
        return self._memory_cache[fname]

    def at(self, index, column):
        raise NotImplementedError

    def __getmetadata__(self, index):
        raise NotImplementedError

    def __getitem__(self, index):

        metadata = self.__getmetadata__(index)
        if isinstance(index, tuple) and "audio" not in index[1]:
            return metadata

        metadata["audio"] = self._load_audio(index)
        return self.transform(metadata)


    def __len__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError