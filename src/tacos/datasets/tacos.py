import pandas as pd
import os
from tacos.datasets.dataset import BaseDataset
from sklearn.model_selection import train_test_split

class Tacos(BaseDataset):

    def __init__(self, data_path, mode="train"):

        self.mode = mode
        super().__init__()
        self.path = os.path.join(data_path, 'tacos')

        self.metadata = pd.read_csv(os.path.join(self.path, 'metadata.csv'))
        self.metadata = self.metadata.set_index('filename')
        self.annotations = pd.read_csv(os.path.join(self.path, 'annotations_strong.csv'))

        test_size = 2000 / len(self.metadata)

        # Stratified split to maintain 'subclass' distribution
        train_df, test_df = train_test_split(
            self.metadata,
            test_size=test_size,
            stratify=self.metadata['subclass'],
            random_state=42
        )

        if mode == 'train':
            self.metadata = train_df
        else:
            self.metadata = test_df

        self.annotations = self.annotations[self.annotations["filename"].isin(self.metadata.index)]

        # all (file, annotator) pairs
        self.unique_annotations = self.annotations[['filename']].drop_duplicates(subset=['filename'])
        self.unique_annotations['fpath'] = self.unique_annotations['filename'].apply(lambda x: os.path.join(self.path , 'audio', x))
        self.unique_annotations['fname'] = self.unique_annotations['filename']

        self.annotations_weak = pd.read_csv(os.path.join(self.path, 'annotations_weak.csv'))
        self.annotations_weak = self.annotations_weak[self.annotations_weak["filename"].isin(self.metadata.index)]

        self.annotations_weak.rename(columns={'caption': 'text'}, inplace=True)
        self.annotations_weak["onset"] = self.annotations_weak["filename"].transform(lambda x: 0)
        self.annotations_weak["offset"] = self.annotations_weak["filename"].transform(lambda x: self.metadata.loc[x]["end_time_s"] - self.metadata.loc[x]["start_time_s"])

        self.unique_annotations = self.unique_annotations.merge(self.metadata.reset_index()[['filename', 'subclass']], on='filename')
        excluded_ids_file = "resources/dcase2025_task6_excluded_freesound_ids.csv"
        if not os.path.exists(excluded_ids_file):
            if not os.path.exists("resources"):
                os.makedirs("resources")
            print("Downloading excluded freesound ids...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/saubhagyapandey27/enhanced_audio_retrieval_dcase25_submission/refs/heads/master/resources/dcase2025_task6_excluded_freesound_ids.csv",
                excluded_ids_file
            )
        forbidden_files = pd.read_csv(excluded_ids_file)
        forbidden_files["filename"] = forbidden_files["sound_id"].transform(lambda x: x + '.mp3')
        self.unique_annotations = self.unique_annotations.loc[~self.unique_annotations["filename"].isin(forbidden_files["filename"])]


    def __getmetadata__(self, index):

        if isinstance(index, tuple):
            return self.unique_annotations.iloc[index[0]][index[1]]

        fn = self.unique_annotations.iloc[index]['filename']

        annotations_weak = self.annotations_weak.loc[self.annotations_weak['filename'] == fn]

        annotations_strong = self.annotations.loc[self.annotations['filename'] == fn]
        # similarity_scores = self.similarity_scors.loc[(self.similarity_scors["annotator"] == annotator) & (self.similarity_scors['filename'] == fn)]

        # if len(similarity_scores) == 0:
        #    similarity_scores = [[3] * len(annotations_strong["text"])] * len(annotations_strong["text"])
        # elif len(similarity_scores) == 1:
        #    import ast
        #    assert annotations_strong["text"].to_list() ==  ast.literal_eval(similarity_scores.iloc[0]["text"])
        #    similarity_scores = ast.literal_eval(similarity_scores.iloc[0]["similarities"])
        #else:
        #    assert False

        return {
            'index': index,
            'subset': 'dev',
            'dataset': 'tacos',
            'sound_link': '',
            'keywords': self.metadata.loc[fn]['keywords'].split(", "),
            'subclass': self.metadata.loc[fn]['subclass'],
            'captions': annotations_weak["text"].to_list(),
            'captions_strong': annotations_strong["text"].to_list(),
            'onsets': annotations_strong["onset"].to_list(),
            'offsets': annotations_strong["offset"].to_list(),
            # 'similarities': similarity_scores,
            'license': '',
            'sr': 32000,
            # 'audio': self.preloaded_audios[index] if index in self.preloaded_audios else self._custom_load_audio(self, index),
            'manufacturer': '',
            'duration': 30,
            'start_end_samples': '',
            'sound_id': int(fn.split('.')[0]),
            'fname': fn
        }

    def at(self, index, column):
        # fn = self.unique_annotations.iloc[index]['filename']
        # annotator = self.unique_annotations.iloc[index]['annotator']
        return self.unique_annotations.iloc[index][column]

    def __len__(self):
        return len(self.unique_annotations)

    def __str__(self):
        return f'Tacos_{self.mode}'


if __name__ == '__main__':
    from collections import Counter
    train_ds = Tacos('data')
    # train_ds.preload_audios()

    # balancing
    labels = train_ds[:, "subclass"].tolist()
    class_counts = Counter(labels)
    num_samples = len(train_ds)

    class_weights = {cls: 1.0 / (count+50) for cls, count in class_counts.items()}
    weights = [class_weights[l] for l in labels]

    import matplotlib.pyplot as plt
    plt.bar(class_weights.keys(), class_weights.values())
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    print(len(train_ds))

    for i in range(len(train_ds)):
        train_ds[i]

    pass