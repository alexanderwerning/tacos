import pandas as pd
import os
from src.datasets.dataset import BaseDataset
from src.datasets.audioset_helpers import as_strong_eval_classes, as_strong_train_classes

class AudioSetStrong(BaseDataset):

    def __init__(self, data_path='data', reduce_classes=True, to_sentences=True):
        super().__init__()

        self.path = os.path.join(data_path, 'audioset_strong')
        self.reduce_classes = reduce_classes

        id_to_name = pd.read_csv(os.path.join(self.path , 'mid_to_display_name.tsv'), sep='\t', header=None, names=['id', 'name'])
        id_to_name = id_to_name.set_index('id')

        self.annotations = pd.read_csv(os.path.join(self.path, 'audioset_eval_strong.tsv'), sep='\t')

        self.annotations["text"] = self.annotations["label"].apply(lambda x: id_to_name.loc[x]['name'])
        self.annotations = self.annotations[~self.annotations["text"].isin(set(as_strong_eval_classes).difference(set(as_strong_train_classes)))]

        if reduce_classes:
            import itertools
            all_categories = list(itertools.chain.from_iterable([l for l in category_to_audioset.values()]))
            self.annotations = self.annotations[self.annotations["text"].isin(all_categories)].copy()
            if to_sentences:
                self.annotations["text"] = self.annotations["text"].transform(lambda x: sound_captions[audioset_to_category[x]])
            else:
                self.annotations["text"] = self.annotations["text"].transform(lambda x: audioset_to_category[x])

        self.files = self.annotations.drop_duplicates(subset='segment_id').copy()
        self.files['id'] = self.files['segment_id'].transform(lambda x: "Y" + "_".join(x.split('_')[:-1]))
        self.files['fpath'] = self.files['id'].apply(lambda x: os.path.join(self.path, 'audio', x + '.mp3'))
        self.files['fname'] = self.files['id'].apply(lambda x: x + '.mp3')

        self.files = self.files[self.files['fpath'].transform(lambda x: os.path.exists(x))]



    def __getmetadata__(self, index):

        if isinstance(index, tuple):
            return self.files.iloc[index[0]][index[1]]

        segment_id = self.files.iloc[index]['segment_id']
        annotations = self.annotations[self.annotations['segment_id'] == segment_id]

        return {
            'fname': self.files.iloc[index]['fname'],
            'subset': 'eval',
            'dataset': 'audioset_strong',
            'segment_id': segment_id,
            'sr': 32000,
            'onsets': annotations["start_time_seconds"].to_list(),
            'offsets': annotations["end_time_seconds"].to_list(),
            'captions_strong': annotations["text"].to_list()
        }

    def at(self, index, column):
        if type(column) == list:
            return pd.DataFrame(self.files.iloc[index][column])
        return self.files.iloc[index][column]

    def __len__(self):
        return len(self.files)

    def __str__(self):
        return f'AudioSet_strong{"_reduced" if self.reduce_classes else ""}'



category_to_audioset = {
    'Speech': ['Speech', 'Conversation', 'Female speech, woman speaking', 'Male speech, man speaking', 'Narration, monologue', 'Child speech, kid speaking', 'Human voice'],
    'Cough': ['Cough'],
    'Sneeze': ['Sneeze'],
    'Laughter': ['Laughter', 'Chuckle, chortle', 'Giggle', 'Belly laugh', 'Baby laughter'],
    'Crying': ['Crying, sobbing', 'Baby cry, infant cry'],
    'Shout': ['Shout', 'Yell', 'Screaming', 'Children shouting', 'Battle cry'],
    'Hiccup': ['Hiccup'],
    'Snoring': ['Snoring'],
    'Footsteps': ['Walk, footsteps'],
    'Clapping': ['Applause', 'Clapping'],
    'Dog Bark': ['Bark'],
    'Cat Meow': ['Meow'],
    'Sheep_Goat Bleat': ['Bleat'],
    'Cow Moo': ['Moo'],
    'Pig Oink': ['Oink'],
    'Horse Neigh': ['Neigh, whinny'],
    'Rooster Crow': ['Crowing, cock-a-doodle-doo', 'Chicken, rooster'],
    'Bird Chirp': ['Chirp, tweet', 'Bird vocalization, bird call, bird song'],
    'Insect Buzz': ['Insect', 'Bee, wasp, etc.', 'Mosquito', 'Fly, housefly'],
    'Singing': ['Singing', 'Male singing', 'Female singing', 'Child singing', 'Synthetic singing', 'Chant', 'Yodeling', 'Choir'],
    'Bell': ['Bell', 'Church bell', 'Mechanical bell', 'Jingle bell', 'Cowbell', 'Bicycle bell'],
    'Car': ['Car', 'Race car, auto racing', 'Car passing by'],
    'Truck': ['Truck'],
    'Motorcycle': ['Motorcycle'],
    'Bicycle': ['Bicycle, tricycle'],
    'Bus': ['Bus'],
    'Train': ['Train', 'Railroad car, train wagon', 'Rail transport', 'Train wheels squealing'],
    'Ship_Boat': ['Boat, Water vehicle', 'Sailboat, sailing ship', 'Ship'],
    'Helicopter': ['Helicopter'],
    'Airplane': ['Fixed-wing aircraft, airplane', 'Aircraft engine', 'Jet engine'],
    'Chainsaw': ['Chainsaw'],
    'Power Drill': ['Drill', 'Dental drill, dentist\'s drill'],
    'Hammer': ['Hammer'],
    'Jackhammer': ['Jackhammer'],
    'Power Saw': ['Power saw, circular saw, table saw', 'Sawing'],
    'Lawn Mower': ['Lawn mower'],
    'Vacuum Cleaner': ['Vacuum cleaner'],
    'Sewing Machine': ['Sewing machine'],
    'Washing Machine': ['Washing machine'],
    'Alarm': ['Alarm', 'Smoke detector, smoke alarm', 'Alarm clock', 'Alert', 'Fire alarm'],
    'Siren': ['Siren', 'Ambulance (siren)', 'Police car (siren)', 'Fire engine, fire truck (siren)', 'Civil defense siren'],
    'Horn Honk': ['Vehicle horn, car horn, honking, toot', 'Honk', 'Air horn, truck horn', 'Train horn', 'Foghorn'],
    'Beep_Bleep': ['Beep, bleep', 'Reversing beeps', 'Telephone dialing, DTMF'],
    'Doorbell': ['Doorbell', 'Ding-dong'],
    'Rain': ['Rain', 'Rain on surface', 'Raindrop'],
    'Thunder': ['Thunder', 'Thunderstorm'],
    'Wind': ['Wind', 'Howl (wind)', 'Wind noise (microphone)'],
    'Fire': ['Fire', 'Wildfire'],
    'Waves': ['Waves, surf', 'Ocean'],
    'Stream_River': ['Stream, river', 'Waterfall']
}

sound_captions = {
    "Speech": "Someone is speaking.",
    "Cough": "Someone is coughing.",
    "Sneeze": "Someone is sneezing.",
    "Laughter": "Someone is laughing.",
    "Crying": "Someone is crying.",
    "Shout": "Someone is shouting.",
    "Hiccup": "A person hiccups.",
    "Snoring": "Someone is snoring.",
    "Footsteps": "Footsteps of a person walking.",
    "Clapping": "People are clapping.",
    "Dog Bark": "A dog is barking.",
    "Cat Meow": "A cat is meowing.",
    "Sheep_Goat Bleat": "A sheep or goat is bleating.",
    "Cow Moo": "A cow is mooing.",
    "Pig Oink": "A pig is oinking.",
    "Horse Neigh": "A horse is neighing.",
    "Rooster Crow": "A rooster is crowing.",
    "Bird Chirp": "Birds are chirping.",
    "Insect Buzz": "Insects are buzzing.",
    "Singing": "Someone is singing.",
    "Bell": "A bell is ringing.",
    "Car": "A car is driving by.",
    "Truck": "A truck is driving by.",
    "Motorcycle": "A motorcycle is passing.",
    "Bicycle": "A bicycle passes by.",
    "Bus": "A bus engine runs.",
    "Train": "A train passes by.",
    "Ship_Boat": "A boat or ship is moving through water.",
    "Helicopter": "A helicopter is flying overhead.",
    "Airplane": "An airplane is flying by.",
    "Chainsaw": "A chainsaw is running.",
    "Power Drill": "A power drill is running.",
    "Hammer": "A hammer is hitting something.",
    "Jackhammer": "A jackhammer is breaking the ground.",
    "Power Saw": "A power saw is cutting.",
    "Lawn Mower": "A lawn mower is running.",
    "Vacuum Cleaner": "A vacuum cleaner is operating.",
    "Sewing Machine": "A sewing machine is stitching.",
    "Washing Machine": "A washing machine is running.",
    "Alarm": "An alarm is going off.",
    "Siren": "A siren is sounding.",
    "Horn Honk": "A horn is honking.",
    "Beep_Bleep": "A beeping sound is heard.",
    "Doorbell": "A doorbell is ringing.",
    "Rain": "Rain is falling",
    "Thunder": "Thunder is rumbling.",
    "Wind": "Wind is blowing.",
    "Fire": "A fire is crackling.",
    "Waves": "Waves are crashing.",
    "Stream_River": "Water is flowing."
}

audioset_to_category = {cn: k for k, v in category_to_audioset.items() for cn in v}

category_to_audioset_ = {
    "Bird Chirp": [
        "Bird vocalization, bird call, bird song",
        "Chirp, tweet",
        # "Bird",
        # "Chirp tone"
    ],
    "Dog Bark": [
        "Bark"#,
        # "Dog",
        # "Canidae, wild dogs, wolves",
        #"Whimper (dog)",
        #"Pant (dog)"
    ],
    "Cat Meow": [
        "Meow",
        # "Cat",
        # "Purr",
        # "Whimper"
    ],
    "Speech": [
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
        "Narration, monologue",
        "Conversation",
        "Human voice",
        "Human sounds",
        "Speech synthesizer",
        # "Rapping",
        "Whispering"#,
        # "Human group actions"
    ],
    "Sneeze": [
        "Sneeze"
    ],
    "Cough": [
        "Cough"
    ],
    "Doorbell": [
        "Doorbell"#,
        # "Telephone bell ringing"
    ],
    "Siren": [
        "Siren",
        "Ambulance (siren)",
        "Fire engine, fire truck (siren)",
        "Police car (siren)"#,
        # "Civil defense siren"
    ],
    "Thunder": [
        "Thunder",
        "Thunderstorm" # ,
        # "Boom",
        # "Explosion"
    ],
    "Motorcycle": [
        "Motorcycle"# ,
        # "Engine",
        # "Accelerating, revving, vroom"
    ],
    "Car": [
        "Car",
        "Motor vehicle (road)",
        "Vehicle",
        # "Engine",
        # "Car alarm",
        "Car passing by"
    ]
}

sound_captions_ = {
    "Bird Chirp": "Birds are chirping.",
    "Dog Bark": "A dog is barking.",
    "Cat Meow": "A cat is meowing.",
    "Speech": "Someone is speaking.",
    "Sneeze": "Someone is sneezing.",
    "Cough": "Someone is coughing.",
    "Doorbell": "A doorbell is ringing.",
    "Siren": "A siren is sounding.",
    "Thunder": "Thunder is rumbling.",
    "Motorcycle": "A motorcycle is passing.",
    "Car": "A car is driving by."
}




if __name__ == '__main__':

    tc = AudioSetStrong(reduce_classes=False)
    # tc.preload_audios()
    print(len(tc[0]))