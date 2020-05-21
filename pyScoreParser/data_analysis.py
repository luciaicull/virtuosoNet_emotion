import _pickle as cPickle
import pickle
from tqdm import tqdm
from pathlib import Path
#from .constants import DEFAULT_SCORE_FEATURES, DEFAULT_PERFORM_FEATURES
#from .data_for_training import EmotionPairDataset

class DataPair:
    def __init__(self, pair_data):
        self.emotion = pair_data.emotion
        self.piece_name = pair_data.piece_path.split('/')[-1]
        self.perform_name = pair_data.perform_path.split('/')[-1]
        self.features = self._init_feature(pair_data)
    
    def _init_feature(self, pair_data):
        features = pair_data.features
        feature_dict = dict()
        feature_dict['beat_tempo'] = pair_data.performance_beat_tempos
        feature_dict['measure_tempo'] = pair_data.performance_measure_tempos
        feature_dict['velocity'] = features['velocity']['data']
        feature_dict['articulation'] = features['articulation']['data']
        feature_dict['deviation'] = features['onset_deviation']['data']
        feature_dict['pedals'] = {'start': features['pedal_at_start']['data'],
                                  'end': features['pedal_at_end']['data'],
                                  'cut': features['pedal_cut']['data'],
                                  'cut_time': features['pedal_cut_time']['data'],
                                  'refresh': features['pedal_refresh']['data'],
                                  'refresh_time': features['pedal_refresh_time']['data'],
                                  'soft_pedal': features['soft_pedal']['data']
                                  }
        return feature_dict
    
    def make_dict(self):
        data = dict()
        data['emotion'] = self.emotion
        data['piece_name'] = self.piece_name
        data['perform_name'] = self.perform_name
        data['feature_dict'] = self.features

        return data

def save_data_pair_dataset_by_piece():
    data_path = Path('/home/yoojin/data/emotionDataset/feature_for_analysis')
    with open(data_path.joinpath('pairdataset.dat'), 'rb') as f:
        u = cPickle.Unpickler(f)
        emotion_pair_data = u.load()

    e1_pairs = []
    e2_pairs = []
    e3_pairs = []
    e4_pairs = []
    e5_pairs = []
    data_pair_set_by_piece = []

    for pair_set in tqdm(emotion_pair_data.data_pair_set_by_piece):
        set_list = []
        for pair_data in pair_set:
            data = DataPair(pair_data).make_dict()
            if pair_data.emotion is 1:
                e1_pairs.append(data)
            elif pair_data.emotion is 2:
                e2_pairs.append(data)
            elif pair_data.emotion is 3:
                e3_pairs.append(data)
            elif pair_data.emotion is 4:
                e4_pairs.append(data)
            else:
                e5_pairs.append(data)
            set_list.append(data)
        data_pair_set_by_piece.append(set_list)
    with open(data_path.joinpath("data_pair_set_by_piece.dat"), "wb") as f:
        pickle.dump(data_pair_set_by_piece, f, protocol=2)

    with open(data_path.joinpath("e1_pairs.dat"), "wb") as f:
        pickle.dump(e1_pairs, f, protocol=2)
    with open(data_path.joinpath("e2_pairs.dat"), "wb") as f:
        pickle.dump(e2_pairs, f, protocol=2)
    with open(data_path.joinpath("e3_pairs.dat"), "wb") as f:
        pickle.dump(e3_pairs, f, protocol=2)
    with open(data_path.joinpath("e4_pairs.dat"), "wb") as f:
        pickle.dump(e4_pairs, f, protocol=2)
    with open(data_path.joinpath("e5_pairs.dat"), "wb") as f:
        pickle.dump(e5_pairs, f, protocol=2)


data_path = Path('/home/yoojin/data/emotionDataset/feature_for_analysis')

with open(data_path.joinpath('data_pair_set_by_piece.dat'), 'rb') as f:
    u = cPickle.Unpickler(f)
    data_pair_set_by_piece = u.load()

for dataset in data_pair_set_by_piece:
    for data in dataset:
        emotion = data['emotion']
        