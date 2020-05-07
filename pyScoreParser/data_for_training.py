import os
import random
import numpy as np
import pickle
from . import dataset_split
from .constants import VNET_COPY_DATA_KEYS, VNET_INPUT_KEYS, VNET_OUTPUT_KEYS
from pathlib import Path
from tqdm import tqdm
import shutil

class ScorePerformPairData:
    def __init__(self, piece, perform):
        self.piece_path = piece.xml_path
        self.perform_path = perform.midi_path
        self.graph_edges = piece.notes_graph
        self.features = {**piece.score_features, **perform.perform_features}
        self.split_type = None
        self.features['num_notes'] = piece.num_notes


class PairDataset:
    def __init__(self, dataset):
        self.dataset_path = dataset.path
        self.data_pairs = []
        self.feature_stats = None
        for piece in dataset.pieces:
            for performance in piece.performances:
                self.data_pairs.append(ScorePerformPairData(piece, performance))

    def get_squeezed_features(self):
        squeezed_values = dict()
        for pair in self.data_pairs:
            for feature_key in pair.features.keys():
                if type(pair.features[feature_key]) is dict:
                    if pair.features[feature_key]['need_normalize']:
                        if isinstance(pair.features[feature_key]['data'], list):
                            if feature_key not in squeezed_values.keys():
                                squeezed_values[feature_key] = []
                            squeezed_values[feature_key] += pair.features[feature_key]['data']
                        else:
                            if feature_key not in squeezed_values.keys():
                                squeezed_values[feature_key] = []
                            squeezed_values[feature_key].append(pair.features[feature_key]['data'])
        return squeezed_values

    def update_mean_stds_of_entire_dataset(self):
        squeezed_values = self.get_squeezed_features()
        self.feature_stats = cal_mean_stds(squeezed_values)

    def update_dataset_split_type(self, valid_set_list=dataset_split.VALID_LIST, test_set_list=dataset_split.TEST_LIST):
        # TODO: the split
        for pair in self.data_pairs:
            path = pair.piece_path
            for valid_name in valid_set_list:
                if valid_name in path:
                    pair.split_type = 'valid'
                    break
            else:
                for test_name in test_set_list:
                    if test_name in path:
                        pair.split_type = 'test'
                        break

            if pair.split_type is None:
                pair.split_type = 'train'

    def shuffle_data(self):
        random.shuffle(self.data_pairs)

    def save_features_for_virtuosoNet(self, save_folder):
        '''
        Convert features into format of VirtuosoNet training data
        :return: None (save file)
        '''
        def _flatten_path(file_path):
            return '_'.join(file_path.parts)

        save_folder = Path(save_folder)
        split_types = ['train', 'valid', 'test']

        save_folder.mkdir(exist_ok=True)
        for split in split_types:
            (save_folder / split).mkdir(exist_ok=True)

        training_data = []
        validation_data = []
        test_data = []

        for pair_data in tqdm(self.data_pairs):
            formatted_data = dict()
            try:
                formatted_data['input_data'], formatted_data['output_data'] = convert_feature_to_VirtuosoNet_format(pair_data.features, self.feature_stats)
                for key in VNET_COPY_DATA_KEYS:
                    formatted_data[key] = pair_data.features[key]
                formatted_data['graph'] = pair_data.graph_edges
                formatted_data['score_path'] = pair_data.piece_path
                formatted_data['perform_path'] = pair_data.perform_path

                save_name = _flatten_path(
                    Path(pair_data.perform_path).relative_to(Path(self.dataset_path))) + '.dat'
            
                with open(save_folder / pair_data.split_type / save_name, "wb") as f:
                    pickle.dump(formatted_data, f, protocol=2)
                
                if pair_data.split_type == 'test':
                    xml_name = Path(pair_data.piece_path).name
                    xml_path = Path(save_folder).joinpath(pair_data.split_type, xml_name)
                    shutil.copy(pair_data.piece_path, str(xml_path))

            except:
                print('Error: No Features with {}'.format(pair_data.perform_path))

  
        with open(save_folder / "stat.dat", "wb") as f:
            pickle.dump(self.feature_stats, f, protocol=2)
        

def get_feature_from_entire_dataset(dataset, target_score_features, target_perform_features):
    # e.g. feature_type = ['score', 'duration'] or ['perform', 'beat_tempo']
    output_values = dict()
    for feat_type in (target_score_features + target_perform_features):
        output_values[feat_type] = []
    for piece in dataset.pieces:
        for performance in piece.performances:
            for feat_type in target_score_features:
                # output_values[feat_type] += piece.score_features[feat_type]
                output_values[feat_type].append(piece.score_features[feat_type])
            for feat_type in target_perform_features:
                output_values[feat_type].append(performance.perform_features[feat_type])
    return output_values


def normalize_feature(data_values, target_feat_keys):
    for feat in target_feat_keys:
        concatenated_data = [note for perf in data_values[feat] for note in perf]
        mean = sum(concatenated_data) / len(concatenated_data)
        var = sum(pow(x-mean,2) for x in concatenated_data) / len(concatenated_data)
        # data_values[feat] = [(x-mean) / (var ** 0.5) for x in data_values[feat]]
        for i, perf in enumerate(data_values[feat]):
            data_values[feat][i] = [(x-mean) / (var ** 0.5) for x in perf]

    return data_values

# def combine_dict_to_array():

def cal_mean_stds_of_entire_dataset(dataset, target_features):
    '''
    :param dataset: DataSet class
    :param target_features: list of dictionary keys of features
    :return: dictionary of mean and stds
    '''
    output_values = dict()
    for feat_type in (target_features):
        output_values[feat_type] = []

    for piece in dataset.pieces:
        for performance in piece.performances:
            for feat_type in target_features:
                if feat_type in piece.score_features:
                    output_values[feat_type] += piece.score_features[feat_type]
                elif feat_type in performance.perform_features:
                    output_values[feat_type] += performance.perform_features[feat_type]
                else:
                    print('Selected feature {} is not in the data'.format(feat_type))

    stats = cal_mean_stds(output_values)

    return stats


def cal_mean_stds(feat_datas):
    stats = dict()
    for feature_key in feat_datas.keys():
        mean = sum(feat_datas[feature_key]) / len(feat_datas[feature_key])
        var = sum((x-mean)**2 for x in feat_datas[feature_key]) / len(feat_datas[feature_key])
        stds = var ** 0.5
        if stds == 0:
            stds = 1
        stats[feature_key] = {'mean': mean, 'stds': stds}
    return stats


def make_note_length_feature_list(feature_data, note_length):
    if not isinstance(feature_data, list) or len(feature_data) != note_length:
        feature_data = [feature_data] * note_length
    
    return feature_data

def normalize_feature_list(note_length_feature_data, mean, stds):
    return [(x - mean) / stds for x in note_length_feature_data]


def make_feature_data_for_VirtuosoNet(feature_data, stats, input_keys=VNET_INPUT_KEYS, output_keys=VNET_OUTPUT_KEYS):
    input_data = dict()
    output_data = dict()

    for key in input_keys:
        feature = make_note_length_feature_list(feature_data[key]['data'], feature_data['num_notes'])
        if key in stats:
            feature = normalize_feature_list(feature, stats[key]['mean'], stats[key]['stds'])
    
    for key in output_keys:
        pass


def convert_feature_to_VirtuosoNet_format(feature_data, stats, input_keys=VNET_INPUT_KEYS, output_keys=VNET_OUTPUT_KEYS):
    input_data = []
    output_data = []

    def check_if_global_and_normalize(key):
        value = feature_data[key]['data']
        if not isinstance(value, list) or len(value) != feature_data['num_notes']:  # global features like qpm_primo, tempo_primo, composer_vec
            value = [value] * feature_data['num_notes']
        if key in stats:  # if key needs normalization,
            value = [(x - stats[key]['mean']) / stats[key]['stds'] for x in value]
        return value

    def add_to_list(alist, item):
        if isinstance(item, list):
            alist += item
        else:
            alist.append(item)
        return alist

    def cal_dimension(data_with_all_features):
        total_length = 0
        for feat_data in data_with_all_features:
            if isinstance(feat_data[0], list):
                length = len(feat_data[0])
            else:
                length = 1
            total_length += length
        return total_length


    for key in input_keys:
        value = check_if_global_and_normalize(key)
        input_data.append(value)
    for key in output_keys:
        value = check_if_global_and_normalize(key)
        output_data.append(value)

    input_dimension = cal_dimension(input_data)
    output_dimension = cal_dimension(output_data)

    input_array = np.zeros((feature_data['num_notes'], input_dimension))
    output_array = np.zeros((feature_data['num_notes'], output_dimension))

    current_idx = 0

    for value in input_data:
        if isinstance(value[0], list):
            length = len(value[0])
            input_array[:, current_idx:current_idx + length] = value
        else:
            length = 1
            input_array[:,current_idx] = value
        current_idx += length
    current_idx = 0
    for value in output_data:
        if isinstance(value[0], list):
            length = len(value[0])
            output_array[:, current_idx:current_idx + length] = value
        else:
            length = 1
            output_array[:,current_idx] = value
        current_idx += length

    return input_array, output_array
