import os
import random
import numpy as np
import pickle
from . import dataset_split
from .constants import VNET_INPUT_KEYS, VNET_OUTPUT_KEYS, PRIME_VNET_OUTPUT_KEYS
from pathlib import Path
from tqdm import tqdm
import shutil

class ScorePerformPairData:
    def __init__(self, piece, perform):
        self.piece_path = piece.xml_path
        self.perform_path = perform.midi_path
        self.piece_name = Path(piece.xml_path).name
        self.perform_name = Path(perform.midi_path).name
        self.graph_edges = piece.notes_graph
        self.features = {**piece.score_features, **perform.perform_features}
        self.split_type = None
        self.features['num_notes'] = piece.num_notes

        self.score_qpm_primo = piece.score_features['qpm_primo']
        self.performance_beat_tempos = perform.beat_tempos
        self.performance_measure_tempos = perform.measure_tempos


class PairDataset:
    def __init__(self, dataset):
        self.dataset_path = dataset.path
        self.data_pairs = []
        self.feature_stats = None
        self.index_dict = None

        self._initialize_data_pairs(dataset)
        
    def _initialize_data_pairs(self, dataset):
        for piece in dataset.pieces:
            for performance in piece.performances:
                self.data_pairs.append(ScorePerformPairData(piece, performance))


class ScorePerformPairData_Emotion(ScorePerformPairData):
    def __init__(self, piece, perform):
        super().__init__(piece, perform)
        self.emotion = perform.emotion
        self.performer = perform.performer


class EmotionPairDataset(PairDataset):
    def __init__(self, dataset):
        self.data_pair_set_by_piece = []
        super().__init__(dataset)
        
    
    def _initialize_data_pairs(self, dataset):
        for piece in dataset.pieces:
            tmp_set = []
            for performance in piece.performances:
                pair_data = ScorePerformPairData_Emotion(piece, performance)
                self.data_pairs.append(pair_data)
                tmp_set.append(pair_data)

            tmp_set = sorted(tmp_set, key=lambda pair:pair.perform_name)
            #assert tmp_set[0].emotion is 1
            #self.data_pair_set_by_piece.append(tmp_set)
            performer_set = set([pair.performer for pair in tmp_set])
            for performer_num in performer_set:
                performer_set = [pair for pair in tmp_set if pair.performer is performer_num]
                performer_set = sorted(performer_set, key=lambda pair:pair.emotion)
                self.data_pair_set_by_piece.append(performer_set)

            

# optimized to emotion dataset
# get PairDataset class and generate data in virtuosoNet format
class DataGenerator:
    def __init__(self, pair_dataset, save_path):
        self.pair_dataset = pair_dataset
        self.save_path = Path(save_path)


    def generate_statistics(self, valid_set_list=dataset_split.EMOTION_VALID_LIST, test_set_list=dataset_split.EMOTION_TEST_LIST):
        self._update_dataset_split_type(valid_set_list, test_set_list)
        self._update_mean_stds_of_entire_dataset()

    def _update_dataset_split_type(self, valid_set_list, test_set_list):
        for pair_data in self.pair_dataset.data_pairs:
            path = pair_data.piece_path
            for valid_name in valid_set_list:
                if valid_name in path:
                    pair_data.split_type = 'valid'
                    break
            else:
                for test_name in test_set_list:
                    if test_name in path:
                        pair_data.split_type = 'test'
                        pair_data.features['qpm_primo'] = pair_data.score_qpm_primo
                        break

            if pair_data.split_type is None:
                pair_data.split_type = 'train'

    def _update_mean_stds_of_entire_dataset(self):
        # get squeezed features
        feature_data = dict()
        for pair in self.pair_dataset.data_pairs:
            for feature_key in pair.features.keys():
                if type(pair.features[feature_key]) is dict:
                    if pair.features[feature_key]['need_normalize']:
                        if feature_key not in feature_data.keys():
                            feature_data[feature_key] = []
                        if isinstance(pair.features[feature_key]['data'], list):
                            feature_data[feature_key] += pair.features[feature_key]['data']
                        else:
                            feature_data[feature_key].append(
                                pair.features[feature_key]['data'])

        # cal mean and stds
        stats = dict()
        for feature_key in feature_data.keys():
            mean = sum(feature_data[feature_key]) / \
                len(feature_data[feature_key])
            var = sum(
                (x-mean)**2 for x in feature_data[feature_key]) / len(feature_data[feature_key])
            stds = var ** 0.5
            if stds == 0:
                stds = 1
            stats[feature_key] = {'mean': mean, 'stds': stds}

        self.pair_dataset.feature_stats = stats

    def save_final_feature_dataset(self, input_feature_keys=VNET_INPUT_KEYS, output_feature_keys=VNET_OUTPUT_KEYS, with_e1_qpm=False, e1_to_input_feature_keys=PRIME_VNET_OUTPUT_KEYS, output_for_classifier=False):
        self._generate_save_folders()

        for pair_data_list in tqdm(self.pair_dataset.data_pair_set_by_piece):
            feature_dict_list = []

            if e1_to_input_feature_keys:
                e1_data, _ = self._convert_feature(pair_data_list[0].features, self.pair_dataset.feature_stats, keys=e1_to_input_feature_keys)

            for pair_data in pair_data_list:
                feature_dict = dict()
                feature_dict['input_data'], input_feature_index_dict = self._convert_feature(pair_data.features, self.pair_dataset.feature_stats, keys=input_feature_keys)

                if output_for_classifier:
                    feature_dict['e1_perform_data'] = e1_data
                elif e1_to_input_feature_keys and not output_for_classifier:
                    feature_dict['input_data'], input_feature_index_dict = self._add_e1_output_feature_to_input_feature(
                                                                            feature_dict['input_data'], input_feature_index_dict, e1_data)
                
                if output_for_classifier: 
                    feature_dict['label'] = pair_data.emotion - 1
                feature_dict['output_data'], output_feature_index_dict = self._convert_feature(pair_data.features, self.pair_dataset.feature_stats, keys=output_feature_keys)
                feature_dict['note_location'] = pair_data.features['note_location']
                feature_dict['align_matched'] = pair_data.features['align_matched']
                feature_dict['articulation_loss_weight'] = pair_data.features['articulation_loss_weight']
                feature_dict['graph'] = pair_data.graph_edges
                feature_dict['score_path'] = pair_data.piece_path
                feature_dict['perform_path'] = pair_data.perform_path

                feature_dict_list.append(feature_dict)
                
                if self.pair_dataset.index_dict is None:
                    self.pair_dataset.index_dict = {'input_index_dict': input_feature_index_dict,
                                                    'output_index_dict': output_feature_index_dict}

                if with_e1_qpm and pair_data.emotion is 1:
                    qpm_index = self.pair_dataset.index_dict['input_index_dict']['qpm_primo']['index']
                    e1_qpm = feature_dict['input_data'][0][qpm_index]
            
            for feature_dict in feature_dict_list:
                if with_e1_qpm:
                    feature_dict['input_data'] = self._change_qpm_primo_to_e1_qpm_primo(
                        feature_dict['input_data'], self.pair_dataset.index_dict['input_index_dict'], e1_qpm)
                self._save_feature_dict(feature_dict, pair_data.split_type, self.pair_dataset.dataset_path)
        
        self._save_dataset_info()

    def _generate_save_folders(self):
        save_folder = Path(self.save_path)
        split_types = ['train', 'valid', 'test']

        save_folder.mkdir(exist_ok=True)
        for split in split_types:
            (save_folder / split).mkdir(exist_ok=True)

    def _check_if_global_and_normalize(self, value, key, note_num, stats):
        # global features like qpm_primo, tempo_primo, composer_vec
        if not isinstance(value, list) or len(value) != note_num:
            value = [value] * note_num
        if key in stats:  # if key needs normalization,
            value = [(x - stats[key]['mean']) / stats[key]['stds']
                     for x in value]
        return value

    def _convert_feature(self, feature_data, stats, keys):
        data = []
        index_dict = dict()
        total_feature_length = 0
        for key in keys:
            value = self._check_if_global_and_normalize(
                feature_data[key]['data'], key, feature_data['num_notes'], stats)
            data.append(value)
            
            # cal feature len
            if isinstance(value[0], list):
                feature_len = len(value[0])
            else:
                feature_len = 1
            
            if key in index_dict.keys(): # since 'beat_tempo' is doubled in output_keys
                index_dict[key]['index'] = [index_dict[key]['index'], total_feature_length]
            else:
                index_dict[key] = {'len': feature_len, 'index': total_feature_length}
            total_feature_length += feature_len
        
        index_dict['total_length'] = total_feature_length
        data_array = np.zeros((feature_data['num_notes'], total_feature_length))

        cur_idx = 0
        for value in data:
            if isinstance(value[0], list):
                length = len(value[0])
                data_array[:, cur_idx:cur_idx+length] = value
            else:
                length = 1
                data_array[:, cur_idx] = value
            cur_idx += length
        
        return data_array, index_dict


    def _add_e1_output_feature_to_input_feature(self, input_data, index_dict, e1_data):
        index_dict['e1_data'] = {'index': len(input_data[0]), 'len': len(e1_data[0])}
        input_data = np.append(input_data, e1_data, axis=1) # b/c shape is (note_num, feature_num)
        
        index_dict['total_length'] += len(e1_data[0])

        return input_data, index_dict

    def _change_qpm_primo_to_e1_qpm_primo(self, input_data, index_dict, e1_qpm):
        qpm_index = index_dict['qpm_primo']['index']
        input_data[:, qpm_index] = e1_qpm

        return input_data

    def _flatten_path(self, file_path):
        return '_'.join(file_path.parts)

    def _save_feature_dict(self, feature_dict, split_type, dataset_path):
        piece_path = feature_dict['score_path']
        perform_path = feature_dict['perform_path']
        data_name = self._flatten_path(Path(perform_path).relative_to(Path(dataset_path))) + '.dat'
        final_save_path = self.save_path.joinpath(split_type, data_name)
        
        with open(final_save_path, "wb") as f:
            pickle.dump(feature_dict, f, protocol=2)
        
        if split_type == 'test':
            xml_name = Path(piece_path).name
            xml_path = Path(self.save_path.joinpath(split_type, xml_name))
            shutil.copy(piece_path, str(xml_path))
    
    def _save_dataset_info(self):
        dataset_info = {'stats': self.pair_dataset.feature_stats,
                        'index_dict': self.pair_dataset.index_dict}
        with open(self.save_path.joinpath("dataset_info.dat"), "wb") as f:
            pickle.dump(dataset_info, f, protocol=2)

'''
class PairDataset:
    def __init__(self, dataset):
        self.dataset_path = dataset.path
        self.data_pairs = []
        self.feature_stats = None

        self._initialize_data_pairs(dataset)

    def _initialize_data_pairs(self, dataset):
        for piece in dataset.pieces:
            for performance in piece.performances:
                self.data_pairs.append(
                    ScorePerformPairData(piece, performance))

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
                            squeezed_values[feature_key].append(
                                pair.features[feature_key]['data'])
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
        
        #Convert features into format of VirtuosoNet training data
        #:return: None (save file)
        
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
                formatted_data['input_data'], formatted_data['output_data'] = convert_feature_to_VirtuosoNet_format(
                    pair_data.features, self.feature_stats)
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
                    xml_path = Path(save_folder).joinpath(
                        pair_data.split_type, xml_name)
                    shutil.copy(pair_data.piece_path, str(xml_path))

            except:
                print('Error: No Features with {}'.format(
                    pair_data.perform_path))

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
                output_values[feat_type].append(
                    piece.score_features[feat_type])
            for feat_type in target_perform_features:
                output_values[feat_type].append(
                    performance.perform_features[feat_type])
    return output_values


def normalize_feature(data_values, target_feat_keys):
    for feat in target_feat_keys:
        concatenated_data = [note for perf in data_values[feat]
                             for note in perf]
        mean = sum(concatenated_data) / len(concatenated_data)
        var = sum(pow(x-mean, 2)
                  for x in concatenated_data) / len(concatenated_data)
        for i, perf in enumerate(data_values[feat]):
            data_values[feat][i] = [(x-mean) / (var ** 0.5) for x in perf]

    return data_values


def cal_mean_stds_of_entire_dataset(dataset, target_features):
    
    #:param dataset: DataSet class
    #:param target_features: list of dictionary keys of features
    #:return: dictionary of mean and stds
    
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
        var = sum(
            (x-mean)**2 for x in feat_datas[feature_key]) / len(feat_datas[feature_key])
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
        feature = make_note_length_feature_list(
            feature_data[key]['data'], feature_data['num_notes'])
        if key in stats:
            feature = normalize_feature_list(
                feature, stats[key]['mean'], stats[key]['stds'])

    for key in output_keys:
        pass


def convert_feature_to_VirtuosoNet_format(feature_data, stats, input_keys=VNET_INPUT_KEYS, output_keys=VNET_OUTPUT_KEYS):
    input_data = []
    output_data = []

    def check_if_global_and_normalize(key):
        value = feature_data[key]['data']
        # global features like qpm_primo, tempo_primo, composer_vec
        if not isinstance(value, list) or len(value) != feature_data['num_notes']:
            value = [value] * feature_data['num_notes']
        if key in stats:  # if key needs normalization,
            value = [(x - stats[key]['mean']) / stats[key]['stds']
                     for x in value]
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
            input_array[:, current_idx] = value
        current_idx += length
    current_idx = 0
    for value in output_data:
        if isinstance(value[0], list):
            length = len(value[0])
            output_array[:, current_idx:current_idx + length] = value
        else:
            length = 1
            output_array[:, current_idx] = value
        current_idx += length

    return input_array, output_array
'''
