'''
split data classes to an individual file
'''
import os
import pickle
import pandas
import math
from pathlib import Path
from abc import abstractmethod
from tqdm import tqdm
import _pickle as cPickle

from .musicxml_parser import MusicXMLDocument
from .midi_utils import midi_utils
from . import score_as_graph as score_graph, xml_midi_matching as matching
from . import xml_utils
from . import feature_extraction
from .alignment import Alignment

# total data class
class DataSet:
    def __init__(self, path, save=False, new_alignment=False):
        self.path = path

        self.pieces = []
        self.scores = []
        self.score_midis = []
        self.perform_midis = []
        self.composers = []

        self.performances = []
        
        self.performs_by_tag = {}
        self.flattened_performs_by_tag = []

        self.num_pieces = 0
        self.num_performances = 0
        self.num_score_notes = 0
        self.num_performance_notes = 0

        # self.default_perforddmances = self.performances
        # TGK: what for?

        # self._load_all_scores()
        scores, score_midis, perform_midis, composers = self.load_data()
        self.scores = scores
        self.score_midis = score_midis
        self.perform_midis = perform_midis
        self.composers = composers
        self.check_data_files(self.scores, self.score_midis, self.perform_midis, new_alignment)
        self.load_all_piece(self.scores, self.perform_midis,
                            self.score_midis, self.composers, save=save, new_alignment=new_alignment)

    @classmethod
    @abstractmethod
    def load_data(self):
        '''return scores, score_midis, performances, composers'''
        raise NotImplementedError
    
    def check_data_files(self, scores, score_midis, perform_midis, direct_matching=False):
        if direct_matching:
            print("start to check data files: _match.txt")
        else:
            print("start to check data files: score midi, _infer_corresp.txt")
        for n in tqdm(range(len(scores))):
            align_tool = Alignment(scores[n], score_midis[n], perform_midis[n], direct_matching)
            checked_perform_midis = align_tool.check_perf_align(direct_matching)
            perform_midis[n] = checked_perform_midis
        print("finished: all data files are checked")

    def load_all_piece(self, scores, perform_midis, score_midis, composers, save, new_alignment):
        print("start to load all pieces: make PieceData, PerformData")
        for n in tqdm(range(len(scores))):
            try:
                piece = PieceData(scores[n], perform_midis[n], score_midis[n], composers[n], save=save, new_alignment=new_alignment)
                self.pieces.append(piece)
                for perf in piece.performances:
                    self.performances.append(perf)
            except Exception as ex:
                # TODO: TGK: this is ambiguous. Can we specify which file 
                # (score? performance? matching?) and in which function the error occur?
                print(f'Error while processing {scores[n]}. Error type :{ex}')
        self.num_performances = len(self.performances)
        print("finished: all pieces are loaded")

    '''
    # TGK : same as extract_selected_features(DEFAULT_SCORE_FEATURES) ...
    def extract_all_features(self):
        score_extractor = feature_extraction.ScoreExtractor(DEFAULT_SCORE_FEATURES)
        perform_extractor = feature_extraction.PerformExtractor(DEFAULT_PERFORM_FEATURES)
        for piece in self.pieces:
            piece.score_features = score_extractor.extract_score_features(piece)
            for perform in piece.performances:
                perform.perform_features = perform_extractor.extract_perform_features(piece, perform)
    '''

    def _sort_performances(self):
        # TODO: move to EmotionDataset
        self.performances.sort(key=lambda x:x.midi_path)
        for tag in self.performs_by_tag:
            self.performs_by_tag[tag].sort(key=lambda x:x.midi_path)

        flattened_performs_by_tag = []
        for tag in self.performs_by_tag:
            for perform in self.performs_by_tag[tag]:
                flattened_performs_by_tag.append(perform)
                # self.perform_name_by_tag.append(perform.midi_path)
        self.flattened_performs_by_tag = flattened_performs_by_tag

    def save_dataset(self, filename='data_set.dat'):
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=2)

    def save_features_as_csv(self, features, feature_names, path='features.csv'):
        feature_type_name = ['MIDI Path'] + feature_names
        feature_with_name = [feature_type_name]
        perform_names = [x.midi_path for x in self.performances]
        for i,feature_by_perf in enumerate(features):
            feature_by_perf = [perform_names[i]] + feature_by_perf
            feature_with_name.append(feature_by_perf)
        dataframe = pandas.DataFrame(feature_with_name)
        dataframe.to_csv(path)

    def save_features_by_features_as_csv(self, feature_data, list_of_features, path='measure.csv'):
        for feature, feature_name in zip(feature_data, list_of_features):
            save_name = feature_name + '_' + path
            self.save_features_as_csv(feature, [feature_name], save_name)

    def features_to_list(self, list_of_feat):
        feature_data = [[] for i in range(len(list_of_feat))]
        for perf in self.performances:
            for i, feature_type in enumerate(list_of_feat):
                feature_data[i].append(perf.perform_features[feature_type]['data'])
        return feature_data

    def get_average_by_perform(self, feature_data):
        # axis 0: feature, axis 1: performance, axis 2: note
        average_data = [[] for i in range(len(feature_data[0]))]
        for feature in feature_data:
            for i, perf in enumerate(feature):
                valid_list = [x for x in perf if x is not None]
                avg = sum(valid_list) / len(valid_list)
                average_data[i].append(avg)

        return average_data

    # def list_features_to_measure(self, feature_data):
    #     # axis 0: performance, axis 1: feature, axis 2: note
    #     for data_by_perf in feature_data:

    # TODO : not refactored
    def get_average_feature_by_measure(self, list_of_features):
        measure_average_features = [[] for i in range(len(list_of_features))]
        for p_index, perf in enumerate(self.performances):
            features_in_performance = [ [] for i in range(len(list_of_features))]
            features_in_previous_measure = [ [] for i in range(len(list_of_features))]
            previous_measure = 0
            for i, pair in enumerate(perf.pairs):
                if pair == []:
                    continue
                if pair['xml'].measure_number != previous_measure:
                    previous_measure = pair['xml'].measure_number
                    if features_in_previous_measure[0] != []:
                        for j, data_of_selected_features in enumerate(features_in_previous_measure):
                            if len(data_of_selected_features) > 0:
                                average_of_selected_feature = sum(data_of_selected_features) / len(data_of_selected_features)
                            else:
                                average_of_selected_feature = 0
                            features_in_performance[j].append(average_of_selected_feature)
                        features_in_previous_measure = [[] for i in range(len(list_of_features))]
                for j, target_feature in enumerate(list_of_features):
                    feature_value = perf.perform_features[target_feature][i]
                    if feature_value is not None:
                        features_in_previous_measure[j].append(feature_value)
            for j, data_of_selected_features in enumerate(features_in_previous_measure):
                if len(data_of_selected_features) > 0:
                    average_of_selected_feature = sum(data_of_selected_features) / len(data_of_selected_features)
                else:
                    average_of_selected_feature = 0
                features_in_performance[j].append(average_of_selected_feature)
            for j, perform_data_of_feature in enumerate(measure_average_features):
                perform_data_of_feature.append(features_in_performance[j])
        return measure_average_features

    def _divide_by_tag(self, list_of_tag):
        # TODO: move to EmotionDataset
        # example of list_of_tag = ['professional', 'amateur']
        for tag in list_of_tag:
            self.performs_by_tag[tag] = []
        for piece in self.pieces:
            for perform in piece.performances:
                for tag in list_of_tag:
                    if tag in perform.midi_path:
                        self.performs_by_tag[tag].append(perform)
                        break

    def __str__(self):
        return str(self.__dict__)



# score data class
class PieceData:
    def __init__(self, xml_path, perform_lists, score_midi_path=None, composer=None, save=False, new_alignment=False):
        # meta information about piece
        self.xml_path = xml_path
        self.folder_path = os.path.dirname(xml_path)
        self.composer = composer
        self.pedal_elongate = False
        self.perform_lists = perform_lists
        self.score_midi_path = score_midi_path

        if score_midi_path == None:
            score_midi_path = os.path.dirname(xml_path) + '/' + Path(xml_path).stem + '_score.mid'
        #self.meta = PieceMeta(xml_path, perform_lists=perform_lists, score_midi_path=score_midi_path, composer=composer)
        self.performances = []
        
        #score_dat_path = os.path.dirname(xml_path) + '/score.dat'
        score_dat_path = Path(xml_path).with_suffix('.dat')

        if save:
            self.score = ScoreData(xml_path, score_midi_path, new_alignment)
            with open(score_dat_path , 'wb') as f:
                pickle.dump(self.score, f, protocol=2)
        else:
            if Path(score_dat_path).exists:
                with open(score_dat_path, 'rb') as f:
                    u = cPickle.Unpickler(f)
                    self.score = u.load()
            else:
                print(f'not exist {score_dat_path}. make one')
                self.score = ScoreData(xml_path, score_midi_path, new_alignment)
                with open(score_dat_path , 'wb') as f:
                    pickle.dump(self.score, f, protocol=2)

        # ScoreData alias
        self.xml_obj = self.score.xml_obj
        self.xml_notes = self.score.xml_notes
        self.num_notes = self.score.num_notes
        self.notes_graph = self.score.notes_graph
        self.score_midi_notes = self.score.score_midi_notes
        self.score_match_list = self.score.score_match_list
        self.score_pairs = self.score.score_pairs
        self.measure_positions = self.score.measure_positions
        self.beat_positions = self.score.beat_positions
        self.section_positions = self.score.section_positions
    
        # TODO: move to ScoreData
        self.score_features = {}

        # TODO: seperate meta from data_class
        #self.meta._check_perf_align()


        for perform in perform_lists:
            perform_dat_path = Path(perform[:-len('.mid')] + '.dat')
            if not save:
                if not perform_dat_path.exists:
                    print(f'not exist {perform_dat_path}.')
                with open(perform_dat_path, 'rb') as f:
                    u = cPickle.Unpickler(f)
                    perform_data = u.load()
                    self.performances.append(perform_data)
            else:
                try:
                    perform_data = PerformData(perform, new_alignment)
                except:
                    print(f'Cannot make performance data of {perform}')
                    self.performances.append(None)
                else:
                    if new_alignment:
                        try:
                            self._align_perform_with_score_directly(perform_data)
                        except:
                            print(f'No alignment result of {perform}')
                            self.performances.append(None)
                        else:
                            self.performances.append(perform_data)
                    else:
                        try: 
                            self._align_perform_with_score(perform_data)
                        except:
                            print(f'Cannot align {perform}')
                            self.performances.append(None)
                        else:
                            self.performances.append(perform_data)

            if save:
                if perform_data is not None: 
                    with open(perform_dat_path, 'wb') as f:
                        pickle.dump(perform_data, f, protocol=2)
    
    def extract_perform_features(self, target_features):
        perform_extractor = feature_extraction.PerformExtractor(target_features)
        for perform in self.performances:
            try:
                print('Performance:', perform.midi_path)
                perform.perform_features = perform_extractor.extract_perform_features(self, perform)
            except:
                print('Performance was not aligned: pass')

    def extract_score_features(self, target_features):
        score_extractor = feature_extraction.ScoreExtractor(target_features)
        self.score_features = score_extractor.extract_score_features(self)

    def _load_performances(self, new_alignment=False):
        for perf_midi_name in self.perform_lists:
            perform_data = PerformData(perf_midi_name, new_alignment)
            self._align_perform_with_score(perform_data)
            self.performances.append(perform_data)

    def _align_perform_with_score(self, perform):
        try:
            print('Performance path is ', perform.midi_path)
            perform.match_between_xml_perf = matching.match_score_pair2perform(
                self.score.score_pairs, perform.midi_notes, perform.corresp)
        except:
            print('matching.match_score_pair2perform error')
        else:
            try:
                perform.pairs = matching.make_xml_midi_pair(
                    self.score.xml_notes, perform.midi_notes, perform.match_between_xml_perf)
            except:
                print('matching.make_xml_midi_pair error')
            else:
                try:
                    perform.pairs, perform.valid_position_pairs = matching.make_available_xml_midi_positions(
                        perform.pairs)
                except:
                    print('matching.make_available_xml_midi_positions error')
                else:
                    perform._count_matched_notes()

    def _align_perform_with_score_directly(self, perform):
        print('Performance path: ', perform.midi_path)

        # get pair
        try:
            perform.pairs = matching.make_direct_xml_midi_pair(self.score.xml_notes, perform.midi_notes, perform.match_list, perform.missing_xml_list)
        except:
            print('matching.make_direct_xml_midi_pair error')
        else:
            try:
                perform.pairs, perform.valid_position_pairs = matching.make_available_xml_midi_positions(
                    perform.pairs)
            except:
                print('matching.make_available_xml_midi_positions error')
            else:
                perform._count_matched_notes()
        

    def __str__(self):
        text = 'Path name: {}, Composer Name: {}, Number of Performances: {}'.format(self.xml_path, self.composer, len(self.performances))
        return text


# performance data class
class PerformData:
    def __init__(self, midi_path, new_alignment=False):
        self.midi_path = midi_path
        self.midi = midi_utils.to_midi_zero(self.midi_path)
        self.midi = midi_utils.add_pedal_inf_to_notes(self.midi)
        self.midi_notes = self.midi.instruments[0].notes
        if new_alignment:
            self.match_path = os.path.splitext(self.midi_path)[
                0] + '_match.txt'
            self.match_list, self.missing_xml_list = matching.read_match_file(self.match_path)
        else:
            self.corresp_path = os.path.splitext(self.midi_path)[0] + '_infer_corresp.txt'
            self.corresp = matching.read_corresp(self.corresp_path)
        self.perform_features = {}
        self.match_between_xml_perf = None
        
        self.pairs = []
        self.valid_position_pairs = []

        self.num_matched_notes = 0
        self.num_unmatched_notes = 0
        self.tempos = []

        self.emotion = self._get_emotion(midi_path)
        self.performer = self._get_performer_number(midi_path)

    def __str__(self):
        return str(self.__dict__)

    def _count_matched_notes(self):
        self.num_matched_notes = 0
        self.num_unmatched_notes = 0
        for pair in self.pairs:
            if pair == []:
                self.num_unmatched_notes += 1
            else:
                self.num_matched_notes += 1
        print(
            'Number of Final Matched Notes: ' + str(self.num_matched_notes) + ', unmatched notes: ' + str(self.num_unmatched_notes))

    def _get_emotion(self, midi_path):
        midi_name = Path(midi_path).name
        emotion_list = ['.E1.', '.E2.', '.E3.', '.E4.', '.E5.']
        emotion = None
        for i, e in enumerate(emotion_list):
            if e in midi_name:
                emotion = i+1
                break
        
        return emotion

    def _get_performer_number(self, midi_path):
        midi_name = Path(midi_path).name
        performer_number = int(midi_name.split('.s0')[-1].split('.')[0])
        return performer_number

class ScoreData:
    def __init__(self, xml_path, score_midi_path, new_alignment=False):
        self.xml_obj = None
        self.xml_notes = None
        self.num_notes = 0

        # self.score_performance_match = []
        self.notes_graph = []
        self.score_midi_notes = []
        self.score_match_list = []
        self.score_pairs = []
        self.measure_positions = []
        self.beat_positions = []
        self.section_positions = []

        self._load_score_xml(xml_path)
        if new_alignment: # for score xml - perform midi direct alignment
            pass
        else: # original score xml - score midi alignment
            self._load_or_make_score_midi(score_midi_path)
            self._match_score_xml_to_midi()
        

    def __str__(self):
        return str(self.__dict__)

    def _load_score_xml(self, xml_path):
        self.xml_obj = MusicXMLDocument(xml_path)
        self._get_direction_encoded_notes()
        self.notes_graph = score_graph.make_edge(self.xml_notes)
        self.measure_positions = self.xml_obj.get_measure_positions()
        self.beat_positions = self.xml_obj.get_beat_positions()
        self.section_positions = xml_utils.find_tempo_change(self.xml_notes)

    def _get_direction_encoded_notes(self):
        notes, rests = self.xml_obj.get_notes()
        directions = self.xml_obj.get_directions()
        time_signatures = self.xml_obj.get_time_signatures()

        self.xml_notes = xml_utils.apply_directions_to_notes(notes, directions, time_signatures)
        self.num_notes = len(self.xml_notes)

    def _load_or_make_score_midi(self, score_midi_path):
        print(score_midi_path)
        if not os.path.isfile(score_midi_path):
            self.make_score_midi(score_midi_path)
        self.score_midi = midi_utils.to_midi_zero(score_midi_path)
        self.score_midi_notes = self.score_midi.instruments[0].notes
        self.score_midi_notes.sort(key=lambda x:x.start)

    def make_score_midi(self, midi_file_name):
        midi_notes, midi_pedals = xml_utils.xml_notes_to_midi(self.xml_notes)
        xml_utils.save_midi_notes_as_piano_midi(midi_notes, [], midi_file_name, bool_pedal=True)
        
    def _match_score_xml_to_midi(self):
        self.score_match_list = matching.match_xml_to_midi(self.xml_notes, self.score_midi_notes)
        self.score_pairs = matching.make_xml_midi_pair(self.xml_notes, self.score_midi_notes, self.score_match_list)

        self._count_matched_notes()

    def _count_matched_notes(self):
        self.num_matched_notes = 0
        self.num_unmatched_notes = 0
        for pair in self.score_pairs:
            if pair == []:
                self.num_unmatched_notes += 1
            else:
                self.num_matched_notes += 1
        print('Number of Score Matched Notes: ' + str(self.num_matched_notes) +
              ', Score unmatched notes: ' + str(self.num_unmatched_notes))



class YamahaDataset(DataSet):
    def __init__(self, path, save, new_alignment=False):
        super().__init__(path, save=save, new_alignment=new_alignment)

    def load_data(self):
        path = Path(self.path)
        #xml_list = sorted(path.glob('**/*.musicxml'))
        xml_list = sorted(path.glob('**/musicxml_cleaned.musicxml'))
        score_midis = [xml.parent / 'midi_cleaned.mid' for xml in xml_list]
        composers = [xml.relative_to(self.path).parts[0] for xml in xml_list]

        perform_lists = []
        for xml in xml_list:
            midis = sorted(xml.parent.glob('*.mid')) + sorted(xml.parent.glob('*.MID'))
            midis = [str(midi) for midi in midis if midi.name not in ['midi.mid', 'midi_cleaned.mid', 'midi_cleaned_error.mid']]
            midis = [midi for midi in midis if not 'XP' in midi]
            perform_lists.append(midis)

        # Path -> string wrapper
        xml_list = [str(xml) for xml in xml_list]
        score_midis = [str(midi) for midi in score_midis]
        return xml_list, score_midis, perform_lists, composers


class EmotionDataset(DataSet):
    def __init__(self, path, save=False, new_alignment=False):
        super().__init__(path, save=save, new_alignment=new_alignment)

    def load_data(self):
        path = Path(self.path)
        xml_list = sorted(path.glob('*.musicxml'))
        score_midis = [xml.parent / (xml.stem + '_midi_cleaned.mid') for xml in xml_list]
        composers = [xml.stem.split('.')[0] for xml in xml_list]

        perform_lists = []
        for xml in xml_list:
            midis = sorted(xml.parent.glob(f'{xml.stem}*.mid'))
            midis = [str(midi)
                     for midi in midis if 'midi_cleaned.mid' not in midi.name]
            perform_lists.append(midis)

        # Path -> string wrapper
        xml_list = [str(xml) for xml in xml_list]
        score_midis = [str(midi) for midi in score_midis]
        return xml_list, score_midis, perform_lists, composers


class StandardDataset(DataSet):
    def __init__(self, path, save=False, new_alignment=False):
        super().__init__(path, save=save, new_alignment=new_alignment)

    def load_data(self):
        path = Path(self.path)
        xml_list = sorted(path.glob('**/*.musicxml'))
        score_midis = [xml.parent / 'midi_cleaned.mid' for xml in xml_list]
        composers = [xml.relative_to(self.path).parts[0] for xml in xml_list]
        
        perform_lists = []
        for xml in xml_list:
            midis = sorted(xml.parent.glob('*.mid')) + sorted(xml.parent.glob('*.MID'))
            midis = [str(midi) for midi in midis if midi.name not in ['midi.mid', 'midi_cleaned.mid']]
            midis = [midi for midi in midis if not 'XP' in midi]
            perform_lists.append(midis)

        # Path -> string wrapper
        xml_list = [str(xml) for xml in xml_list]
        score_midis = [str(midi) for midi in score_midis]
        return xml_list, score_midis, perform_lists, composers

