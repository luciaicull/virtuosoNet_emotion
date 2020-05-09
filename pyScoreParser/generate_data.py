from .data_class import YamahaDataset, EmotionDataset, DataSet, DEFAULT_SCORE_FEATURES, DEFAULT_PERFORM_FEATURES
from .data_for_training import PairDataset, EmotionPairDataset
from . import dataset_split
import pickle
import _pickle as cPickle
import csv
from .parser import get_parser

parser = get_parser()
args = parser.parse_args()
emotion_path = args.emotion_path
emotion_save_path = args.emotion_save_path

# make dataset
emotion_dataset = EmotionDataset(
    emotion_path, emotion_save_path, new_alignment=True)

# save dataset
with open(emotion_save_path.joinpath("total_dataset.dat"), "wb") as f:
    pickle.dump(emotion_dataset, f, protocol=2)

'''
print('Start: load dataset')
with open(emotion_save_path + '/total_dataset.dat', 'rb') as f:
    u = cPickle.Unpickler(f)
    emotion_dataset = u.load()
print('Finished: load dataset')
'''
'''
print('Start: save note matched result')
f = open(emotion_save_path + '/final_match_result.csv', 'w', encoding='utf-8')
wr = csv.writer(f)
wr.writerow(['performance.midi_path',
             'num_matched_notes', 'num_unmatched_notes'])
for piece in emotion_dataset.pieces:
    for performance in piece.performances:
        wr.writerow([performance.midi_path.split('/')[-1], str(
            performance.num_matched_notes), str(performance.num_unmatched_notes)])
f.close()
print('Finished: save note matched result')
'''

# extract features
for piece in emotion_dataset.pieces:
    piece.extract_perform_features(DEFAULT_PERFORM_FEATURES)
    piece.extract_score_features(DEFAULT_SCORE_FEATURES)

# make PairDatset
emotion_pair_data = EmotionPairDataset(emotion_dataset)

# save PairDataset for entire dataset for feature tracking
# you can check feature dictionary in each ScorePerformPairData() object in PairDataset.data_pairs
# reference : data_for_training.py or use test_check_features.py
# features in ScorePerformData.features() are shape of dictionary
#       : {feature_key1:(len(notes)), feature_key2:(len(notes)), ...}
'''
with open(emotion_save_path + "/pairdataset.dat", "wb") as f:
    pickle.dump(emotion_pair_data, f, protocol=2)
'''
 
# statistics
emotion_pair_data.update_dataset_split_type(
    valid_set_list=dataset_split.EMOTION_VALID_LIST, test_set_list=dataset_split.EMOTION_TEST_LIST)
emotion_pair_data.update_mean_stds_of_entire_dataset()


# save dataset
# features will saved at {emotion_save_path}.{train OR test OR validation}.{perform_midi_name}.dat
# features are in shape of list
#        : (len(notes), len(features))
emotion_pair_data.save_features_for_virtuosoNet(emotion_save_path)
