import argparse
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser("pyScoreParser")
    '''
    parser.add_argument("--emotion_path", type=Path,
                        default="/home/yoojin/data/emotionDataset/test/total/", help="emotion data folder name")
    parser.add_argument("--emotion_save_path", type=Path,
                        default="/home/yoojin/data/emotionDataset/test/save", help="emotion data save folder name")
    '''
    parser.add_argument("--path", type=Path,
                        default="/home/yoojin/data/emotionDataset/test/", help="emotion data folder name")
    parser.add_argument("--with_emotion", type=bool,
                        default=False)
    parser.add_argument("--with_e1_qpm", type=bool,
                        default=False)
    parser.add_argument("--e1_to_input_feature_keys", type=bool,
                        default=False)
    parser.add_argument("--output_for_classifier", type=bool,
                        default=False)
    return parser
