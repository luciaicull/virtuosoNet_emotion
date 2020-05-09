import argparse
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser("pyScoreParser")
    parser.add_argument("--emotion_path", type=Path,
                        default="/home/yoojin/data/emotionDataset/test/total/", help="emotion data folder name")
    parser.add_argument("--emotion_save_path", type=Path,
                        default="/home/yoojin/data/emotionDataset/test/save", help="emotion data save folder name")

    return parser