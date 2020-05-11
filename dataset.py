import numpy as np
import torch as th
import pickle
import _pickle as cPickle
from abc import abstractmethod
from collections import defaultdict, namedtuple
from tqdm import tqdm
from bisect import bisect_left, bisect_right
from pathlib import Path

from pyScoreParser import xml_matching
from pyScoreParser.data_for_training import *


ChunkInfo = namedtuple('ChunkInfo', ['file_index', 'local_index', 'midi_index'])


class PerformDataset():
    def __init__(self, path, split, graph=False, samples=None, stride=None):
        self.path = path
        self.split = split
        self.graph = graph
        self.samples = samples
        if stride is None:
            stride = samples
        self.stride = stride

        # iterate over whole dataset to get length
        self.files = self.feature_files()
        self.num_chunks = 0
        self.entry = [] 

        for file_idx in tqdm(range(len(self.files))):
            if samples is None:
                self.entry.append(ChunkInfo(file_idx, 0, 0))
                continue
            feature_file = self.files[file_idx]
            feature_lists = self.load_file(feature_file)
            length = feature_lists['input_data'].shape[0]

            if length < samples:
                continue
            num_segs = (length - samples) // stride + 1
            for seg_idx in range(num_segs):
                self.entry.append(ChunkInfo(file_idx, seg_idx, seg_idx*stride))


    def __getitem__(self, index):
        file_idx, seg_idx, note_offset = self.entry[index]
        feature = self.load_file(self.files[file_idx])
        if self.samples is None:
            return feature
        else:
            feature['input_data'] = \
                feature['input_data'][note_offset:note_offset+self.samples, :]
            feature['output_data'] = feature['output_data'][note_offset:note_offset+self.samples, :]
            feature['note_location'] = feature['note_location'][note_offset:note_offset+self.samples]
            feature['align_matched'] = feature['align_matched'][note_offset:note_offset+self.samples]
            feature['articulation_loss_weight'] = feature['articulation_loss_weight'][note_offset:note_offset+self.samples]
            notes_in_graph = [el[0] for el in feature['graph']]
            idx_left = bisect_left(notes_in_graph, note_offset)
            idx_right = bisect_right(notes_in_graph, note_offset + self.samples)
            feature['graph'] = feature['graph'][idx_left: idx_right]
            return feature
        
    def feature_files(self):
        ''' return feature .dat file paths'''
        return sorted(Path(self.path).glob(f'{self.split}/*.dat'))

    def __len__(self):
        return len(self.entry)

    @staticmethod
    def load_file(data_path):
        with open(data_path, "rb") as f:
            u = cPickle.Unpickler(f)
            features = u.load()
        return features