from pathlib import Path
import pickle
from .model_parameters import ModelConfig
import torch
from . import model_parameters as param
import torch
import torch.nn as nn

DROP_OUT = 0.1

class TrillRNN(nn.Module):
    def __init__(self, model_config, device):
        super(TrillRNN, self).__init__()
        self.device = device
        self.config = model_config


        #self.hidden_size = network_parameters.note.size
        #self.num_layers = network_parameters.note.layers
        #self.input_size = network_parameters.input_size
        #self.output_size = network_parameters.output_size
        #self.device = device
        #self.is_graph = False
        #self.loss_type = 'MSE'

        # self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=DROP_OUT)
        # self.fc = nn.Linear(hidden_size * 2, num_output)  # 2 for

        self.note_fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.note_lstm = nn.LSTM(self.hidden_size, self.hidden_size,
                                 num_layers=self.num_layers, bidirectional=True, batch_first=True)

        self.out_fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(self.hidden_size, self.output_size),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, edges, note_locations, start_index, initial_z=0):
        note_contracted = self.note_fc(x)
        hidden_out, _ = self.note_lstm(note_contracted)

        out = self.out_fc(hidden_out)

        
        up_trill = self.sigmoid(out[:, :, -1])
        out[:, :, -1] = up_trill
        
'''
model_config = ModelConfig()
model_config.input_size = 78
model_config.output_size = 11
model_config.graph_keys.append('slur')
model_config.graph_keys.append('voice')
model_config.num_edge_types = len(model_config.graph_keys) * 2

model_config.input_size = 78 + 11
model_config.output_size = 5
model_config.note.layers = 2
model_config.note.size = 32
model_config.is_trill = True

param.save_parameters(model_config, Path('/home/yoojin/repositories/virtuosoNet/virtuoso/parameters/parameters_for_guide/'),
                      'trill_default_converted_param')
'''
'''
trill_model_path = '/home/yoojin/repositories/virtuosoNet/virtuoso/checkpoints/checkpoint_for_guide/trill_default_best.pth.tar'
if torch.cuda.is_available():
        def map_location(storage, loc): return storage.cuda()
checkpoint = torch.load(trill_model_path, map_location=map_location)

print('checkpoint loaded')

path = Path('/home/yoojin/repositories/virtuosoNet/virtuoso/parameters/parameters_for_guide/trill_default_param.dat')
with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        net_params = u.load()

a = model_parameters.ModelConfig
'''
