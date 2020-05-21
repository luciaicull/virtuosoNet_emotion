import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedGraph(nn.Module):
    def __init__(self, size, num_edge_style, device=0, secondary_size=0):
        super(GatedGraph, self).__init__()
        if secondary_size == 0:
            secondary_size = size
        self.size = size
        self.secondary_size = secondary_size

        self.ba = torch.nn.Parameter(torch.Tensor(size))
        self.wz = torch.nn.Parameter(torch.Tensor(
            num_edge_style, size, secondary_size))
        self.wr = torch.nn.Parameter(torch.Tensor(
            num_edge_style, size, secondary_size))
        self.wh = torch.nn.Parameter(torch.Tensor(
            num_edge_style, size, secondary_size))
        self.uz = torch.nn.Parameter(torch.Tensor(size, secondary_size))
        self.ur = torch.nn.Parameter(torch.Tensor(size, secondary_size))
        self.uh = torch.nn.Parameter(
            torch.Tensor(secondary_size, secondary_size))

        nn.init.xavier_normal_(self.wz)
        nn.init.xavier_normal_(self.wr)
        nn.init.xavier_normal_(self.wh)
        nn.init.xavier_normal_(self.uz)
        nn.init.xavier_normal_(self.ur)
        nn.init.xavier_normal_(self.uh)
        nn.init.zeros_(self.ba)

        self.sigmoid = torch.nn.Sigmoid()

        self.tanh = torch.nn.Tanh()

    def forward(self, input, edge_matrix, iteration=10):
        for i in range(iteration):
            activation = torch.matmul(
                edge_matrix.transpose(1, 2), input) + self.ba
            temp_z = self.sigmoid(torch.bmm(activation, self.wz).sum(
                0) + torch.matmul(input, self.uz))
            temp_r = self.sigmoid(torch.bmm(activation, self.wr).sum(
                0) + torch.matmul(input, self.ur))

            if self.secondary_size == self.size:
                temp_hidden = self.tanh(
                    torch.bmm(activation, self.wh).sum(0) + torch.matmul(temp_r * input, self.uh))
                input = (1 - temp_z) * input + temp_r * temp_hidden
            else:
                temp_hidden = self.tanh(
                    torch.bmm(activation, self.wh).sum(0) + torch.matmul(temp_r * input[:, :, -self.secondary_size:], self.uh))
                temp_result = (
                    1 - temp_z) * input[:, :, -self.secondary_size:] + temp_r * temp_hidden
                input = torch.cat(
                    (input[:, :, :-self.secondary_size], temp_result), 2)

        return input


class ContextAttention(nn.Module):
    def __init__(self, size, num_head):
        super(ContextAttention, self).__init__()
        self.attention_net = nn.Linear(size, size)
        self.num_head = num_head

        if size % num_head != 0:
            raise ValueError("size must be dividable by num_head", size, num_head)

        self.head_size = int(size/num_head)
        self.context_vector = torch.nn.Parameter(
            torch.Tensor(num_head, self.head_size, 1))
        nn.init.uniform_(self.context_vector, a=-1, b=1)

    def forward(self, x):
        attention = self.attention_net(x)
        attention_tanh = F.tanh(attention)
        if self.head_size != 1:
            attention_split = torch.cat(attention_tanh.split(
                split_size=self.head_size, dim=2), dim=0)
            similarity = torch.bmm(attention_split, self.context_vector)
            softmax_weight = F.softmax(similarity, dim=1)
            x_split = torch.cat(
                x.split(split_size=self.head_size, dim=2), dim=0)

            weighted_mul = torch.bmm(softmax_weight.transpose(1, 2), x_split)

            restore_size = int(weighted_mul.size(0) / self.num_head)
            attention = torch.cat(weighted_mul.split(
                split_size=restore_size, dim=0), dim=2)
        else:
            softmax_weight = F.softmax(attention, dim=1)
            attention = softmax_weight * x

        sum_attention = torch.sum(attention, dim=1)
        return sum_attention

class Han_Note_AR(nn.Module):
    def __init__(self, model_config, device, cons, step_by_step=False):
        self.device = device
        self.step_by_step = step_by_step
        self.config = model_config
        self.constants = cons

        self.lstm = nn.LSTM(self.config.note.size, self.config.note.size, self.config.note.layers,
                            batch_first=True, bidirectional=True, dropout=cons.DROP_OUT)

        self.voice_net = nn.LSTM(self.config.note.size, self.config.voice.size, self.config.voice.layers,
                                 batch_first=True, bidirectional=True, dropout=cons.DROP_OUT)
        self.beat_attention = ContextAttention((self.config.note.size + self.config.voice.size) * 2,
                                               self.config.num_attention_head)
        self.beat_rnn = nn.LSTM((self.config.note.size + self.config.voice.size) * 2, self.config.beat.size,
                                self.config.beat.layers, batch_first=True, bidirectional=True, dropout=cons.DROP_OUT)

        self.measure_attention = ContextAttention(
            self.config.beat.size*2, self.config.num_attention_head)
        self.measure_rnn = nn.LSTM(self.config.beat.size * 2, self.config.measure.size, self.config.measure.layers,
                                   batch_first=True, bidirectional=True)
        self.perform_style_to_measure = nn.LSTM(self.config.measure.size * 2 + self.config.encoder.size,
                                                self.config.encoder.size, num_layers=1, bidirectional=False)

        self.beat_tempo_forward = nn.LSTM(
            (self.config.beat.size + self.config.measure.size) * 2 +
            5 + 3 + self.config.output_size + self.config.encoder.size,
            self.config.beat.size, num_layers=1, batch_first=True, bidirectional=False)
        self.result_for_tempo_attention = ContextAttention(
            self.config.output_size - 1, 1)

        self.beat_tempo_fc = nn.Linear(self.config.beat.size, 1)

        self.note_fc = nn.Sequential(
            nn.Linear(self.config.input_size, self.config.note.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.config.note.size, self.config.note.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.config.note.size, self.config.note.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU(),
        )

        self.output_lstm = nn.LSTM(self.config.final.input, self.config.final.size, num_layers=self.config.final.layers,
                                   batch_first=True, bidirectional=False)

        self.fc = nn.Linear(self.config.final.size,
                            self.config.output_size - 1)

        self.performance_note_encoder = nn.LSTM(
            self.config.encoder.size, self.config.encoder.size, bidirectional=True)
        if self.config.encoder.size % self.config.num_attention_head == 0:
            self.performance_measure_attention = ContextAttention(
                self.config.encoder.size * 2, self.config.num_attention_head)
        else:
            self.performance_measure_attention = ContextAttention(
                self.config.encoder.size * 2, self.config.encoder.size * 2)
        self.performance_embedding_layer = nn.Sequential(
            nn.Linear(self.config.output_size, self.config.note.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.config.note.size, self.config.note.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU()
        )
        self.performance_contractor = nn.Sequential(
            nn.Linear(self.config.encoder.input, self.config.encoder.size),
            nn.Dropout(cons.DROP_OUT),
            # nn.BatchNorm1d(self.encoder_size),
            nn.ReLU()
        )
        self.performance_encoder = nn.LSTM(self.config.encoder.size * 2, self.config.encoder.size,
                                           num_layers=self.config.encoder.layers, batch_first=True, bidirectional=True)
        self.performance_final_attention = ContextAttention(
            self.config.encoder.size * 2, self.config.num_attention_head)
        self.performance_encoder_mean = nn.Linear(
            self.config.encoder.size * 2, self.config.encoded_vector_size)
        self.performance_encoder_var = nn.Linear(
            self.config.encoder.size * 2, self.config.encoded_vector_size)

        self.style_vector_expandor = nn.Sequential(
            nn.Linear(self.config.encoded_vector_size,
                      self.config.encoder.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU()
        )

        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, edges, note_locations, start_index, initial_z=False, rand_threshold=0.2, return_z=False):
        # score encoder part
        beat_numbers = [x.beat for x in note_locations]
        measure_numbers = [x.measure for x in note_locations]
        voice_numbers = [x.voice for x in note_locations]

        num_notes = x.size(1)

        note_out, beat_hidden_out, measure_hidden_out = \
            self._run_offline_score_model(
                x, edges, beat_numbers, measure_numbers, voice_numbers, start_index)
        beat_out_spanned = self._span_beat_to_note_num(
            beat_hidden_out, beat_numbers, num_notes, start_index)
        measure_out_spanned = self._span_beat_to_note_num(measure_hidden_out, measure_numbers, num_notes,
                                                         start_index)

        # performance encoder part
        if type(initial_z) is not bool:  # test session
            if type(initial_z) is str and initial_z == 'zero':
                zero_mean = torch.zeros(self.config.encoded_vector_size)
                one_std = torch.zeros(self.config.encoded_vector_size)
                perform_z = self._reparameterize(
                    zero_mean, one_std).to(self.device)
            elif not initial_z.is_cuda:
                perform_z = torch.Tensor(initial_z).to(self.device).view(1, -1)
            else:
                perform_z = initial_z.view(1, -1)
            perform_mu = 0
            perform_var = 0
        else:  # if initial_z = False (default), in case of training and validation
            expanded_y = self.performance_embedding_layer(y)

            perform_concat = torch.cat(
                (note_out, beat_out_spanned, measure_out_spanned, expanded_y), 2)
        
        perform_concat = self._masking_half(perform_concat)
        perform_contracted = self.performance_contractor(perform_concat)
        perform_note_encoded, _ = self.performance_note_encoder(
            perform_contracted)

        perform_measure = self._make_higher_node(perform_note_encoded, self.performance_measure_attention,
                                                beat_numbers, measure_numbers, start_index, lower_is_note=True)
        perform_style_encoded, _ = self.performance_encoder(
            perform_measure)
        perform_style_vector = self.performance_final_attention(
            perform_style_encoded)
        perform_z, perform_mu, perform_var = \
            self._encode_with_net(
                perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)

        if return_z:
            total_perform_z = [perform_z]
            for i in range(10):
                temp_z = self._reparameterize(perform_mu, perform_var)
                total_perform_z.append(temp_z)

            return total_perform_z

        # performance decoder
        perform_z = self.style_vector_expandor(perform_z)
        perform_z_batched = perform_z.repeat(
            x.shape[1], 1).view(1, x.shape[1], -1)
        perform_z = perform_z.view(-1)

        tempo_hidden = self._init_hidden(1, 1, x.size(0), self.config.beat.size)
        num_beats = beat_hidden_out.size(1)
        #TODO: why -1?
        result_nodes = torch.zeros(
            num_beats, self.config.output_size - 1).to(self.device)

        num_measures = measure_numbers[start_index +
                                       num_notes - 1] - measure_numbers[start_index] + 1
        perform_z_measure_spanned = perform_z.repeat(
            num_measures, 1).view(1, num_measures, -1)
        perform_z_measure_cat = torch.cat(
            (perform_z_measure_spanned, measure_hidden_out), 2)
        measure_perform_style, _ = self.perform_style_to_measure(
            perform_z_measure_cat)
        measure_perform_style_spanned = self._span_beat_to_note_num(measure_perform_style, measure_numbers,
                                                                   num_notes, start_index)

        final_hidden = self._init_hidden(
            1, 1, x.size(0), self.config.final.size)

        qpm_primo = x[:, 0, self.constants.QPM_PRIMO_IDX]
        tempo_primo = x[0, 0,
                        self.constants.TEMPO_PRIMO_IDX:self.constants.TEMPO_PRIMO_IDX+self.constants.TEMPO_PRIMO_LEN]

        prev_out = torch.zeros(self.config.output_size).to(self.device)
        prev_tempo = prev_out[self.constants.OUTPUT_TEMPO_INDEX:
                              self.constants.OUTPUT_TEMPO_INDEX+1]
        prev_beat = -1
        prev_beat_end = 0
        out_total = torch.zeros(
            num_notes, self.config.output_size).to(self.device)
        prev_out_list = []
        has_ground_truth = y.size(1) > 1

        for i in range(num_notes):
            current_beat = beat_numbers[start_index +
                                                    i] - beat_numbers[start_index]
            current_measure = measure_numbers[start_index +
                                                          i] - measure_numbers[start_index]
            if current_beat > prev_beat:  # beat changed
                if i - prev_beat_end > 0:  # if there are outputs to consider
                    corresp_result = torch.stack(
                                    prev_out_list).unsqueeze_(0)
                else:  # there is no previous output
                    corresp_result = torch.zeros(
                                    (1, 1, self.config.output_size-1)).to(self.device)
                result_node = self.result_for_tempo_attention(
                                corresp_result)
                prev_out_list = []
                result_nodes[current_beat, :] = result_node

                tempos = torch.zeros(1, num_beats, 1).to(self.device)

                beat_tempo_vec = x[0, i, self.constants.TEMPO_IDX:
                                   self.constants.TEMPO_IDX + self.constants.TEMPO_PARAM_LEN]
                beat_tempo_cat = torch.cat((beat_hidden_out[0, current_beat, :], measure_hidden_out[0, current_measure, :], prev_tempo,
                                            qpm_primo, tempo_primo, beat_tempo_vec,
                                            result_nodes[current_beat, :], perform_z)).view(1, 1, -1)

                beat_forward, tempo_hidden = self.beat_tempo_forward(
                    beat_tempo_cat, tempo_hidden)
                tmp_tempos = self.beat_tempo_fc(beat_forward)
                prev_beat_end = i
                prev_tempo = tmp_tempos.view(1)
                prev_beat = current_beat
            
            tmp_voice = voice_numbers[start_index + i] - 1

            out_combined = torch.cat(
                (note_out[0, i, :], beat_hidden_out[0, current_beat, :],
                 measure_hidden_out[0, current_measure, :],
                 prev_out, qpm_primo, tempo_primo, perform_z)).view(1, 1, -1)
            
            out, final_hidden = self.output_lstm(out_combined, final_hidden)
            out = out.view(-1)
            out = self.fc(out)

            prev_out_list.append(out)
            out = torch.cat((prev_tempo, out))

            prev_out = out
            out_total[i, :] = out

        out_total = out_total.view(1, num_notes, -1)
        hidden_total = torch.cat(
            (note_out, beat_out_spanned, measure_out_spanned), 2)
        
        return out_total, perform_mu, perform_var, hidden_total

    def _run_offline_score_model(self, x, edges, beat_numbers, measure_numbers, voice_numbers, start_index):
        hidden = self._init_hidden(
            self.config.note.layers, 2, x.size(0), self.config.note.size)
        beat_hidden = self._init_hidden(
            self.config.beat.layers, 2, x.size(0), self.config.beat.size)
        measure_hidden = self._init_hidden(
            self.config.measure.layers, 2, x.size(0), self.config.measure.size)

        x = self.note_fc(x)

        temp_voice_numbers = voice_numbers[start_index:start_index + x.size(1)]
        if temp_voice_numbers == []:
            temp_voice_numbers = voice_numbers[start_index:]
        max_voice = max(temp_voice_numbers)
        voice_hidden = self._init_voice_layer(1, max_voice)
        voice_out, voice_hidden = self._run_voice_net(
            x, voice_hidden, temp_voice_numbers, max_voice) # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        hidden_out, hidden = self.lstm(x, hidden)
        hidden_out = torch.cat((hidden_out, voice_out), 2)

        beat_nodes = self._make_higher_node(
            hidden_out, self.beat_attention, beat_numbers, beat_numbers, start_index, lower_is_note=True)
        beat_hidden_out, beat_hidden = self.beat_rnn(beat_nodes, beat_hidden)
        measure_nodes = self._make_higher_node(
            beat_hidden_out, self.measure_attention, beat_numbers, measure_numbers, start_index)
        measure_hidden_out, measure_hidden = self.measure_rnn(
            measure_nodes, measure_hidden)

        return hidden_out, beat_hidden_out, measure_hidden_out

    def _init_voice_layer(self, batch_size, max_voice):
        layers = []
        for i in range(max_voice):
            # h0 = torch.zeros(self.num_voice_layers * 2, batch_size, self.voice_hidden_size).to(device)
            h0 = torch.zeros(self.config.voice.layers * 2,
                             batch_size, self.config.voice.size).to(self.device)
            layers.append((h0, h0))
        return layers

    def _run_voice_net(self, batch_x, voice_hidden, voice_numbers, max_voice):
        num_notes = batch_x.size(1)
        output = torch.zeros(1, batch_x.size(
            1), self.config.voice.size * 2).to(self.device)
        voice_numbers = torch.Tensor(voice_numbers)
        for i in range(1, max_voice+1):
            voice_x_bool = voice_numbers == i
            num_voice_notes = torch.sum(voice_x_bool)
            if num_voice_notes > 0:
                span_mat = torch.zeros(num_notes, num_voice_notes)
                note_index_in_voice = 0
                for j in range(num_notes):
                    if voice_x_bool[j] == 1:
                        span_mat[j, note_index_in_voice] = 1
                        note_index_in_voice += 1
                span_mat = span_mat.view(1, num_notes, -1).to(self.device)
                voice_x = batch_x[0, voice_x_bool, :].view(
                    1, -1, self.config.note.size)
                ith_hidden = voice_hidden[i-1]

                ith_voice_out, ith_hidden = self.voice_net(voice_x, ith_hidden)
                # ith_voice_out, ith_hidden = self.lstm(voice_x, ith_hidden)
                output += torch.bmm(span_mat, ith_voice_out)
        return output, voice_hidden
    
    def _make_higher_node(self, lower_out, attention_weights, lower_indexes, higher_indexes, start_index, lower_is_note=False):
        higher_nodes = []
        prev_higher_index = higher_indexes[start_index]
        lower_node_start = 0
        lower_node_end = 0
        num_lower_nodes = lower_out.shape[1]
        start_lower_index = lower_indexes[start_index]
        lower_hidden_size = lower_out.shape[2]
        for low_index in range(num_lower_nodes):
            absolute_low_index = start_lower_index + low_index
            if lower_is_note:
                current_note_index = start_index + low_index
            else:
                current_note_index = lower_indexes.index(absolute_low_index)

            if higher_indexes[current_note_index] > prev_higher_index:
                # new beat start
                lower_node_end = low_index
                corresp_lower_out = lower_out[:,
                                              lower_node_start:lower_node_end, :]
                higher = attention_weights(corresp_lower_out)
                higher_nodes.append(higher)

                lower_node_start = low_index
                prev_higher_index = higher_indexes[current_note_index]

        corresp_lower_out = lower_out[:, lower_node_start:, :]
        higher = attention_weights(corresp_lower_out)
        higher_nodes.append(higher)

        higher_nodes = torch.cat(higher_nodes, dim=1).view(
            1, -1, lower_hidden_size)

        return higher_nodes

    def _span_beat_to_note_num(self, beat_out, beat_number, num_notes, start_index):
        start_beat = beat_number[start_index]
        num_beat = beat_out.shape[1]
        span_mat = torch.zeros(1, num_notes, num_beat)
        node_size = beat_out.shape[2]
        for i in range(num_notes):
            beat_index = beat_number[start_index+i] - start_beat
            if beat_index >= num_beat:
                beat_index = num_beat-1
            span_mat[0, i, beat_index] = 1
        span_mat = span_mat.to(self.device)

        spanned_beat = torch.bmm(span_mat, beat_out)
        return spanned_beat

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def _masking_half(self, y):
        num_notes = y.shape[1]
        y = y[:, :num_notes//2, :]
        return y

    def _encode_with_net(self, score_input, mean_net, var_net):
        mu = mean_net(score_input)
        var = var_net(score_input)

        z = self._reparameterize(mu, var)
        return z, mu, var

    def _init_hidden(self, num_layer, num_direction, batch_size, hidden_size):
        h0 = torch.zeros(num_layer * num_direction,
                         batch_size, hidden_size).to(self.device)
        return (h0, h0)

class Direct_Note(nn.Module):
    def __init__(self, model_config, device, cons, step_by_step=False):
        super(Direct_Note, self).__init__()
        self.device = device
        self.step_by_step = step_by_step
        self.config = model_config
        self.constants = cons

        self.note_fc = nn.Sequential(
            nn.Linear(self.config.input_size, self.config.note.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.config.note.size, self.config.note.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.config.note.size, self.config.note.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU(),
        )
        
        self.voice_net = nn.LSTM(self.config.note.size, self.config.voice.size, self.config.voice.layers,
                                 batch_first=True, bidirectional=True, dropout=cons.DROP_OUT)
        
        self.lstm = nn.LSTM(self.config.note.size, self.config.note.size, self.config.note.layers,
                            batch_first=True, bidirectional=True, dropout=cons.DROP_OUT)
        
        self.beat_attention = ContextAttention((self.config.note.size + self.config.voice.size) * 2,
                                               self.config.num_attention_head)
        self.beat_rnn = nn.LSTM((self.config.note.size + self.config.voice.size) * 2, self.config.beat.size,
                                self.config.beat.layers, batch_first=True, bidirectional=True, dropout=cons.DROP_OUT)

        self.measure_attention = ContextAttention(
            self.config.beat.size*2, self.config.num_attention_head)
        self.measure_rnn = nn.LSTM(self.config.beat.size * 2, self.config.measure.size, self.config.measure.layers,
                                   batch_first=True, bidirectional=True)

        self.result_for_tempo_attention = ContextAttention(
            self.config.output_size - 1, 1)
        self.beat_tempo_forward = nn.LSTM(
            (self.config.beat.size + self.config.measure.size) * 2 +
            5 + 3 + self.config.output_size + self.config.encoder.size,
            self.config.beat.size, num_layers=1, batch_first=True, bidirectional=False)
        self.beat_tempo_fc = nn.Linear(self.config.beat.size, 1)
        self.output_lstm = nn.LSTM(self.config.final.input, self.config.final.size, num_layers=self.config.final.layers,
                                   batch_first=True, bidirectional=False)
        self.final_fc = nn.Linear(self.config.final.size,
                            self.config.output_size - 1)
        

    def forward(self, x, y, edges, note_locations, start_index, initial_z=False, rand_threshold=0.2, return_z=False):
        '''
        score encoder part
        '''
        beat_numbers = [x.beat for x in note_locations]
        measure_numbers = [x.measure for x in note_locations]
        voice_numbers = [x.voice for x in note_locations]

        # changed
        #score_input = pass
        #e1_input = pass
        
        num_notes = x.size(1)

        note_out, beat_hidden_out, measure_hidden_out = \
            self._run_offline_score_model(x, beat_numbers, measure_numbers, voice_numbers, start_index)

        beat_out_spanned = self._span_beat_to_note_num(
            beat_hidden_out, beat_numbers, num_notes, start_index)
        measure_out_spanned = self._span_beat_to_note_num(measure_hidden_out, measure_numbers, num_notes,
                                                          start_index)

        '''
        output
        '''
        tempo_hidden = self._init_hidden(1,1,x.size(0), self.config.beat.size)
        num_beats = beat_hidden_out.size(1)
        #TODO: why -1?
        result_nodes = torch.zeros(num_beats, self.config.output_size - 1).to(self.device)

        num_measures = measure_numbers[start_index + num_notes - 1] - measure_numbers[start_index] + 1
        
        final_hidden = self._init_hidden(1, 1, x.size(0), self.config.final.size)
        qpm_primo = x[:, 0, self.constants.QPM_PRIMO_IDX]
        tempo_primo = x[0, 0,
                        self.constants.TEMPO_PRIMO_IDX:self.constants.TEMPO_PRIMO_IDX+self.constants.TEMPO_PRIMO_LEN]

        prev_out = torch.zeros(self.config.output_size).to(self.device)
        prev_tempo = prev_out[self.constants.OUTPUT_TEMPO_INDEX:
                                self.constants.OUTPUT_TEMPO_INDEX+1]
        prev_beat = -1
        prev_beat_end = 0
        out_total = torch.zeros(num_notes, self.config.output_size).to(self.device)
        prev_out_list = []
        has_ground_truth = y.size(1) > 1

        for i in range(num_notes):
            current_beat = beat_numbers[start_index + i] - beat_numbers[start_index]
            current_measure = measure_numbers[start_index + i] - measure_numbers[start_index]
            if current_beat > prev_beat:  # beat changed
                if i - prev_beat_end > 0:  # if there are outputs to consider
                    corresp_result = torch.stack(prev_out_list).unsqueeze_(0)
                else:  # there is no previous output
                    corresp_result = torch.zeros((1,1,self.config.output_size-1)).to(self.device)
                
                result_node = self.result_for_tempo_attention(corresp_result)
                prev_out_list = []
                result_nodes[current_beat, :] = result_node

                tempos = torch.zeros(1, num_beats, 1).to(self.device)

                beat_tempo_vec = x[0, i, self.constants.TEMPO_IDX:
                                   self.constants.TEMPO_IDX + self.constants.TEMPO_PARAM_LEN]
                beat_tempo_cat = torch.cat((beat_hidden_out[0, current_beat, :], measure_hidden_out[0, current_measure, :], prev_tempo,
                                            qpm_primo, tempo_primo, beat_tempo_vec,
                                            result_nodes[current_beat, :])).view(1, 1, -1)
                beat_forward, tempo_hidden = self.beat_tempo_forward(beat_tempo_cat, tempo_hidden)

                tmp_tempos = self.beat_tempo_fc(beat_forward)

                prev_beat_end = i
                prev_tempo = tmp_tempos.view(1)
                prev_beat = current_beat

            tmp_voice = voice_numbers[start_index + i] - 1

            out_combined = torch.cat(
                (note_out[0, i, :], beat_hidden_out[0, current_beat, :],
                 measure_hidden_out[0, current_measure, :],
                 prev_out, qpm_primo, tempo_primo)).view(1, 1, -1)

            out, final_hidden = self.output_lstm(out_combined, final_hidden)
            out = out.view(-1)
            out = self.final_fc(out)

            prev_out_list.append(out)
            out = torch.cat((prev_tempo, out))

            prev_out = out
            out_total[i, :] = out
        
        out_total = out_total.view(1, num_notes, -1)
        hidden_total = torch.cat(
            (note_out, beat_out_spanned, measure_out_spanned), 2)
        return out_total, hidden_total


    def _run_offline_score_model(self, x, beat_numbers, measure_numbers, voice_numbers, start_index):
        hidden = self._init_hidden(
            self.config.note.layers, 2, x.size(0), self.config.note.size)
        beat_hidden = self._init_hidden(
            self.config.beat.layers, 2, x.size(0), self.config.beat.size)
        measure_hidden = self._init_hidden(
            self.config.measure.layers, 2, x.size(0), self.config.measure.size)

        x = self.note_fc(x)

        temp_voice_numbers = voice_numbers[start_index:start_index + x.size(1)]
        if temp_voice_numbers == []:
            temp_voice_numbers = voice_numbers[start_index:]
        max_voice = max(temp_voice_numbers)
        voice_hidden = self._init_voice_layer(1, max_voice)
        voice_out, voice_hidden = self._run_voice_net(
            x, voice_hidden, temp_voice_numbers, max_voice)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        hidden_out, hidden = self.lstm(x, hidden)
        hidden_out = torch.cat((hidden_out, voice_out), 2)

        beat_nodes = self._make_higher_node(
            hidden_out, self.beat_attention, beat_numbers, beat_numbers, start_index, lower_is_note=True)
        beat_hidden_out, beat_hidden = self.beat_rnn(beat_nodes, beat_hidden)
        measure_nodes = self._make_higher_node(
            beat_hidden_out, self.measure_attention, beat_numbers, measure_numbers, start_index)
        measure_hidden_out, measure_hidden = self.measure_rnn(
            measure_nodes, measure_hidden)

        return hidden_out, beat_hidden_out, measure_hidden_out

    def _init_hidden(self, num_layer, num_direction, batch_size, hidden_size):
        h0 = torch.zeros(num_layer * num_direction,
                         batch_size, hidden_size).to(self.device)
        return (h0, h0)

    def _init_voice_layer(self, batch_size, max_voice):
        layers = []
        for i in range(max_voice):
            # h0 = torch.zeros(self.num_voice_layers * 2, batch_size, self.voice_hidden_size).to(device)
            h0 = torch.zeros(self.config.voice.layers * 2,
                             batch_size, self.config.voice.size).to(self.device)
            layers.append((h0, h0))
        return layers

    def _run_voice_net(self, batch_x, voice_hidden, voice_numbers, max_voice):
        num_notes = batch_x.size(1)
        output = torch.zeros(1, batch_x.size(
            1), self.config.voice.size * 2).to(self.device)
        voice_numbers = torch.Tensor(voice_numbers)
        for i in range(1, max_voice+1):
            voice_x_bool = voice_numbers == i
            num_voice_notes = torch.sum(voice_x_bool)
            if num_voice_notes > 0:
                span_mat = torch.zeros(num_notes, num_voice_notes)
                note_index_in_voice = 0
                for j in range(num_notes):
                    if voice_x_bool[j] == 1:
                        span_mat[j, note_index_in_voice] = 1
                        note_index_in_voice += 1
                span_mat = span_mat.view(1, num_notes, -1).to(self.device)
                voice_x = batch_x[0, voice_x_bool, :].view(
                    1, -1, self.config.note.size)
                ith_hidden = voice_hidden[i-1]

                ith_voice_out, ith_hidden = self.voice_net(voice_x, ith_hidden)
                # ith_voice_out, ith_hidden = self.lstm(voice_x, ith_hidden)
                output += torch.bmm(span_mat, ith_voice_out)
        return output, voice_hidden

    def _make_higher_node(self, lower_out, attention_weights, lower_indexes, higher_indexes, start_index, lower_is_note=False):
        higher_nodes = []
        prev_higher_index = higher_indexes[start_index]
        lower_node_start = 0
        lower_node_end = 0
        num_lower_nodes = lower_out.shape[1]
        start_lower_index = lower_indexes[start_index]
        lower_hidden_size = lower_out.shape[2]
        for low_index in range(num_lower_nodes):
            absolute_low_index = start_lower_index + low_index
            if lower_is_note:
                current_note_index = start_index + low_index
            else:
                current_note_index = lower_indexes.index(absolute_low_index)

            if higher_indexes[current_note_index] > prev_higher_index:
                # new beat start
                lower_node_end = low_index
                corresp_lower_out = lower_out[:,
                                              lower_node_start:lower_node_end, :]
                higher = attention_weights(corresp_lower_out)
                higher_nodes.append(higher)

                lower_node_start = low_index
                prev_higher_index = higher_indexes[current_note_index]

        corresp_lower_out = lower_out[:, lower_node_start:, :]
        higher = attention_weights(corresp_lower_out)
        higher_nodes.append(higher)

        higher_nodes = torch.cat(higher_nodes, dim=1).view(
            1, -1, lower_hidden_size)

        return higher_nodes

    def _span_beat_to_note_num(self, beat_out, beat_number, num_notes, start_index):
        start_beat = beat_number[start_index]
        num_beat = beat_out.shape[1]
        span_mat = torch.zeros(1, num_notes, num_beat)
        node_size = beat_out.shape[2]
        for i in range(num_notes):
            beat_index = beat_number[start_index+i] - start_beat
            if beat_index >= num_beat:
                beat_index = num_beat-1
            span_mat[0, i, beat_index] = 1
        span_mat = span_mat.to(self.device)

        spanned_beat = torch.bmm(span_mat, beat_out)
        return spanned_beat


class Direct_Measure(nn.Module):
    pass

# based on han_single_ar
class Classifier(nn.Module):
    def __init__(self, model_config, device, cons, step_by_step=False):
        super(Classifier, self).__init__()
        self.device = device
        self.step_by_step = step_by_step
        self.config = model_config
        self.constants = cons

        # score encoder
        self.note_fc = nn.Sequential(
            nn.Linear(self.config.input_size, self.config.note.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.config.note.size, self.config.note.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.config.note.size, self.config.note.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(self.config.note.size, self.config.note.size, self.config.note.layers,
                            batch_first=True, bidirectional=True, dropout=cons.DROP_OUT)

        self.voice_net = nn.LSTM(self.config.note.size, self.config.voice.size, self.config.voice.layers,
                                 batch_first=True, bidirectional=True, dropout=cons.DROP_OUT)

        self.beat_attention = ContextAttention((self.config.note.size + self.config.voice.size) * 2,
                                                       self.config.num_attention_head)
        self.beat_rnn = nn.LSTM((self.config.note.size + self.config.voice.size) * 2, self.config.beat.size,
                                        self.config.beat.layers, batch_first=True, bidirectional=True, dropout=cons.DROP_OUT)
        self.measure_attention = ContextAttention(self.config.beat.size*2, self.config.num_attention_head)
        self.measure_rnn = nn.LSTM(self.config.beat.size * 2, self.config.measure.size, self.config.measure.layers,
                                       batch_first=True, bidirectional=True)
        
        # performance encoder
        self.performance_embedding_layer = nn.Sequential(
            nn.Linear(self.config.output_size, self.config.note.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.config.note.size, self.config.note.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU()
        )
        self.performance_contractor = nn.Sequential(
            nn.Linear(self.config.encoder.input, self.config.encoder.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU()
        )
        self.performance_note_encoder = nn.LSTM(self.config.encoder.size, self.config.encoder.size, bidirectional=True)
        if self.config.encoder.size % self.config.num_attention_head == 0:
            self.performance_measure_attention = ContextAttention(self.config.encoder.size * 2, self.config.num_attention_head)
        else:
            self.performance_measure_attention = ContextAttention(self.config.encoder.size * 2, self.config.encoder.size * 2)
        
        self.performance_encoder = nn.LSTM(self.config.encoder.size * 2, self.config.encoder.size,
                                           num_layers=self.config.encoder.layers, batch_first=True, bidirectional=True)
        self.performance_final_attention = ContextAttention(self.config.encoder.size * 2, self.config.num_attention_head)
        self.performance_encoder_mean = nn.Linear(self.config.encoder.size * 2, self.config.encoded_vector_size)
        self.performance_encoder_var = nn.Linear(self.config.encoder.size * 2, self.config.encoded_vector_size)


        # performance decoder
        self.style_vector_expandor = nn.Sequential(
            nn.Linear(self.config.encoded_vector_size, self.config.encoder.size),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU()
        )
        self.perform_style_to_measure = nn.LSTM(self.config.measure.size * 2 + self.config.encoder.size,
                                                self.config.encoder.size, num_layers=1, bidirectional=False)
        self.make_performance_vector = nn.Sequential(
            nn.Linear(128, 16),
            nn.Dropout(cons.DROP_OUT),
            nn.ReLU()
        )

        # classifier network
        self.emotion_fc = nn.Linear(16, 5)

    def forward(self, score_data, eN_input, e1_input, note_locations, start_index, initial_z=False, rand_threshold=0.2, return_z=False):
        '''
        score encoder
        '''
        beat_numbers = [x.beat for x in note_locations]
        measure_numbers = [x.measure for x in note_locations]
        voice_numbers = [x.voice for x in note_locations]

        num_notes = score_data.size(1)
        note_out, beat_hidden_out, measure_hidden_out = \
            self._run_offline_score_model(score_data, beat_numbers, measure_numbers, voice_numbers, start_index)
        beat_out_spanned = self._span_higher_to_note_num(beat_hidden_out, beat_numbers, num_notes, start_index)
        measure_out_spanned = self._span_higher_to_note_num(measure_hidden_out, measure_numbers, num_notes, start_index)

        '''
        performance encoder: encoded score + emotion N performance
        '''
        original_perform_concat = torch.cat((e1_input, eN_input), 2)
        expanded_perform = self.performance_embedding_layer(original_perform_concat)
        perform_score_concat = torch.cat((note_out, beat_out_spanned, measure_out_spanned, expanded_perform), 2)
        perform_score_concat = self._masking_half(perform_score_concat)
        contracted = self.performance_contractor(perform_score_concat)
        perform_note_encoded, _ = self.performance_note_encoder(contracted)

        perform_measure = self._make_higher_node(perform_note_encoded, self.performance_measure_attention,
                                                beat_numbers, measure_numbers, start_index, lower_is_note=True)
        perform_style_encoded, _ = self.performance_encoder(perform_measure)
        perform_style_vector = self.performance_final_attention(perform_style_encoded)
        #perform_z, perform_mu, perform_var = \
        #    self._encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
        perform_z = self.make_performance_vector(perform_style_vector)
        
        '''
        performance decoder - depends on model 
        '''
        '''
        # perform_z : (1, 16)
        perform_z = self.style_vector_expandor(perform_z)
        perform_z_batched = perform_z.repeat(x.shape[1], 1).view(1,x.shape[1], -1)
        perform_z = perform_z.view(-1)
        # perform_z : (1, 64)

        
        num_measures = measure_numbers[start_index + num_notes - 1] - measure_numbers[start_index] + 1
        perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1, num_measures, -1)
        perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_hidden_out), 2)
        measure_perform_style, _ = self.perform_style_to_measure(perform_z_measure_cat)
        # measure_perform_style: (1, measure_num, 64)
        '''
        emotion_vec = self.emotion_fc(perform_z)    

        return emotion_vec



    def _run_offline_score_model(self, x, beat_numbers, measure_numbers, voice_numbers, start_index):
        hidden = self._init_hidden(
            self.config.note.layers, 2, x.size(0), self.config.note.size)
        beat_hidden = self._init_hidden(
            self.config.beat.layers, 2, x.size(0), self.config.beat.size)
        measure_hidden = self._init_hidden(
            self.config.measure.layers, 2, x.size(0), self.config.measure.size)

        x = self.note_fc(x)

        temp_voice_numbers = voice_numbers[start_index:start_index + x.size(1)]
        if temp_voice_numbers == []:
            temp_voice_numbers = voice_numbers[start_index:]
        max_voice = max(temp_voice_numbers)
        voice_hidden = self._init_voice_layer(1, max_voice)
        voice_out, voice_hidden = self._run_voice_net(
            x, voice_hidden, temp_voice_numbers, max_voice)
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        hidden_out, hidden = self.lstm(x, hidden)
        hidden_out = torch.cat((hidden_out, voice_out), 2)

        beat_nodes = self._make_higher_node(hidden_out, self.beat_attention, beat_numbers, beat_numbers, start_index, lower_is_note=True)
        beat_hidden_out, beat_hidden = self.beat_rnn(beat_nodes, beat_hidden)
        measure_nodes = self._make_higher_node(
            beat_hidden_out, self.measure_attention, beat_numbers, measure_numbers, start_index)
        measure_hidden_out, measure_hidden = self.measure_rnn(measure_nodes, measure_hidden)

        return hidden_out, beat_hidden_out, measure_hidden_out


    def _init_hidden(self, num_layer, num_direction, batch_size, hidden_size):
        h0 = torch.zeros(num_layer * num_direction, batch_size, hidden_size).to(self.device)
        return (h0, h0)

    def _init_voice_layer(self, batch_size, max_voice):
        layers = []
        for i in range(max_voice):
            # h0 = torch.zeros(self.num_voice_layers * 2, batch_size, self.voice_hidden_size).to(device)
            h0 = torch.zeros(self.config.voice.layers * 2, batch_size, self.config.voice.size).to(self.device)
            layers.append((h0, h0))
        return layers


    def _run_voice_net(self, batch_x, voice_hidden, voice_numbers, max_voice):
        num_notes = batch_x.size(1)
        output = torch.zeros(1, batch_x.size(1), self.config.voice.size * 2).to(self.device)
        voice_numbers = torch.Tensor(voice_numbers)
        for i in range(1, max_voice+1):
            voice_x_bool = voice_numbers == i
            num_voice_notes = torch.sum(voice_x_bool)
            if num_voice_notes > 0:
                span_mat = torch.zeros(num_notes, num_voice_notes)
                note_index_in_voice = 0
                for j in range(num_notes):
                    if voice_x_bool[j] == 1:
                        span_mat[j, note_index_in_voice] = 1
                        note_index_in_voice += 1
                span_mat = span_mat.view(1, num_notes, -1).to(self.device)
                voice_x = batch_x[0, voice_x_bool, :].view(1, -1, self.config.note.size)
                ith_hidden = voice_hidden[i-1]

                ith_voice_out, ith_hidden = self.voice_net(voice_x, ith_hidden)
                output += torch.bmm(span_mat, ith_voice_out)
        return output, voice_hidden

    def _make_higher_node(self, lower_out, attention_weights, lower_indexes, higher_indexes, start_index, lower_is_note=False):
        higher_nodes = []
        prev_higher_index = higher_indexes[start_index]
        lower_node_start = 0
        lower_node_end = 0
        num_lower_nodes = lower_out.shape[1]
        start_lower_index = lower_indexes[start_index]
        lower_hidden_size = lower_out.shape[2]
        for low_index in range(num_lower_nodes):
            absolute_low_index = start_lower_index + low_index
            if lower_is_note:
                current_note_index = start_index + low_index
            else:
                current_note_index = lower_indexes.index(absolute_low_index)

            if higher_indexes[current_note_index] > prev_higher_index:
                # new beat start
                lower_node_end = low_index
                corresp_lower_out = lower_out[:, lower_node_start:lower_node_end, :]
                higher = attention_weights(corresp_lower_out)
                higher_nodes.append(higher)

                lower_node_start = low_index
                prev_higher_index = higher_indexes[current_note_index]

        corresp_lower_out = lower_out[:, lower_node_start:, :]
        higher = attention_weights(corresp_lower_out)
        higher_nodes.append(higher)

        higher_nodes = torch.cat(higher_nodes, dim=1).view(1,-1,lower_hidden_size)

        return higher_nodes
    
    def _span_higher_to_note_num(self, higher_out, higher_number, num_notes, start_index):
        start_beat = higher_number[start_index]
        num_beat = higher_out.shape[1]
        span_mat = torch.zeros(1, num_notes, num_beat)
        node_size = higher_out.shape[2]
        for i in range(num_notes):
            beat_index = higher_number[start_index+i] - start_beat
            if beat_index >= num_beat:
                beat_index = num_beat-1
            span_mat[0,i,beat_index] = 1
        span_mat = span_mat.to(self.device)

        spanned_beat = torch.bmm(span_mat, higher_out)
        return spanned_beat

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def _masking_half(self, y):
        num_notes = y.shape[1]
        y = y[:, :num_notes//2, :]
        return y
    
    def _encode_with_net(self, score_input, mean_net, var_net):
        mu = mean_net(score_input)
        var = var_net(score_input)

        z = self._reparameterize(mu, var)
        return z, mu, var
