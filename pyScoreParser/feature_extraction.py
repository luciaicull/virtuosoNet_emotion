from . import xml_direction_encoding as dir_enc
from . import xml_utils, utils, feature_utils
import copy
import math
import warnings

class ScoreExtractor:
    """
    """
    def __init__(self, feature_keys):
        self.selected_feature_keys = feature_keys

        self.dyn_emb_tab = dir_enc.define_dyanmic_embedding_table()
        self.tem_emb_tab = dir_enc.define_tempo_embedding_table()

    def extract_score_features(self, piece_data):
        for key in self.selected_feature_keys:
            data, need_normalize = getattr(self, 'get_' + key)(piece_data)
            piece_data.score_features[key] = dict()
            piece_data.score_features[key]['data'] = data
            piece_data.score_features[key]['need_normalize'] = need_normalize

        return piece_data.score_features

    def crescendo_to_continuous_value(self, note, feature):
        cresc_words = ['cresc', 'decresc', 'dim']
        if feature.dynamic[1] != 0:
            for rel in note.dynamic.relative:
                for word in cresc_words:
                    if word in rel.type['type'] or word in rel.type['content']:
                        rel_length = rel.end_xml_position - rel.xml_position
                        if rel_length == float("inf") or rel_length == 0:
                            rel_length = note.state_fixed.divisions * 10
                        ratio = (note.note_duration.xml_position - rel.xml_position) / rel_length
                        feature.dynamic[1] *= (ratio + 0.05)
                        break

    def get_note_location(self, piece_data):
        # TODO: need check up
        locations = []
        for _, note in enumerate(piece_data.xml_notes):
            measure_index = note.measure_number - 1
            locations.append(
                feature_utils.NoteLocation(beat=utils.binary_index(piece_data.beat_positions, note.note_duration.xml_position),
                             measure=measure_index,
                             voice=note.voice,
                             section=utils.binary_index(piece_data.section_positions, note.note_duration.xml_position)))
        locations = feature_utils.make_index_continuous(locations)
        return locations, False

    def get_qpm_primo(self, piece_data):
        piece_data.qpm_primo = piece_data.xml_notes[0].state_fixed.qpm
        return piece_data.qpm_primo, True

    def get_midi_pitch(self, piece_data):
        return [note.pitch[1] for note in piece_data.xml_notes], True

    def get_pitch(self, piece_data):
        return [feature_utils.pitch_into_vector(
            note.pitch[1]) for note in piece_data.xml_notes], False

    def get_duration(self, piece_data):
        return [note.note_duration.duration / note.state_fixed.divisions
                for note in piece_data.xml_notes], True

    def get_grace_order(self, piece_data):
        return [note.note_duration.grace_order for note in piece_data.xml_notes], False

    def get_is_grace_note(self, piece_data):
        return [int(note.note_duration.is_grace_note) for note in piece_data.xml_notes], False

    def get_preceded_by_grace_note(self, piece_data):
        return [int(note.note_duration.preceded_by_grace_note) for note in piece_data.xml_notes], False

    def get_time_sig_vec(self, piece_data):
        return [feature_utils.time_signature_to_vector(note.tempo.time_signature) for note in piece_data.xml_notes], False

    def get_following_rest(self, piece_data):
        return [note.following_rest_duration /
                note.state_fixed.divisions for note in piece_data.xml_notes], True

    def get_followed_by_fermata_rest(self, piece_data):
        return [int(note.followed_by_fermata_rest) for note in piece_data.xml_notes], False

    def get_notation(self, piece_data):
        return [feature_utils.note_notation_to_vector(
            note) for note in piece_data.xml_notes], False

    def get_slur_beam_vec(self, piece_data):
        return [[int(note.note_notations.is_slur_start),
                 int(note.note_notations.is_slur_continue),
                 int(note.note_notations.is_slur_stop),
                 int(note.note_notations.is_beam_start),
                 int(note.note_notations.is_beam_continue),
                 int(note.note_notations.is_beam_stop)]
                for note in piece_data.xml_notes], False

    def get_dynamic(self, piece_data):
        return [
            dir_enc.dynamic_embedding(
                dir_enc.direction_words_flatten(note.dynamic), self.dyn_emb_tab, len_vec=4)
            for note in piece_data.xml_notes], False

    def get_tempo(self, piece_data):
        return [
            dir_enc.dynamic_embedding(
                dir_enc.direction_words_flatten(note.tempo), self.tem_emb_tab, len_vec=5)
            for note in piece_data.xml_notes], False

    def get_xml_position(self, piece_data):
        total_length = xml_utils.cal_total_xml_length(
            piece_data.xml_notes)
        return [note.note_duration.xml_position /
                total_length for note in piece_data.xml_notes], False

    def get_beat_position(self, piece_data):
        beat_positions = []
        for _, note in enumerate(piece_data.xml_notes):
            measure_index = note.measure_number - 1
            note_position = note.note_duration.xml_position

            if measure_index + 1 < len(piece_data.measure_positions):
                measure_length = piece_data.measure_positions[measure_index +
                                                                1] - piece_data.measure_positions[measure_index]
            else:
                measure_length = piece_data.measure_positions[measure_index] - \
                    piece_data.measure_positions[measure_index - 1]
            beat_position = (
                note_position - piece_data.measure_positions[measure_index]) / measure_length
            beat_positions.append(beat_position)
        return beat_positions, False

    def get_beat_importance(self, piece_data):
        try: 
            beat_positions, _ = piece_data.score_features['beat_position']
        except:
            beat_positions, _ = self.get_beat_position(piece_data)
        beat_importances = []
        for i, note in enumerate(piece_data.xml_notes):
            importance = feature_utils.cal_beat_importance(
                beat_positions[i], note.tempo.time_numerator)
            beat_importances.append(importance)
        return beat_importances, True

    def get_measure_length(self, piece_data):
        measure_lengthes = []
        for _, note in enumerate(piece_data.xml_notes):
            measure_index = note.measure_number - 1

            if measure_index + 1 < len(piece_data.measure_positions):
                measure_length = piece_data.measure_positions[measure_index +
                                                                1] - piece_data.measure_positions[measure_index]
            else:
                measure_length = piece_data.measure_positions[measure_index] - \
                    piece_data.measure_positions[measure_index - 1]
            measure_lengthes.append(
                measure_length / note.state_fixed.divisions)
        return measure_lengthes, True

    def get_cresciuto(self, piece_data):
        # This function converts cresciuto class information into single numeric value
        cresciutos = []
        for note in piece_data.xml_notes:
            if note.dynamic.cresciuto:
                cresciuto = (note.dynamic.cresciuto.overlapped + 1) / 2
                if note.dynamic.cresciuto.type == 'diminuendo':
                    cresciuto *= -1
            else:
                cresciuto = 0
            cresciutos.append(cresciuto)
        return cresciutos, True

    def get_distance_from_abs_dynamic(self, piece_data):
        return [(note.note_duration.xml_position - note.dynamic.absolute_position)
                / note.state_fixed.divisions for note in piece_data.xml_notes], True

    def get_distance_from_recent_tempo(self, piece_data):
        return [(note.note_duration.xml_position - note.tempo.recently_changed_position)
                / note.state_fixed.divisions for note in piece_data.xml_notes], True

    def get_composer_vec(self, piece_data):
        return feature_utils.composer_name_to_vec(piece_data.composer), False

    def get_tempo_primo(self, piece_data):
        tempo_primo_word = dir_enc.direction_words_flatten(
            piece_data.xml_notes[0].tempo)
        if tempo_primo_word:
            piece_data.tempo_primo = dir_enc.dynamic_embedding(
                tempo_primo_word, self.tem_emb_tab, 5)
            piece_data.tempo_primo = piece_data.tempo_primo[0:2]
        else:
            piece_data.tempo_primo = [0, 0]
        return piece_data.tempo_primo, False

class PerformExtractor:
    def __init__(self, selected_feature_keys):
        self.selected_feature_keys = selected_feature_keys

    def extract_perform_features(self, piece_data, perform_data):
        # perform_data.perform_features = {}
        for feature_key in self.selected_feature_keys:
            data, need_normalize = getattr(self, 'get_' + feature_key)(piece_data, perform_data)
            if type(data) is dict:
                for key in data.keys():
                    perform_data.perform_features[key] = dict()
                    perform_data.perform_features[key]['data'] = data[key]['data']
                    perform_data.perform_features[key]['need_normalize'] = data[key]['need_normalize']

            else:
                perform_data.perform_features[feature_key] = dict()
                perform_data.perform_features[feature_key]['data'] = data
                perform_data.perform_features[feature_key]['need_normalize'] = need_normalize

        return perform_data.perform_features
    
    def get_emotion(self, piece_data, perform_data):
        emotion_vector = [0, 0, 0, 0, 0] # [origin, relaxed, sad, happy, angry]
        emotion_num = perform_data.emotion
        try:
            emotion_vector[emotion_num-1] = 1
        except Exception as e:
            print("No emotion")
            print(e)
        
        return emotion_vector, False

    def get_beat_tempo(self, piece_data, perform_data):
        tempos = feature_utils.cal_tempo_by_positions(piece_data.beat_positions, perform_data.valid_position_pairs)
        # update tempos for perform data
        perform_data.tempos = tempos
        perform_data.beat_tempos = [t.qpm for t in tempos]
        return [math.log(utils.get_item_by_xml_position(tempos, note).qpm, 10) for note in piece_data.xml_notes], True

    def get_measure_tempo(self, piece_data, perform_data):
        tempos = feature_utils.cal_tempo_by_positions(piece_data.measure_positions, perform_data.valid_position_pairs)
        perform_data.measure_tempos = [t.qpm for t in tempos]
        return [math.log(utils.get_item_by_xml_position(tempos, note).qpm, 10) for note in piece_data.xml_notes], True

    def get_section_tempo(self, piece_data, perform_data):
        tempos = feature_utils.cal_tempo_by_positions(piece_data.section_positions, perform_data.valid_position_pairs)
        perform_data.section_tempos = [t.qpm for t in tempos]
        return [math.log(utils.get_item_by_xml_position(tempos, note).qpm, 10) for note in piece_data.xml_notes], True

    def get_qpm_primo(self, piece_data, perform_data, view_range=10):
        if 'beat_tempo' not in perform_data.perform_features:
            perform_data.perform_features['beat_tempo'] = dict()
            perform_data.perform_features['beat_tempo']['data'], perform_data.perform_features['beat_tempo']['need_normalize'] = self.get_beat_tempo(
                piece_data, perform_data)
        qpm_primo = 0
        for i in range(view_range):
            tempo = perform_data.tempos[i]
            qpm_primo += tempo.qpm

        return math.log(qpm_primo / view_range, 10), True

    def get_elongated_duration(self, piece_data, perform_data):
        features = []
        for i, pair in enumerate(perform_data.pairs):
            if pair == []:
                duration = 0
            else:
                note = pair['xml']
                midi = pair['midi']
                if midi.elongated_offset_time > midi.end:
                    duration = midi.elongated_offset_time - midi.start
                else:
                    duration = midi.end - midi.start
            features.append(duration)
        
        return features, True
    
    def get_not_elongated_duration(self, piece_data, perform_data):
        features = []
        for i, pair in enumerate(perform_data.pairs):
            if pair == []:
                duration = 0
            else:
                note = pair['xml']
                midi = pair['midi']
                duration = midi.end - midi.start
            features.append(duration)

        return features, True

    def get_articulation(self, piece_data, perform_data):
        features = []
        if 'beat_tempo' not in perform_data.perform_features:
            perform_data.perform_features['beat_tempo'] = dict()
            perform_data.perform_features['beat_tempo']['data'],  perform_data.perform_features['beat_tempo']['need_normalize'] = self.get_beat_tempo(
                piece_data, perform_data)
        if 'num_trills' not in perform_data.perform_features:
            data, _ = self.get_trill_parameters(piece_data, perform_data)
            for key in data.keys():
                perform_data.perform_features[key] = dict()
                perform_data.perform_features[key]['data'] = data[key]['data']
                perform_data.perform_features[key]['need_normalize'] = data[key]['need_normalize']
        for i, pair in enumerate(perform_data.pairs):
            if pair == []:
                articulation = 0
            else:
                note = pair['xml']
                midi = pair['midi']
                tempo = utils.get_item_by_xml_position(perform_data.tempos, note)
                xml_duration = note.note_duration.duration
                if xml_duration == 0:
                    articulation = 1
                elif note.is_overlapped:
                    articulation = 1
                else:
                    duration_as_quarter = xml_duration / note.state_fixed.divisions
                    second_in_tempo = duration_as_quarter / tempo.qpm * 60
                    if note.note_notations.is_trill:
                        _, actual_second = xml_utils.find_corresp_trill_notes_from_midi(piece_data, perform_data, i)
                    else:
                        actual_second = midi.end - midi.start
                    articulation = actual_second / second_in_tempo
                    if actual_second == 0:
                        articulation = 1
                    if articulation <= 0:
                        # print('Articulation error: tempo.qpm was{}, ')
                    # if articulation > 6:
                        print('check: articulation is {} in {}th note of perform {}. '
                              'Tempo QPM was {}, in-tempo duration was {} seconds and was performed in {} seconds'
                              .format(articulation, i , perform_data.midi_path, tempo.qpm, second_in_tempo, actual_second))
                        print('xml_duration of note was {}'.format(xml_duration / note.state_fixed.divisions))
                        print('midi start was {} and end was {}'.format(midi.start, midi.end))
                articulation = math.log(articulation, 10)
            features.append(articulation)

        return features, True

    def get_onset_deviation(self, piece_data, perform_data):
        ''' Deviation of individual note's onset
        :param piece_data: PieceData class
        :param perform_data: PerformData class
        :return: note-level onset deviation of performance notes in quarter-notes
        Onset deviation is defined as how much the note's onset is apart from its "in-tempo" position.
        The "in-tempo" position is defined by pre-calculated beat-level tempo.
        '''

        features = []
        if 'beat_tempo' not in perform_data.perform_features:
            perform_data.perform_features['beat_tempo'] = dict()
            perform_data.perform_features['beat_tempo']['data'],  perform_data.perform_features['beat_tempo']['need_normalize'] = self.get_beat_tempo(
                piece_data, perform_data)
        for pair in perform_data.pairs:
            if pair == []:
                deviation = 0
            else:
                note = pair['xml']
                midi = pair['midi']
                tempo = utils.get_item_by_xml_position(perform_data.tempos, note)
                tempo_start = tempo.time_position

                passed_duration = note.note_duration.xml_position - tempo.xml_position
                actual_passed_second = midi.start - tempo_start
                actual_passed_duration = actual_passed_second / 60 * tempo.qpm * note.state_fixed.divisions

                xml_pos_difference = actual_passed_duration - passed_duration
                pos_diff_in_quarter_note = xml_pos_difference / note.state_fixed.divisions
                # deviation_time = xml_pos_difference / note.state_fixed.divisions / tempo_obj.qpm * 60
                # if pos_diff_in_quarter_note >= 0:
                #     pos_diff_sqrt = math.sqrt(pos_diff_in_quarter_note)
                # else:
                #     pos_diff_sqrt = -math.sqrt(-pos_diff_in_quarter_note)
                # pos_diff_cube_root = float(pos_diff_
                deviation = pos_diff_in_quarter_note
            features.append(deviation)
        return features, True

    def get_align_matched(self, piece_data, perform_data):
        features = []
        for pair in perform_data.pairs:
            if pair == []:
                matched = 0
            else:
                matched = 1
            features.append(matched)
        return features, False

    def get_velocity(self, piece_data, perform_data):
        '''
        :param piece_data:
        :param perform_data:
        :return: List of MIDI velocities of notes in score-performance pair.
        '''
        features = []
        prev_velocity = 64
        for pair in perform_data.pairs:
            if pair == []:
                velocity = prev_velocity
            else:
                velocity = pair['midi'].velocity
                prev_velocity = velocity
            features.append(velocity)
        return features, True

    # TODO: get pedal _ can be simplified

    def get_pedal_at_start(self, piece_data, perform_data):
        features = []
        prev_pedal = 0
        for pair in perform_data.pairs:
            if pair == []:
                pedal = prev_pedal
            else:
                pedal = feature_utils.pedal_sigmoid(pair['midi'].pedal_at_start)
                prev_pedal = pedal
            features.append(pedal)
        return features, True

    def get_pedal_at_end(self, piece_data, perform_data):
        features = []
        prev_pedal = 0
        for pair in perform_data.pairs:
            if pair == []:
                pedal = prev_pedal
            else:
                pedal = feature_utils.pedal_sigmoid(pair['midi'].pedal_at_end)
                prev_pedal = pedal
            features.append(pedal)
        return features, True

    def get_pedal_refresh(self, piece_data, perform_data):
        features = []
        for pair in perform_data.pairs:
            if pair == []:
                pedal = 0
            else:
                pedal = feature_utils.pedal_sigmoid(pair['midi'].pedal_refresh)
            features.append(pedal)
        return features, True

    def get_pedal_refresh_time(self, piece_data, perform_data):
        features = []
        for pair in perform_data.pairs:
            if pair == []:
                pedal = 0
            else:
                pedal = feature_utils.pedal_sigmoid(pair['midi'].pedal_refresh_time)
            features.append(pedal)
        return features, True

    def get_pedal_cut(self, piece_data, perform_data):
        features = []
        prev_pedal = 0
        for pair in perform_data.pairs:
            if pair == []:
                pedal = prev_pedal
            else:
                pedal = feature_utils.pedal_sigmoid(pair['midi'].pedal_cut)
                prev_pedal = pedal
            features.append(pedal)
        return features, True

    def get_pedal_cut_time(self, piece_data, perform_data):
        features = []
        for pair in perform_data.pairs:
            if pair == []:
                pedal = 0
            else:
                pedal = feature_utils.pedal_sigmoid(pair['midi'].pedal_cut_time)
            features.append(pedal)
        return features, True

    def get_soft_pedal(self, piece_data, perform_data):
        features = []
        prev_pedal = 0
        for pair in perform_data.pairs:
            if pair == []:
                pedal = prev_pedal
            else:
                pedal = feature_utils.pedal_sigmoid(pair['midi'].soft_pedal)
                prev_pedal = pedal
            features.append(pedal)
        return features, True

    def get_attack_deviation(self, piece, perform):
        previous_xml_onset = 0
        previous_onset_timings = []
        previous_onset_indices = []
        attack_deviations = [0] * len(perform.pairs)
        for i, pair in enumerate(perform.pairs):
            if pair == []:
                attack_deviations[i] = 0
                continue
            if pair['xml'].note_duration.xml_position > previous_xml_onset:
                if previous_onset_timings != []:
                    avg_onset_time = sum(previous_onset_timings) / len(previous_onset_timings)
                    for j, prev_idx in enumerate(previous_onset_indices):
                        attack_deviations[prev_idx] = abs(previous_onset_timings[j] - avg_onset_time)
                    previous_onset_timings = []
                    previous_onset_indices = []

            previous_onset_timings.append(pair['midi'].start)
            previous_onset_indices.append(i)
            previous_xml_onset = pair['xml'].note_duration.xml_position

        if previous_onset_timings != []:
            avg_onset_time = sum(previous_onset_timings) / len(previous_onset_timings)
            for j, prev_idx in enumerate(previous_onset_indices):
                attack_deviations[prev_idx] = abs(previous_onset_timings[j] - avg_onset_time)
                
        return attack_deviations, True

    def get_non_abs_attack_deviation(self, piece, perform):
        previous_xml_onset = 0
        previous_onset_timings = []
        previous_onset_indices = []
        attack_deviations = [0] * len(perform.pairs)
        for i, pair in enumerate(perform.pairs):
            if pair == []:
                attack_deviations[i] = 0
                continue
            if pair['xml'].note_duration.xml_position > previous_xml_onset:
                if previous_onset_timings != []:
                    avg_onset_time = sum(previous_onset_timings) / len(previous_onset_timings)
                    for j, prev_idx in enumerate(previous_onset_indices):
                        attack_deviations[prev_idx] = previous_onset_timings[j] - avg_onset_time
                    previous_onset_timings = []
                    previous_onset_indices = []

            previous_onset_timings.append(pair['midi'].start)
            previous_onset_indices.append(i)
            previous_xml_onset = pair['xml'].note_duration.xml_position

        if previous_onset_timings != []:
            avg_onset_time = sum(previous_onset_timings) / len(previous_onset_timings)
            for j, prev_idx in enumerate(previous_onset_indices):
                attack_deviations[prev_idx] = previous_onset_timings[j] - avg_onset_time

        return attack_deviations, True

    def get_tempo_fluctuation(self, piece, perform):
        tempo_fluctuations = [None] * len(perform.pairs)
        for i in range(1, len(perform.pairs)):
            prev_qpm = perform.perform_features['beat_tempo'][i - 1]
            curr_qpm = perform.perform_features['beat_tempo'][i]
            if curr_qpm == prev_qpm:
                continue
            else:
                tempo_fluctuations[i] = abs(curr_qpm - prev_qpm)
        return tempo_fluctuations, True

    def get_abs_deviation(self, piece, perform):
        return [abs(x) for x in perform.perform_features['onset_deviation']], True

    def get_left_hand_velocity(self, piece, perform):
        features = [None] * len(perform.pairs)
        for i, pair in enumerate(perform.pairs):
            if pair != [] and pair['xml'].staff == 2:
                features[i] = pair['midi'].velocity
        return features, True

    def get_right_hand_velocity(self, piece, perform):
        features = [None] * len(perform.pairs)
        for i, pair in enumerate(perform.pairs):
            if pair != [] and pair['xml'].staff == 1:
                features[i] = pair['midi'].velocity
        return features, True

    def get_articulation_loss_weight(self, piece_data, perform_data):
        if 'pedal_at_end' not in perform_data.perform_features:
            perform_data.perform_features['pedal_at_end'] = dict()
            perform_data.perform_features['pedal_at_end']['data'], perform_data.perform_features['pedal_at_end']['need_normalize'] = self.get_pedal_at_end(
                piece_data, perform_data)
        if 'pedal_refresh' not in perform_data.perform_features:
            perform_data.perform_features['pedal_refresh'] = dict()
            perform_data.perform_features['pedal_refresh']['data'], perform_data.perform_features['pedal_refresh']['need_normalize'] = self.get_pedal_at_end(
                piece_data, perform_data)
        features = []
        for pair, pedal, pedal_refresh in zip(perform_data.pairs,
                                              perform_data.perform_features['pedal_at_end']['data'],
                                              perform_data.perform_features['pedal_refresh']['data']):
            if pair == []:
                articulation_loss_weight = 0
            elif pedal > 70:
                articulation_loss_weight = 0.05
            elif pedal > 60:
                articulation_loss_weight = 0.5
            else:
                articulation_loss_weight = 1

            if pedal > 64 and pedal_refresh < 64:
                # pedal refresh occurs in the note
                articulation_loss_weight = 1

            features.append(articulation_loss_weight)
        return features, False

    def get_trill_parameters(self, piece_data, perform_data):
        features = {'num_trills': {'data': [], 'need_normalize': True}, 'trill_last_note_velocity': {'data': [], 'need_normalize': True},
                    'trill_first_note_ratio': {'data': [], 'need_normalize': True}, 'trill_last_note_ratio': {'data': [], 'need_normalize': True}, 'up_trill': {'data': [], 'need_normalize': False}}
        for i, note in enumerate(piece_data.xml_notes):
            if note.note_notations.is_trill:
                trill_parameter_list, _ = xml_utils.find_corresp_trill_notes_from_midi(piece_data, perform_data, i)
            else:
                trill_parameter_list = [0, 0, 0, 0, 0]
            num_trills, last_note_velocity, first_note_ratio, last_note_ratio, up_trill = trill_parameter_list
            
            features['num_trills']['data'].append(num_trills)
            features['trill_last_note_velocity']['data'].append(last_note_velocity)
            features['trill_first_note_ratio']['data'].append(first_note_ratio)
            features['trill_last_note_ratio']['data'].append(last_note_ratio)
            features['up_trill']['data'].append(up_trill)

        return features, True

    def get_left_hand_attack_deviation(self, piece, perform):
        if 'non_abs_attack_deviation' not in perform.perform_features:
            perform.perform_features['non_abs_attack_deviation'] = dict()
            perform.perform_features['non_abs_attack_deviation']['data'], perform.perform_features[
                'non_abs_attack_deviation']['need_normalize'] = self.get_non_abs_attack_deviation(piece, perform)
        features = [None] * len(perform.pairs)
        for i, pair in enumerate(perform.pairs):
            if pair != [] and pair['xml'].staff == 2:
                features[i] = perform.perform_features['non_abs_attack_deviation'][i]
        return features, True

    def get_right_hand_attack_deviation(self, piece, perform):
        if 'non_abs_attack_deviation' not in perform.perform_features:
            perform.perform_features['non_abs_attack_deviation'] = dict()
            perform.perform_features['non_abs_attack_deviation']['data'], perform.perform_features[
                'non_abs_attack_deviation']['need_normalize'] = self.get_non_abs_attack_deviation(piece, perform)
        features = [None] * len(perform.pairs)
        for i, pair in enumerate(perform.pairs):
            if pair != [] and pair['xml'].staff == 1:
                features[i] = perform.perform_features['non_abs_attack_deviation'][i]
        return features, True

    def get_beat_dynamics(self, piece_data, perform_data):
        if 'velocity' not in perform_data.perform_features:
            perform_data.perform_features['velocity'] = dict()
            perform_data.perform_features['velocity']['data'], perform_data.perform_features['velocity']['need_normalize'] = self.get_velocity(
                piece_data, perform_data)
        if 'align_matched' not in perform_data.perform_features:
            perform_data.perform_features['align_matched'] = dict()
            perform_data.perform_features['align_matched']['data'], perform_data.perform_features['align_matched']['need_normalize'] = self.get_align_matched(
                piece_data, perform_data)
        if 'note_location' not in piece_data.score_features:
            score_extractor = ScoreExtractor(['note_location'])
            piece_data.score_features = score_extractor.extract_score_features(piece_data)
        return feature_utils.get_longer_level_dynamics(perform_data.perform_features, piece_data.score_features['note_location'], length='beat'), True

    def get_measure_dynamics(self, piece_data, perform_data):
        if 'velocity' not in perform_data.perform_features:
            perform_data.perform_features['velocity'] = dict()
            perform_data.perform_features['velocity']['data'], perform_data.perform_features['velocity']['need_normalize'] = self.get_velocity(
                piece_data, perform_data)
        if 'align_matched' not in perform_data.perform_features:
            perform_data.perform_features['align_matched'] = dict()
            perform_data.perform_features['align_matched']['data'], perform_data.perform_features['align_matched']['need_normalize'] = self.get_align_matched(
                piece_data, perform_data)
        if 'note_location' not in piece_data.score_features:
            score_extractor = ScoreExtractor(['note_location'])
            piece_data.score_features = score_extractor.extract_score_features(piece_data)

        return feature_utils.get_longer_level_dynamics(perform_data.perform_features,
                                                       piece_data.score_features['note_location'], length='measure'), True

    def get_staff(self, piece_data, perform_data):
        return [note.staff for note in piece_data.xml_notes], True
