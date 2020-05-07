import os
import _pickle as cPickle
import copy
import pretty_midi

from pyScoreParser.musicxml_parser import MusicXMLDocument
from pyScoreParser import xml_utils, pedal_cleaning
from pyScoreParser.utils import binary_index

def read_xml_to_notes(path):
    xml_object = MusicXMLDocument(path)
    notes, rests = xml_object.get_notes()
    directions = xml_object.get_directions()
    time_signatures = xml_object.get_time_signatures()

    xml_notes = xml_utils.apply_directions_to_notes(
        notes, directions, time_signatures)
    
    return xml_object, xml_notes

def get_score_information(xml_path, feature_path, feature_stats,
                      start_tempo, composer,
                      vel_pair):
    xml_object, xml_notes = read_xml_to_notes(xml_path)
    beats = xml_object.get_beat_positions()
    measure_positions = xml_object.get_measure_positions()
    
    with open(feature_path, "rb") as f:
        u = cPickle.Unpickler(f)
        feature_dict = u.load()
    
    # TODO: not an array I think..?
    test_x = feature_dict['input_data']
    note_locations = feature_dict['note_location']['data']
    edges = feature_dict['graph']

    return test_x, xml_notes, xml_object, edges, note_locations
    
def get_average_onset_time(notes_in_chord_saved, threshold=0.2):
    # notes_in_chord: list of PosTempoPair, len > 0
    notes_in_chord = copy.copy(notes_in_chord_saved)
    average_onset_time = 0
    for pos_pair in notes_in_chord:
        average_onset_time += pos_pair.time_position
        if pos_pair.is_arpeggiate:
            threshold = 1
    average_onset_time /= len(notes_in_chord)

    # check whether there is mis-matched note
    deviations = list()
    for pos_pair in notes_in_chord:
        dev = abs(pos_pair.time_position - average_onset_time)
        deviations.append(dev)
    if max(deviations) > threshold:
        # print(deviations)
        if len(notes_in_chord) == 2:
            del notes_in_chord[0:2]
        else:
            index = deviations.index(max(deviations))
            del notes_in_chord[index]
            notes_in_chord, average_onset_time = get_average_onset_time(
                notes_in_chord, threshold)

    return notes_in_chord, average_onset_time

def make_average_onset_cleaned_pair(position_pairs, maximum_qpm=600):
    previous_position = -float("Inf")
    previous_time = -float("Inf")
    previous_index = 0
    cleaned_list = []
    notes_in_chord = []
    mismatched_indexes = []

    for i, pair in enumerate(position_pairs):
        current_position = pair.xml_position
        current_time = pair.time_position
        if current_position > previous_position >= 0:
            minimum_time_interval = (
                current_position - previous_position) / pair.divisions / maximum_qpm * 60 + 0.01
        else:
            minimum_time_interval = 0
        if current_position > previous_position and current_time > previous_time + minimum_time_interval:
            if len(notes_in_chord) > 0:
                average_pos_pair = copy.copy(notes_in_chord[0])
                notes_in_chord_cleaned, average_pos_pair.time_position = get_average_onset_time(
                    notes_in_chord)
                if len(cleaned_list) == 0 or average_pos_pair.time_position > cleaned_list[-1].time_position + (
                        (average_pos_pair.xml_position - cleaned_list[-1].xml_position) /
                        average_pos_pair.divisions / maximum_qpm * 60 + 0.01):
                    cleaned_list.append(average_pos_pair)
                    for note in notes_in_chord:
                        if note not in notes_in_chord_cleaned:
                            # print('the note is far from other notes in the chord')
                            mismatched_indexes.append(note.index)
                else:
                    # print('the onset is too close to the previous onset', average_pos_pair.xml_position, cleaned_list[-1].xml_position, average_pos_pair.time_position, cleaned_list[-1].time_position)
                    for note in notes_in_chord:
                        mismatched_indexes.append(note.index)
            notes_in_chord = []
            notes_in_chord.append(pair)
            previous_position = current_position
            previous_time = current_time
            previous_index = i
        elif current_position == previous_position:
            notes_in_chord.append(pair)
        else:
            # print('the note is too close to the previous note', current_position - previous_position, current_time - previous_time)
            # print(previous_position, current_position, previous_time, current_time)
            mismatched_indexes.append(position_pairs[previous_index].index)
            mismatched_indexes.append(pair.index)

    return cleaned_list, mismatched_indexes


def make_available_note_feature_list(notes, features):
    class PosTempoPair:
        def __init__(self, xml_pos, pitch, qpm, index, divisions, time_pos):
            self.xml_position = xml_pos
            self.qpm = qpm
            self.index = index
            self.divisions = divisions
            self.pitch = pitch
            self.time_position = time_pos
            self.is_arpeggiate = False
    
    available_notes = []
    for i, feat in enumerate(features):
        xml_note = notes[i]
        xml_pos = xml_note.note_duration.xml_position
        time_pos = xml_note.note_duration.time_position
        divisions = xml_note.state_fixed.divisions
        pos_pair = PosTempoPair(xml_pos, xml_note.pitch[1], feat['qpm'], i, divisions, time_pos)
        available_notes.append(pos_pair)

    available_notes, _ = make_average_onset_cleaned_pair(available_notes)
    return available_notes


def get_item_by_xml_position(alist, item):
    if hasattr(item, 'xml_position'):
        item_pos = item.xml_position
    elif hasattr(item, 'note_duration'):
        item_pos = item.note_duration.xml_position
    elif hasattr(item, 'start_xml_position'):
        item_pos = item.start.xml_position
    else:
        item_pos = item

    repre = alist[0]

    if hasattr(repre, 'xml_position'):
        pos_list = [x.xml_position for x in alist]
    elif hasattr(repre, 'note_duration'):
        pos_list = [x.note_duration.xml_position for x in alist]
    elif hasattr(item, 'start_xml_position'):
        pos_list = [x.start_xml_position for x in alist]
    else:
        pos_list = alist

    index = binary_index(pos_list, item_pos)

    return alist[index]

class Tempo:
    def __init__(self, xml_position, qpm, time_position, end_xml, end_time):
        self.qpm = qpm
        self.xml_position = xml_position
        self.time_position = time_position
        self.end_time = end_time
        self.end_xml = end_xml

    def __str__(self):
        string = '{From ' + str(self.xml_position)
        string += ' to ' + str(self.end_xml)
        return string


def apply_feat_to_a_note(note, feat, prev_vel):
    if 'articulation' in feat.keys():
        note.note_duration.seconds *= 10 ** (feat['articulation'])
        # note.note_duration.seconds *= feat.articulation

    if 'velocity' in feat.keys():
        note.velocity = feat['velocity']
        prev_vel = note.velocity
    else:
        note.velocity = prev_vel

    if 'pedal_at_start' in feat.keys():
        note.pedal.at_start = int(round(feat['pedal_at_start']))
        note.pedal.at_end = int(round(feat['pedal_at_end']))
        note.pedal.refresh = int(round(feat['pedal_refresh']))
        note.pedal.refresh_time = feat['pedal_refresh_time']
        note.pedal.cut = int(round(feat['pedal_cut']))
        note.pedal.cut_time = feat['pedal_cut_time']
        note.pedal.soft = int(round(feat['soft_pedal']))
    return note, prev_vel


def get_measure_accidentals(xml_notes, index):
    accs = ['bb', 'b', '♮', '#', 'x']
    note = xml_notes[index]
    num_note = len(xml_notes)
    measure_accidentals=[]
    for i in range(1,num_note):
        prev_note = xml_notes[index - i]
        if prev_note.measure_number != note.measure_number:
            break
        else:
            for acc in accs:
                if acc in prev_note.pitch[0]:
                    pitch = prev_note.pitch[0][0] + prev_note.pitch[0][-1]
                    for prev_acc in measure_accidentals:
                        if prev_acc['pitch'] == pitch:
                            break
                    else:
                        accident = accs.index(acc) - 2
                        temp_pair = {'pitch': pitch, 'accident': accident}
                        measure_accidentals.append(temp_pair)
                        break
    return measure_accidentals

def cal_up_trill_pitch(pitch_tuple, key, final_key, measure_accidentals):
    pitches = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
    corresp_midi_pitch = [0, 2, 4, 5, 7, 9, 11]
    pitch_name = pitch_tuple[0][0:1].lower()
    octave = int(pitch_tuple[0][-1])
    next_pitch_name = pitches[(pitches.index(pitch_name)+1)%7]
    if next_pitch_name == 'c':
        octave += 1

    accidentals = ['f', 'c', 'g', 'd', 'a', 'e', 'b']
    if key > 0 and next_pitch_name in accidentals[:key]:
        acc = +1
    elif key < 0 and next_pitch_name in accidentals[key:]:
        acc = -1
    else:
        acc= 0

    pitch_string = next_pitch_name + str(octave)
    for pitch_acc_pair in measure_accidentals:
        if pitch_string == pitch_acc_pair['pitch'].lower():
            acc = pitch_acc_pair['accident']

    if not final_key == None:
        acc= final_key

    if acc == 0:
        acc_in_string = ''
    elif acc ==1:
        acc_in_string = '#'
    elif acc ==-1:
        acc_in_string = '♭'
    else:
        acc_in_string = ''
    final_pitch_string = next_pitch_name.capitalize() + acc_in_string + str(octave)
    up_pitch = 12 * (octave + 1) + corresp_midi_pitch[pitches.index(next_pitch_name)] + acc

    return up_pitch, final_pitch_string


def apply_tempo_perform_features(xml_doc, xml_notes, features, start_time=0, predicted=False):
    beats = xml_doc.get_beat_positions()
    num_beats = len(beats)
    num_notes = len(xml_notes)
    tempos = []
    ornaments = []
    prev_vel = 64
    previous_position = None
    current_sec = start_time
    key_signatures = xml_doc.get_key_signatures()
    trill_accidentals = xml_doc.get_accidentals

    valid_notes = make_available_note_feature_list(xml_notes, features)
    previous_tempo = 0

    for i in range(num_beats - 1):
        beat = beats[i]
        feat = get_item_by_xml_position(valid_notes, beat)
        start_position = feat.xml_position
        if start_position == previous_position:
            continue

        if predicted:
            qpm_saved = 10 ** feat.qpm
            num_added = 1
            next_beat = beats[i+1]
            start_index = feat.index

            for j in range(1, 20):
                if start_index-j < 0:
                    break
                previous_note = xml_notes[start_index-j]
                previous_pos = previous_note.note_duration.xml_position
                if previous_pos == start_position:
                    qpm_saved += 10 ** features[start_index-j]['qpm']
                    num_added += 1
                else:
                    break

            for j in range(1, 40):
                if start_index + j >= num_notes:
                    break
                next_note = xml_notes[start_index+j]
                next_position = next_note.note_duration.xml_position
                if next_position < next_beat:
                    qpm_saved += 10 ** features[start_index+j]['qpm']
                    num_added += 1
                else:
                    break

            qpm = qpm_saved / num_added

        else:
            qpm = 10 ** feat.qpm

        divisions = feat.divisions

        if previous_tempo != 0:
            passed_second = (start_position - previous_position) / \
                divisions / previous_tempo * 60
        else:
            passed_second = 0
        current_sec += passed_second
        tempo = Tempo(start_position, qpm,
                      time_position=current_sec, end_xml=0, end_time=0)
        if len(tempos) > 0:
            tempos[-1].end_time = current_sec
            tempos[-1].end_xml = start_position

        tempos.append(tempo)

        previous_position = start_position
        previous_tempo = qpm
    
    def cal_time_position_with_tempo(note, xml_dev, tempos):
        corresp_tempo = get_item_by_xml_position(tempos, note)
        previous_sec = corresp_tempo.time_position
        passed_duration = note.note_duration.xml_position + \
            xml_dev - corresp_tempo.xml_position
        passed_second = passed_duration / \
            note.state_fixed.divisions / corresp_tempo.qpm * 60

        return previous_sec + passed_second

    for note, feat in zip(xml_notes, features):
        if not feat['xml_deviation'] == None:
            xml_deviation = feat['xml_deviation'] * note.state_fixed.divisions
        else:
            xml_deviation = 0

        note.note_duration.time_position = cal_time_position_with_tempo(
            note, xml_deviation, tempos)

        end_note = copy.copy(note)
        end_note.note_duration = copy.copy(note.note_duration)
        end_note.note_duration.xml_position = note.note_duration.xml_position + \
            note.note_duration.duration

        end_position = cal_time_position_with_tempo(end_note, 0, tempos)
        if note.note_notations.is_trill:
            note, _ = apply_feat_to_a_note(note, feat, prev_vel)

            trill_density = feat['num_trills']
            last_velocity = feat['trill_last_note_velocity'] * note.velocity
            first_note_ratio = feat['trill_first_note_ratio']
            last_note_ratio = feat['trill_last_note_ratio']
            up_trill = feat['up_trill']
            total_second = end_position - note.note_duration.time_position
            num_trills = int(trill_density * total_second)
            first_velocity = note.velocity

            key = get_item_by_xml_position(key_signatures, note)
            key = key.key
            final_key = None
            for acc in trill_accidentals:
                if acc.xml_position == note.note_duration.xml_position:
                    if acc.type['content'] == '#':
                        final_key = 7
                    elif acc.type['content'] == '♭':
                        final_key = -7
                    elif acc.type['content'] == '♮':
                        final_key = 0
            measure_accidentals = get_measure_accidentals(xml_notes, i)
            trill_pitch = note.pitch
            up_pitch, up_pitch_string = cal_up_trill_pitch(
                note.pitch, key, final_key, measure_accidentals)

            if up_trill:
                up = True
            else:
                up = False

            if num_trills > 2:
                mean_second = total_second / num_trills
                normal_second = (total_second - mean_second *
                                 (first_note_ratio + last_note_ratio)) / (num_trills - 2)
                prev_end = note.note_duration.time_position
                for j in range(num_trills):
                    if up:
                        pitch = (up_pitch_string, up_pitch)
                        up = False
                    else:
                        pitch = trill_pitch
                        up = True
                    if j == 0:
                        note.pitch = pitch
                        note.note_duration.seconds = mean_second * first_note_ratio
                        prev_end += mean_second * first_note_ratio
                    else:
                        new_note = copy.copy(note)
                        new_note.pedals = None
                        new_note.pitch = copy.copy(note.pitch)
                        new_note.pitch = pitch
                        new_note.note_duration = copy.copy(note.note_duration)
                        new_note.note_duration.time_position = prev_end
                        if j == num_trills - 1:
                            new_note.note_duration.seconds = mean_second * last_note_ratio
                        else:
                            new_note.note_duration.seconds = normal_second
                        new_note.velocity = copy.copy(note.velocity)
                        new_note.velocity = first_velocity + \
                            (last_velocity - first_velocity) * (j / num_trills)
                        prev_end += new_note.note_duration.seconds
                        ornaments.append(new_note)
            elif num_trills == 2:
                mean_second = total_second / num_trills
                prev_end = note.note_duration.time_position
                for j in range(2):
                    if up:
                        pitch = (up_pitch_string, up_pitch)
                        up = False
                    else:
                        pitch = trill_pitch
                        up = True
                    if j == 0:
                        note.pitch = pitch
                        note.note_duration.seconds = mean_second * first_note_ratio
                        prev_end += mean_second * first_note_ratio
                    else:
                        new_note = copy.copy(note)
                        new_note.pedals = None
                        new_note.pitch = copy.copy(note.pitch)
                        new_note.pitch = pitch
                        new_note.note_duration = copy.copy(note.note_duration)
                        new_note.note_duration.time_position = prev_end
                        new_note.note_duration.seconds = mean_second * last_note_ratio
                        new_note.velocity = copy.copy(note.velocity)
                        new_note.velocity = last_velocity
                        prev_end += mean_second * last_note_ratio
                        ornaments.append(new_note)
            else:
                note.note_duration.seconds = total_second

        else:
            note.note_duration.seconds = end_position - note.note_duration.time_position

        note, prev_vel = apply_feat_to_a_note(note, feat, prev_vel)

    for i in range(num_notes):
        note = xml_notes[i]
        feat = features[i]

        if note.note_duration.is_grace_note and note.note_duration.duration == 0:
            for j in range(i+1, num_notes):
                next_note = xml_notes[j]
                if not next_note.note_duration.duration == 0 \
                        and next_note.note_duration.xml_position == note.note_duration.xml_position \
                        and next_note.voice == note.voice:
                    next_second = next_note.note_duration.time_position
                    note.note_duration.seconds = (
                        next_second - note.note_duration.time_position) / note.note_duration.num_grace
                    break

    xml_notes = xml_notes + ornaments
    xml_notes.sort(key=lambda x: (x.note_duration.xml_position,
                                  x.note_duration.time_position, -x.pitch[1]))
    return xml_notes


def xml_notes_to_midi(xml_notes):
    midi_notes = []
    for note in xml_notes:
        if note.is_overlapped:  # ignore overlapped notes.
            continue

        pitch = note.pitch[1]
        start = note.note_duration.time_position
        end = start + note.note_duration.seconds
        if note.note_duration.seconds < 0.005:
            end = start + 0.005
        elif note.note_duration.seconds > 10:
            end = start + 10
        velocity = int(min(max(note.velocity, 0), 127))
        midi_note = pretty_midi.Note(
            velocity=velocity, pitch=pitch, start=start, end=end)

        midi_notes.append(midi_note)

    midi_pedals = pedal_cleaning.predicted_pedals_to_midi_pedals(xml_notes)

    return midi_notes, midi_pedals
