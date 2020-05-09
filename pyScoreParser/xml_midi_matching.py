import csv
import copy
import pretty_midi


def match_xml_to_midi(xml_notes, midi_notes):
    candidates_list = []
    match_list = []
    # midi_positions = [note.start for note in midi_notes]
    # def find_candidate_list(xml_note, midi_notes, midi_positions):
    #     num_midi = len(midi_notes)
    #     temp_list =[]
    #     match_threshold = 0.1
    #     if note.is_rest:
    #         return([])
    #     note_start = xml_note.note_duration.time_position
    #     if note.note_duration.preceded_by_grace_note:
    #         note_start += 0.5
    #         match_threshold = 0.6
    #     elif note.note_notations.is_arpeggiate:
    #         note_start += 0.3
    #         match_threshold = 0.4
    #
    #     nearby_index = binaryIndex(midi_positions, note_start)
    #
    #     for i in range(-10, 10):
    #         index = nearby_index+i
    #         if index < 0:
    #             index = 0
    #         elif index >= num_midi:
    #             break
    #         midi_note = midi_notes[index]
    #         if midi_note.pitch == note.pitch[1] or abs(midi_note.start - note_start) < match_threshold:
    #             temp_list.append({'index': index, 'midi_note':midi_note})
    #
    #         if midi_note.start > note_start + match_threshold:
    #             break
    #
    #     return temp_list

    # for each note in xml, make candidates of the matching midi note
    for note in xml_notes:
        match_threshold = 0.05
        if note.is_rest:
            candidates_list.append([])
            continue
        note_start = note.note_duration.time_position
        if note.note_duration.preceded_by_grace_note:
            note_start += 0.5
            match_threshold = 0.6
        elif note.note_notations.is_arpeggiate:
            match_threshold = 0.5
        # check grace note and adjust time_position
        note_pitch = note.pitch[1]
        temp_list = [{'index': index, 'midi_note': midi_note} for index, midi_note in enumerate(midi_notes)
                     if abs(midi_note.start - note_start) < match_threshold and midi_note.pitch == note_pitch]
        # temp_list = find_candidate_list(note, midi_notes, midi_positions)
        candidates_list.append(temp_list)


    for candidates in candidates_list:
        if len(candidates) ==1:
            matched_index = candidates[0]['index']
            match_list.append(matched_index)
        elif len(candidates) > 1:
            added = False
            for cand in candidates:
                if cand['index'] not in match_list:
                    match_list.append(cand['index'])
                    added = True
                    break
            if not added:
                match_list.append([])
        else:
            match_list.append([])
    return match_list


def make_xml_midi_pair(xml_notes, midi_notes, match_list):
    pairs = []
    for i in range(len(match_list)):
        if not match_list[i] ==[]:
            temp_pair = {'xml': xml_notes[i], 'midi': midi_notes[match_list[i]]}
            pairs.append(temp_pair)
        else:
            pairs.append([])
    return pairs


def read_corresp(txtpath):
    file = open(txtpath, 'r')
    reader = csv.reader(file, dialect='excel', delimiter='\t')
    corresp_list = []
    for row in reader:
        if len(row) == 1:
            continue
        temp_dic = {'alignID': row[0], 'alignOntime': row[1], 'alignSitch': row[2], 'alignPitch': row[3], 'alignOnvel': row[4], 'refID':row[5], 'refOntime':row[6], 'refSitch':row[7], 'refPitch':row[8], 'refOnvel':row[9] }
        corresp_list.append(temp_dic)

    return corresp_list

# for new alignment - read _match.txt file and return lists
def read_match_file(match_path):
    f = open(match_path, 'r')
    reader = csv.reader(f, dialect='excel', delimiter='\t')
    match_txt = {'match_list': [], 'missing': []}
    for row in reader:
        if len(row) == 1:
            continue
        elif len(row) == 2:
            dic = {'scoreTime': int(row[0].split(' ')[-1]), 'xmlNoteID': row[1]}
            match_txt['missing'].append(dic)
        else:
            dic = {'midiID': int(row[0]), 'midiStartTime': float(row[1]), 'midiEndTime': float(row[2]), 'pitch': row[3], 'midiOnsetVel': int(row[4]), 'midiOffsetVel': int(
                row[5]), 'channel': int(row[6]), 'matchStatus': int(row[7]), 'scoreTime': int(row[8]), 'xmlNoteID': row[9], 'errorIndex': int(row[10]), 'skipIndex': row[11], 'used': False}
            match_txt['match_list'].append(dic)
    
    return match_txt['match_list'], match_txt['missing']


def make_direct_xml_midi_pair(xml_notes, midi_notes, match_list, missing_xml_list):
    """
    Main method for score xml - performance midi direct matching process

    Parameters
    -----------
    xml_notes : list of xml note object
    midi_notes : list of midi note object
    match_list : lines of match result in _match.txt, in dictionary(dict) format
    missing_xml_list : lines of missing result in _match.txt, in dictionary(dict) format

    Returns
    -----------
    pairs : list of match pair in dictionary format - {'xml': xml_index, 'midi': midi_index}

    """
    index_dict_list = []
    for xml_index, xml_note in enumerate(xml_notes):
        dic = match_xml_midi_directly(xml_index, xml_note, match_list, midi_notes)
        index_dict_list.append(dic)
        #print(dic)

    pairs = pair_transformation(xml_notes, midi_notes, index_dict_list)
    return pairs


def match_xml_midi_directly(xml_index, xml_note, match_list, midi_notes):
    """
    Match method for one xml note

    Parameters
    -----------
    xml_index : index of xml_note in xml_notes list
    xml_note : xml note object
    midi_notes : list of midi note object
    match_list : lines of match result in _match.txt, in dictionary(dict) format

    Returns
    -----------
    dic : result of match result in dictionary format

    """
    dic = {'match_index': [], 'xml_index': {'idx':xml_index, 'pitch':xml_note.pitch[0]}, 'midi_index': [
    ], 'is_trill': False, 'is_ornament': False, 'is_overlapped': xml_note.is_overlapped, 'overlap_xml_index': [], 'unmatched': False, 'fixed_trill_idx': []}

    find_corresp_match_and_midi(
        dic, match_list, midi_notes, xml_note.note_duration.xml_position, xml_note.pitch[0])

    find_trill_midis(dic, match_list, midi_notes)

    find_ornament_midis(
        dic, xml_note.note_duration.xml_position, match_list, midi_notes)

    if len(dic['midi_index']) == 0:
        dic['unmatched'] = True
    else:
        for idx in dic['match_index']:
            match_list[idx]['used'] = True

    return dic


def find_corresp_match_and_midi(dic, match_list, midi_notes, score_time, score_pitch):
    """
    find corresponding match dictionary and midi note for one xml note

    Parameters
    -----------
    dic : result of match result in dictionary format
    match_list : lines of match result in _match.txt, in dictionary(dict) format
    midi_notes : list of midi note object
    score_time : score time of xml note
    score_pitch : pitch of xml note (in string)     
    """
    for match_index, match in enumerate(match_list):
        if match['xmlNoteID'] != '*':
            if score_time == match['scoreTime'] and score_pitch == match['pitch']:
                dic['match_index'].append(match_index)
                midi_idx = find_midi_note_index(
                    midi_notes, match['midiStartTime'], match['midiEndTime'], match['pitch'])
                if midi_idx != -1:
                    dic['midi_index'].append(midi_idx)
                    match['used'] = True


def find_midi_note_index(midi_notes, start, end, pitch, ornament=False):
    """
    find corresponding midi note index for one xml note

    Parameters
    -----------
    midi_notes : list of midi note object
    start: midi start time in match_list
    end : midi end time in match_list
    pitch : midi pitch in match_list (in string)
    ornament : whether it's ornament

    Returns
    -----------
    dictionary of midi index and pitch
    """
    pitch = check_pitch(pitch)
    if not ornament:
        for i, note in enumerate(midi_notes):
            if (abs(note.start - start) < 0.001) and (abs(note.end - end) < 0.001) and (note.pitch == pretty_midi.note_name_to_number(pitch)):
                return {'idx': i, 'pitch': pretty_midi.note_number_to_name(note.pitch)}
    else:
        for i, note in enumerate(midi_notes):
            if (abs(note.start - start) < 0.001) and (abs(note.end - end) < 0.001) and (abs(note.pitch - pretty_midi.note_name_to_number(pitch)) <= 2):
                return {'idx': i, 'pitch': pretty_midi.note_number_to_name(note.pitch)}
    return -1


def check_pitch(pitch):
    """
    check string pitch format and fix it

    Parameters
    -----------
    pitch : midi string pitch

    Returns
    -----------
    pitch : midi string pitch
    """
    if len(pitch) == 4:
        base_pitch_num = pretty_midi.note_name_to_number(pitch[0]+pitch[-1])
        if pitch[1:3] == 'bb':
            pitch = pretty_midi.note_number_to_name(base_pitch_num-2)
        if pitch[1:3] == '##':
            pitch = pretty_midi.note_number_to_name(base_pitch_num+2)
    return pitch


def find_trill_midis(dic, match_list, midi_notes):
    """
    find possible trill midi note indices

    Parameters
    -----------
    dic : result of match result in dictionary format
    match_list : lines of match result in _match.txt, in dictionary(dict) format
    midi_notes : list of midi note object

    """
    if len(dic['match_index']) > 1:
        # 미디 여러 개 - xml 하나라서 match가 여러 개 뜰 경우
        dic['is_trill'] = True

        start_idx = dic['match_index'][0]
        end_idx = dic['match_index'][-1]
        match_id = match_list[start_idx]['xmlNoteID']
        pitch = match_list[start_idx]['pitch']

        new_match_idx = []

        # find trill
        trill_pitch = None
        for idx in range(start_idx, end_idx + 1):
            if idx in dic['match_index']:
                continue
            else:
                if (match_list[idx]['xmlNoteID'] == match_id) or (match_list[idx]['errorIndex'] == 3):
                    midi_idx = find_midi_note_index(
                        midi_notes, match_list[idx]['midiStartTime'], match_list[idx]['midiEndTime'], match_list[idx]['pitch'])
                    if midi_idx != -1:
                        dic['midi_index'].append(midi_idx)
                        new_match_idx.append(idx)
                        trill_pitch = match_list[idx]['pitch']
                        if match_list[idx]['xmlNoteID'] != match_id:
                            dic['fixed_trill_idx'].append(midi_idx)
                        match_list[idx]['used'] = True

        # find one prev trill
        prev_idx = start_idx - 1
        prev = match_list[prev_idx]
        if prev['pitch'] == trill_pitch:
            if (prev['xmlNoteID'] == match_id) or (prev['errorIndex'] == 3):
                midi_idx = find_midi_note_index(
                    midi_notes, prev['midiStartTime'], prev['midiEndTime'], prev['pitch'])
                if midi_idx != -1:
                    dic['midi_index'].append(midi_idx)
                    new_match_idx.append(prev_idx)
                    if prev['xmlNoteID'] != match_id:
                        dic['fixed_trill_idx'].append(midi_idx)
                    prev['used'] = True

        dic['match_index'] += new_match_idx
        dic['match_index'].sort()
        prev_midi_index = dic['midi_index']
        dic['midi_index'] = sorted(
            prev_midi_index, key=lambda prev_midi_index: prev_midi_index['idx'])


def find_ornament_midis(dic, score_time, match_list, midi_notes):
    """
    find possible ornament midi note indices

    Parameters
    -----------
    dic : result of match result in dictionary format
    score_time : score time of xml note
    match_list : lines of match result in _match.txt, in dictionary(dict) format
    midi_notes : list of midi note object
    
    """
    if len(dic['match_index']) > 0:
        match = match_list[dic['match_index'][0]]
        cand_match_idx = [idx for idx, match in enumerate(
            match_list) if match['scoreTime'] == score_time]
        new_match_idx = []
        for cand in cand_match_idx:
            cand_match = match_list[cand]
            if not cand_match['used']:
                if (cand_match['xmlNoteID'] == match['xmlNoteID']):
                    midi_idx = find_midi_note_index(
                        midi_notes, cand_match['midiStartTime'], cand_match['midiEndTime'], match['pitch'], ornament=True)
                    if midi_idx != -1:
                        dic['midi_index'].append(midi_idx)
                        new_match_idx.append(cand)
                        if cand_match['xmlNoteID'] != match['xmlNoteID']:
                            dic['fixed_trill_idx'].append(midi_idx)
                        cand_match['used'] = True
                        dic['is_ornament'] = True
        dic['match_index'] += new_match_idx
        new_match_idx = []
        if len(dic['match_index']) >= 2:
            for cand in cand_match_idx:
                cand_match = match_list[cand]
                if not cand_match['used']:
                    if (cand_match['errorIndex'] == 3):
                        midi_idx = find_midi_note_index(
                            midi_notes, cand_match['midiStartTime'], cand_match['midiEndTime'], match['pitch'], ornament=True)
                        if midi_idx != -1:
                            dic['midi_index'].append(midi_idx)
                            new_match_idx.append(cand)
                            if cand_match['xmlNoteID'] != match['xmlNoteID']:
                                dic['fixed_trill_idx'].append(midi_idx)
                            cand_match['used'] = True
                            dic['is_ornament'] = True

        dic['match_index'] += new_match_idx
        dic['match_index'].sort()
        prev_midi_index = dic['midi_index']
        dic['midi_index'] = sorted(
            prev_midi_index, key=lambda prev_midi_index: prev_midi_index['idx'])



def pair_transformation(xml_notes, midi_notes, index_dict_list):
    """
    Transform pair format from index_dict_list to original pair

    Parameters
    -----------
    xml_notes : list of xml note object
    midi_notes : list of midi note object
    index_dict_list 
            : list of dictionary
            {'match_index': [], 'xml_index': {xml_index, xml_note.pitch[0]}, 'midi_index': [], 
             'is_trill': False, 'is_ornament': False, 'is_overlapped': xml_note.is_overlapped, 
             'overlap_xml_index': [], 'unmatched': False, 'fixed_trill_idx': []}
    
    Returns
    -----------
    pairs : list of dictionary
            {'xml': xml_notes[i], 'midi': midi_notes[match_list[i]]}
    """
    pairs = []
    for dic in index_dict_list:
        xml_idx = dic['xml_index']['idx']
        midi_idx_list = dic['midi_index']
        if dic['unmatched']:
            pair = []
        else:
            midi_idx = midi_idx_list[0]['idx']
            pair = {'xml': xml_notes[xml_idx], 'midi': midi_notes[midi_idx]}
        
        pairs.append(pair)
    
    return pairs


def match_score_pair2perform(pairs, perform_midi, corresp_list):
    match_list = []
    for pair in pairs:
        if pair == []:
            match_list.append([])
            continue
        ref_midi = pair['midi']
        index_in_corresp = find_by_key(corresp_list, 'refOntime', ref_midi.start, 'refPitch', ref_midi.pitch)
        if index_in_corresp == -1:
            match_list.append([])
        else:
            corresp_pair = corresp_list[index_in_corresp]
            index_in_perform_midi = find_by_attr(perform_midi, float(corresp_pair['alignOntime']),  int(corresp_pair['alignPitch']))
            if index_in_perform_midi == []:
                print('perf midi missing: ', corresp_pair, ref_midi.start, ref_midi.pitch)

            match_list.append(index_in_perform_midi)
    return match_list


def match_xml_midi_perform(xml_notes, midi_notes, perform_notes, corresp):
    # xml_notes = apply_tied_notes(xml_notes)
    match_list = match_xml_to_midi(xml_notes, midi_notes)
    score_pairs = make_xml_midi_pair(xml_notes, midi_notes, match_list)
    xml_perform_match = match_score_pair2perform(score_pairs, perform_notes, corresp)
    perform_pairs = make_xml_midi_pair(xml_notes, perform_notes, xml_perform_match)

    return score_pairs, perform_pairs


def find_by_key(alist, key1, value1, key2, value2):
    for i, dic in enumerate(alist):
        if abs(float(dic[key1]) - value1) < 0.02 and int(dic[key2]) == value2:
            return i
    return -1


def find_by_attr(alist, value1, value2):
    for i, obj in enumerate(alist):
        if abs(obj.start - value1) < 0.02 and obj.pitch == value2:
            return i
    return []


def make_available_xml_midi_positions(pairs):
    # global NUM_EXCLUDED_NOTES
    class PositionPair:
        def __init__(self, xml_pos, time, pitch, index, divisions):
            self.xml_position = xml_pos
            self.time_position = time
            self.pitch = pitch
            self.index = index
            self.divisions = divisions
            self.is_arpeggiate = False

    available_pairs = []
    num_pairs = len(pairs)
    for i in range(num_pairs):
        pair = pairs[i]
        if not pair == []:
            xml_note = pair['xml']
            midi_note = pair['midi']
            xml_pos = xml_note.note_duration.xml_position
            time = midi_note.start
            divisions = xml_note.state_fixed.divisions
            if not xml_note.note_duration.is_grace_note:
                pos_pair = {'xml_position': xml_pos, 'time_position': time, 'pitch': xml_note.pitch[1], 'index':i, 'divisions':divisions}
                # pos_pair = PositionPair(xml_pos, time, xml_note.pitch[1], i, divisions)
                if xml_note.note_notations.is_arpeggiate:
                    pos_pair['is_arpeggiate'] = True
                else:
                    pos_pair['is_arpeggiate'] = False
                available_pairs.append(pos_pair)

    # available_pairs = save_lowest_note_on_same_position(available_pairs)
    available_pairs, mismatched_indexes = make_average_onset_cleaned_pair(available_pairs)
    print('Number of mismatched notes: ', len(mismatched_indexes))
    # NUM_EXCLUDED_NOTES += len(mismatched_indexes)
    for index in mismatched_indexes:
        pairs[index] = []

    return pairs, available_pairs


def make_average_onset_cleaned_pair(position_pairs, maximum_qpm=600):
    length = len(position_pairs)
    previous_position = -float("Inf")
    previous_time = -float("Inf")
    previous_index = 0
    # position_pairs.sort(key=lambda x: (x.xml_position, x.pitch))
    cleaned_list = list()
    notes_in_chord = list()
    mismatched_indexes = list()
    for i in range(length):
        pos_pair = position_pairs[i]
        current_position = pos_pair['xml_position']
        current_time = pos_pair['time_position']
        if current_position > previous_position >= 0:
            minimum_time_interval = (current_position - previous_position) / pos_pair['divisions'] / maximum_qpm * 60 + 0.001
        else:
            minimum_time_interval = 0
        if current_position > previous_position and current_time > previous_time + minimum_time_interval:
            if len(notes_in_chord) > 0:
                average_pos_pair = copy.copy(notes_in_chord[0])
                notes_in_chord_cleaned, average_pos_pair['time_position'] = get_average_onset_time(notes_in_chord)
                if len(cleaned_list) == 0 or average_pos_pair['time_position'] > cleaned_list[-1]['time_position'] + (
                        (average_pos_pair['xml_position'] - cleaned_list[-1]['xml_position']) /
                        average_pos_pair['divisions'] / maximum_qpm * 60 + 0.01):
                    cleaned_list.append(average_pos_pair)
                    for note in notes_in_chord:
                        if note not in notes_in_chord_cleaned:
                            # print('the note is far from other notes in the chord')
                            mismatched_indexes.append(note['index'])
                else:
                    # print('the onset is too close to the previous onset', average_pos_pair.xml_position, cleaned_list[-1].xml_position, average_pos_pair.time_position, cleaned_list[-1].time_position)
                    for note in notes_in_chord:
                        mismatched_indexes.append(note['index'])
            notes_in_chord = list()
            notes_in_chord.append(pos_pair)
            previous_position = current_position
            previous_time = current_time
            previous_index = i
        elif current_position == previous_position:
            notes_in_chord.append(pos_pair)
        else:
            # print('the note is too close to the previous note', current_position - previous_position, current_time - previous_time)
            # print(previous_position, current_position, previous_time, current_time)
            mismatched_indexes.append(position_pairs[previous_index]['index'])
            mismatched_indexes.append(pos_pair['index'])

    return cleaned_list, mismatched_indexes


def make_available_note_feature_list(notes, features, predicted):
    class PosTempoPair:
        def __init__(self, xml_pos, pitch, qpm, index, divisions, time_pos):
            self.xml_position = xml_pos
            self.qpm = qpm
            self.index = index
            self.divisions = divisions
            self.pitch = pitch
            self.time_position = time_pos
            self.is_arpeggiate = False

    if not predicted:
        available_notes = []
        num_features = len(features)
        for i in range(num_features):
            feature = features[i]
            if not feature.qpm == None:
                xml_note = notes[i]
                xml_pos = xml_note.note_duration.xml_position
                time_pos = feature.midi_start
                divisions = xml_note.state_fixed.divisions
                qpm = feature.qpm
                pos_pair = PosTempoPair(xml_pos, xml_note.pitch[1], qpm, i, divisions, time_pos)
                if xml_note.note_notations.is_arpeggiate:
                    pos_pair.is_arpeggiate = True
                available_notes.append(pos_pair)

    else:
        available_notes = []
        num_features = len(features)
        for i in range(num_features):
            feature = features[i]
            xml_note = notes[i]
            xml_pos = xml_note.note_duration.xml_position
            time_pos = xml_note.note_duration.time_position
            divisions = xml_note.state_fixed.divisions
            qpm = feature.qpm
            pos_pair = PosTempoPair(xml_pos, xml_note.pitch[1], qpm, i, divisions, time_pos)
            available_notes.append(pos_pair)

    # minimum_time_interval = 0.05
    # available_notes = save_lowest_note_on_same_position(available_notes, minimum_time_interval)
    available_notes, _ = make_average_onset_cleaned_pair(available_notes)
    return available_notes


def get_average_onset_time(notes_in_chord_saved, threshold=0.2):
    # notes_in_chord: list of PosTempoPair Dictionary, len > 0
    notes_in_chord = copy.copy(notes_in_chord_saved)
    average_onset_time = 0
    for pos_pair in notes_in_chord:
        average_onset_time += pos_pair['time_position']
        if pos_pair['is_arpeggiate']:
            threshold = 1
    average_onset_time /= len(notes_in_chord)

    # check whether there is mis-matched note
    deviations = list()
    for pos_pair in notes_in_chord:
        dev = abs(pos_pair['time_position'] - average_onset_time)
        deviations.append(dev)
    if max(deviations) > threshold:
        # print(deviations)
        if len(notes_in_chord) == 2:
            del notes_in_chord[0:2]
        else:
            index = deviations.index(max(deviations))
            del notes_in_chord[index]
            notes_in_chord, average_onset_time = get_average_onset_time(notes_in_chord, threshold)

    return notes_in_chord, average_onset_time
