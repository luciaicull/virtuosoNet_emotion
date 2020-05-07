import os
import ntpath
import shutil
import subprocess
from pathlib import Path

from .constants import ALIGN_DIR
from .midi_utils import midi_utils
from . import xml_utils
from .musicxml_parser import MusicXMLDocument

class Alignment:
    def __init__(self, xml_path, score_midi_path, perform_lists, direct_matching=False):
        self.xml_path = xml_path
        if not direct_matching:
            self.score_midi_path = self._check_score_midi(score_midi_path)
        self.perform_lists = perform_lists

    def _check_score_midi(self, score_midi_path):
        if not os.path.isfile(score_midi_path):
            self._make_score_midi(score_midi_path)
        
        return score_midi_path

    def _make_score_midi(self, midi_file_name):
        self.xml_obj = MusicXMLDocument(self.xml_path)
        self.xml_notes = self._get_direction_encoded_notes()

        midi_notes, midi_pedals = xml_utils.xml_notes_to_midi(self.xml_notes)
        xml_utils.save_midi_notes_as_piano_midi(
            midi_notes, [], midi_file_name, bool_pedal=True)
    
    def _get_direction_encoded_notes(self):
        notes, rests = self.xml_obj.get_notes()
        directions = self.xml_obj.get_directions()
        time_signatures = self.xml_obj.get_time_signatures()

        xml_notes = xml_utils.apply_directions_to_notes(
            notes, directions, time_signatures)

        return xml_notes

    def check_perf_align(self, direct_matching=False):
        if direct_matching:
            return self._xml_midi_align()
        else:
            return self._midi_midi_align()
    
    def _xml_midi_align(self):
        print('processing score: {}'.format(self.xml_path.split('/')[-1]))
        aligned_perf = []
        for perf in self.perform_lists:
            print('processing performance: {}'.format(perf.split('/')[-1]))
            align_file_name = os.path.splitext(perf)[0] + '_match.txt'
            if os.path.isfile(align_file_name):
                aligned_perf.append(perf)
                continue
            self.align_score_xml_and_perf_midi_with_nakamura(self.xml_path, perf)
            if os.path.isfile(align_file_name):
                aligned_perf.append(perf)
        
        return aligned_perf

    def _midi_midi_align(self):
        print('processing score: {}'.format(self.score_midi_path))
        aligned_perf = []
        for perf in self.perform_lists:
            print('processing performance: {}'.format(perf))
            align_file_name = os.path.splitext(perf)[0] + '_infer_corresp.txt'
            if os.path.isfile(align_file_name):
                aligned_perf.append(perf)
                continue
            self.align_score_and_perf_with_nakamura(
                os.path.abspath(perf), self.score_midi_path)
            # check once again whether the alignment was successful
            if os.path.isfile(align_file_name):
                aligned_perf.append(perf)

        return aligned_perf

    def align_score_and_perf_with_nakamura(self, midi_file_path, score_midi_path):
        file_folder, file_name = ntpath.split(midi_file_path)
        perform_midi = midi_file_path

        shutil.copy(perform_midi, os.path.join(ALIGN_DIR, 'infer.mid'))
        shutil.copy(score_midi_path, os.path.join(ALIGN_DIR, 'score.mid'))
        current_dir = os.getcwd()
        try:
            os.chdir(ALIGN_DIR)
            subprocess.check_call(
                ["sudo", "sh", "MIDIToMIDIAlign.sh", "score", "infer"])
        except:
            print('Error to process {}'.format(midi_file_path))
            print('Trying to fix MIDI file {}'.format(midi_file_path))
            os.chdir(current_dir)
            shutil.copy(midi_file_path, midi_file_path+'old')
            midi_utils.to_midi_zero(
                midi_file_path, save_midi=True, save_name=midi_file_path)
            shutil.copy(midi_file_path, os.path.join(ALIGN_DIR, 'infer.mid'))
            try:
                os.chdir(ALIGN_DIR)
                subprocess.check_call(
                    ["sudo", "sh", "MIDIToMIDIAlign.sh", "score", "infer"])
            except:
                align_success = False
                print('Fail to process {}'.format(midi_file_path))
                os.chdir(current_dir)
            else:
                align_success = True
                print('Success to process {}'.format(midi_file_path))
        else:
            align_success = True
            print('Success to process {}'.format(midi_file_path))

        if align_success:
            shutil.move('infer_corresp.txt', midi_file_path[:-len('.mid')]+'_infer_corresp.txt')
            shutil.move('infer_match.txt',
                        midi_file_path[:-len('.mid')] + '_infer_match.txt')
            shutil.move('infer_spr.txt', midi_file_path[:-len('.mid')]+'_infer_spr.txt')
            shutil.move('score_spr.txt', os.path.join(
                ALIGN_DIR, '_score_spr.txt'))
            shutil.move('score_fmt3x.txt',
                        midi_file_path[:-len('.mid')] + '_score_fmt3x.txt')
            os.chdir(current_dir)

    def align_score_xml_and_perf_midi_with_nakamura(self, xml_path, midi_path):
        shutil.copy(str(midi_path), os.path.join(ALIGN_DIR, 'infer.mid'))
        shutil.copy(str(xml_path), os.path.join(ALIGN_DIR, 'ref_score.xml'))
        current_dir = os.getcwd()
        os.chdir(ALIGN_DIR)
        subprocess.check_call(
            ["sudo", "sh", "MusicXMLToMIDIAlign.sh", "ref_score", "infer"])
        print('Success to process {}'.format(midi_path))

        midi_path = Path(midi_path)
        xml_path = Path(xml_path)
        result_match_path = midi_path.with_name(midi_path.name[:-len('.mid')] + '_match.txt')
        result_fmt3_path = xml_path.with_name(xml_path.name[:-len('.xml')] + '_fmt3x.txt')
        shutil.move('infer_match.txt', str(result_match_path))
        shutil.move('ref_score_fmt3x.txt', str(result_fmt3_path))
        os.chdir(current_dir)
