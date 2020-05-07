# TODO: to args?
# path for test
ALIGN_DIR = '/home/yoojin/repositories/AlignmentTool_v190813'

# constants
DEFAULT_SCORE_FEATURES = ['midi_pitch', 'duration', 'beat_importance', 'measure_length', 'qpm_primo',
                          'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo',
                          'beat_position', 'xml_position', 'grace_order', 'preceded_by_grace_note',
                          'followed_by_fermata_rest', 'pitch', 'tempo', 'dynamic', 'time_sig_vec',
                          'slur_beam_vec',  'composer_vec', 'notation', 'tempo_primo', 'note_location']
# with qpm_primo
'''
DEFAULT_PERFORM_FEATURES = ['beat_tempo', 'velocity', 'onset_deviation', 'articulation', 'pedal_refresh_time',
                            'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                            'pedal_refresh', 'pedal_cut', 'qpm_primo', 'align_matched', 'articulation_loss_weight',
                            'beat_dynamics', 'measure_tempo', 'measure_dynamics', 'emotion']
'''
# without qpm_primo
DEFAULT_PERFORM_FEATURES = ['beat_tempo', 'velocity', 'onset_deviation', 'articulation', 'pedal_refresh_time',
                            'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                            'pedal_refresh', 'pedal_cut', 'align_matched', 'articulation_loss_weight',
                            'beat_dynamics', 'measure_tempo', 'measure_dynamics', 'emotion']

VNET_COPY_DATA_KEYS = ('note_location', 'align_matched',
                       'articulation_loss_weight')

# without emotion
'''
VNET_INPUT_KEYS = ('midi_pitch', 'duration', 'beat_importance', 'measure_length', 'qpm_primo',
                   'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo',
                   'beat_position', 'xml_position', 'grace_order', 'preceded_by_grace_note',
                   'followed_by_fermata_rest', 'pitch', 'tempo', 'dynamic', 'time_sig_vec',
                   'slur_beam_vec',  'composer_vec', 'notation', 'tempo_primo')
'''
# with emotion
VNET_INPUT_KEYS = ('midi_pitch', 'duration', 'beat_importance', 'measure_length', 'qpm_primo',
                   'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo',
                   'beat_position', 'xml_position', 'grace_order', 'preceded_by_grace_note',
                   'followed_by_fermata_rest', 'pitch', 'tempo', 'dynamic', 'time_sig_vec',
                   'slur_beam_vec',  'composer_vec', 'notation', 'emotion', 'tempo_primo')


VNET_OUTPUT_KEYS = ('beat_tempo', 'velocity', 'onset_deviation', 'articulation', 'pedal_refresh_time',
                    'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                    'pedal_refresh', 'pedal_cut', 'beat_tempo', 'beat_dynamics',
                    'measure_tempo', 'measure_dynamics', 'num_trills', 'trill_last_note_velocity', 
                    'trill_first_note_ratio', 'trill_last_note_ratio', 'up_trill')
