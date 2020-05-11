from pyScoreParser.constants import VNET_INPUT_KEYS, VNET_OUTPUT_KEYS

class ModelConstants():
    def __init__(self):
        self.DROP_OUT = 0.2
        self.is_trill_index_concated = -22 # when training trill model.. maybe wrong value

        # input feature length
        self.SCORE_INPUT = None
        self.TEMPO_IDX = None
        self.TEMPO_PARAM_LEN = None
        self.QPM_PRIMO_IDX = None
        self.TEMPO_PRIMO_IDX = None
        self.TEMPO_PRIMO_LEN = None

        # output feature length and numbers
        self.NUM_PRIME_PARAM = 11
        self.OUTPUT_TEMPO_PARAM_LEN = None
        self.VEL_PARAM_IDX = None
        self.DEV_PARAM_IDX = None
        self.ARTICULATION_PARAM_IDX = None
        self.PEDAL_PARAM_IDX = None
        self.PEDAL_PARAM_LEN = None
        self.TRILL_PARAM_IDX = None
        self.NUM_TRILL_PARAM = None
        self.OUTPUT_TEMPO_INDEX = None
        self.MEAS_TEMPO_IDX = None
        self.BEAT_TEMPO_IDX = None

        

def initialize_model_constants(index_dict):
    model_constants = ModelConstants()
    input_dict = index_dict['input_index_dict']
    output_dict = index_dict['output_index_dict']
    
    model_constants.SCORE_INPUT = input_dict['total_length']
    model_constants.TEMPO_IDX = input_dict['tempo']['index']
    model_constants.TEMPO_PARAM_LEN = input_dict['tempo']['len']
    model_constants.QPM_PRIMO_IDX = input_dict['qpm_primo']['index']
    model_constants.TEMPO_PRIMO_IDX = input_dict['tempo_primo']['index']
    model_constants.TEMPO_PRIMO_LEN = input_dict['tempo_primo']['len']

    model_constants.OUTPUT_TEMPO_PARAM_LEN = output_dict['beat_tempo']['len']
    model_constants.VEL_PARAM_IDX = output_dict['velocity']['index']
    model_constants.DEV_PARAM_IDX = output_dict['onset_deviation']['index']
    model_constants.ARTICULATION_PARAM_IDX = output_dict['articulation']['index']
    model_constants.PEDAL_PARAM_IDX = output_dict['pedal_refresh_time']['index']
    model_constants.PEDAL_PARAM_LEN = output_dict['pedal_refresh_time']['len'] \
                                      + output_dict['pedal_cut_time']['len'] \
                                      + output_dict['pedal_at_start']['len'] \
                                      + output_dict['pedal_at_end']['len'] \
                                      + output_dict['soft_pedal']['len'] \
                                      + output_dict['pedal_refresh']['len'] \
                                      + output_dict['pedal_cut']['len']
    model_constants.TRILL_PARAM_IDX = output_dict['num_trills']['index']
    model_constants.NUM_TRILL_PARAM = output_dict['num_trills']['len'] \
                                      + output_dict['trill_last_note_velocity']['len'] \
                                      + output_dict['trill_first_note_ratio']['len'] \
                                      + output_dict['trill_last_note_ratio']['len'] \
                                      + output_dict['up_trill']['len']
    model_constants.OUTPUT_TEMPO_INDEX = output_dict['beat_tempo']['index'][0]
    model_constants.MEAS_TEMPO_IDX = output_dict['measure_tempo']['index']
    model_constants.BEAT_TEMPO_IDX = output_dict['beat_tempo']['index'][1]


    return model_constants
'''
#SCORE_INPUT = 78 #score information only
SCORE_INPUT = 83 # with emotion information
DROP_OUT = 0.2
TOTAL_OUTPUT = 16

NUM_PRIME_PARAM = 11
NUM_TEMPO_PARAM = 1  # utils.py -> cal_tempo_loss_in_beat // output feature len
VEL_PARAM_IDX = 1  # utils.py -> cal_loss_by_output_type // output feature len
DEV_PARAM_IDX = 2  # utils.py -> cal_loss_by_output_type // output feature len
PEDAL_PARAM_IDX = 3  # utils.py -> cal_loss_by_output_type // output feature len
#num_second_param = 0
NUM_TRILL_PARAM = 5  # utils.py -> cal_loss_by_output_type // output feature len
#num_voice_feed_param = 0 # velocity, onset deviation
#num_tempo_info = 0
#num_dynamic_info = 0 # distance from marking, dynamics vector 4, mean_piano, forte marking and velocity = 4
#is_trill_index_score = -11
#is_trill_index_concated = -11 - (NUM_PRIME_PARAM + num_second_param) # when training trill model
is_trill_index_concated = -22 # only when trill model

QPM_INDEX = 0
# VOICE_IDX = 11
TEMPO_IDX = 26
QPM_PRIMO_IDX = VNET_INPUT_KEYS.index('qpm_primo')
TEMPO_PRIMO_IDX = -2

MEAS_TEMPO_IDX = VNET_OUTPUT_KEYS.index('measure_tempo')
BEAT_TEMPO_IDX = VNET_OUTPUT_KEYS[1:].index('beat_tempo')+1
'''
'''
# test_piece_list = [('schumann', 'Schumann'),
#                 ('mozart545-1', 'Mozart'),
#                 ('chopin_nocturne', 'Chopin'),
#                 ('chopin_fantasie_impromptu', 'Chopin'),
#                 ('cho_waltz_69_2', 'Chopin'),
#                 ('lacampanella', 'Liszt'),
#                 ('bohemian_rhapsody', 'Liszt')
#                 ]


test_piece_list = [
                ('bps_5_1', 'Beethoven'),
                ('bps_27_1', 'Beethoven'),
                ('bps_17_1', 'Beethoven'),
                ('bps_7_2', 'Beethoven'),
                ('bps_31_2', 'Beethoven'),
                ('bwv_858_prelude', 'Bach'),
                ('bwv_891_prelude', 'Bach'),
                ('bwv_858_fugue', 'Bach'),
                ('bwv_891_fugue', 'Bach'),
                ('schubert_piano_sonata_664-1', 'Schubert'),
                ('haydn_keyboard_31_1', 'Haydn'),
                ('haydn_keyboard_49_1', 'Haydn'),
                ('schubert_impromptu', 'Schubert'),
                ('mozart545-1', 'Mozart'),
                # ('mozart_symphony', 'Mozart'),
                ('liszt_pag', 'Liszt'),
                ('liszt_5', 'Liszt'),
                ('liszt_9', 'Liszt'),
                ('chopin_etude_10_2', 'Chopin'),
                ('chopin_etude_10_12', 'Chopin'),
                ('chopin_etude_25_12', 'Chopin'),
                ('cho_waltz_69_2', 'Chopin'),
                ('chopin_nocturne', 'Chopin'),
                ('cho_noc_9_1', 'Chopin'),
                ('chopin_prelude_1', 'Chopin'),
                ('chopin_prelude_4', 'Chopin'),
                ('chopin_prelude_5', 'Chopin'),
                ('chopin_prelude_6', 'Chopin'),
                ('chopin_prelude_8', 'Chopin'),
                ('chopin_prelude_15', 'Chopin'),
                ('kiss_the_rain', 'Chopin'),
                ('bohemian_rhapsody', 'Liszt'),
                ('chopin_fantasie_impromptu', 'Chopin', 180),
                ('schumann', 'Schumann'),
                ('chopin_barcarolle', 'Chopin'),
                ('chopin_scherzo', 'Chopin'),
                   ]

# test_piece_list = [
#     # ('dmc_glass', 'Liszt', 120),
#                    ('dmc_prokofiev', 'Prokofiev', 160),
#                    # ('dmc_shostakovich', 'Bach', 80),
#                    ('dmc_sho_fugue', 'Liszt', 160),
#                     # ('dmc_messiaen', 'Debussy', 72)
#                    ]

emotion_folder_path = 'test_pieces/emotionNet/'
emotion_key_list = ['OR', 'Anger', 'Enjoy', 'Relax', 'Sad']
emotion_data_path  = [('Bach_Prelude_1', 'Bach', 1),
                      ('Clementi_op.36-1_mov3', 'Haydn', 3),
                      ('Kuhlau_op.20-1_mov1', 'Haydn', 2),
                      ]
'''
