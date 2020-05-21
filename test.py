import random
import os
import torch
from pathlib import Path
import numpy as np

import xml_utils
import model_constants as cons
import model as modelzoo
import model_parameters as param
import graph
import utils
from pyScoreParser.feature_utils import NoteLocation
from pyScoreParser import xml_utils as pSP_xml_utils

def prediction_to_feature(prediction):
    output_features = []
    
    for pred in prediction:
        feature_dict = dict()
        feature_dict['qpm'] = pred[0]
        feature_dict['velocity'] = pred[1]
        feature_dict['xml_deviation'] = pred[2]
        feature_dict['articulation'] = pred[3]
        feature_dict['pedal_refresh_time'] = pred[4]
        feature_dict['pedal_cut_time'] = pred[5]
        feature_dict['pedal_at_start'] = pred[6]
        feature_dict['pedal_at_end'] = pred[7]
        feature_dict['soft_pedal'] = pred[8]
        feature_dict['pedal_refresh'] = pred[9]
        feature_dict['pedal_cut'] = pred[10]

        feature_dict['num_trills'] = pred[11]
        feature_dict['trill_last_note_velocity'] = (pred[12])
        feature_dict['trill_first_note_ratio'] = (pred[13])
        feature_dict['trill_last_note_ratio'] = (pred[14])
        feature_dict['up_trill'] = round(pred[15])

        output_features.append(feature_dict)

    return output_features
    

def add_note_location_to_features(features, note_locations):
    for feat, loc in zip(features, note_locations):
        feat['note_location'] = NoteLocation(loc.beat, loc.measure, None, None)
    
    return features

def scale_model_prediction_to_original(args, prediction, feature_stats):
    # TODO: for what?
    for key in feature_stats.keys():
        if feature_stats[key]['stds'] < 1e-4:
            feature_stats[key]['stds'] = 1

    prediction = np.squeeze(np.asarray(prediction.cpu()))
    num_notes = len(prediction)

    
    feature_index_map = ['beat_tempo', 'velocity', 'onset_deviation', 'articulation', 'pedal_refresh_time',
                         'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                         'pedal_refresh', 'pedal_cut', 'num_trills', 'trill_last_note_velocity',
                         'trill_first_note_ratio', 'trill_last_note_ratio']
    for i, key in enumerate(feature_index_map):
        prediction[:, i] *= feature_stats[key]['stds']
        prediction[:, i] += feature_stats[key]['mean']


    return prediction


def load_file_and_generate_performance(test_data, args, model, device, cons, composer, z, start_tempo, feature_stats, return_features=False, hier_model=None, trill_model=None):
    # mean velocity of piano and forte
    # vel_pair = (piano mean, forte mean). default: (50, 65)
    vel_pair = (int(args.velocity.split(',')[0]), int(
        args.velocity.split(',')[1]))

    test_xml_list = sorted(list(Path(test_data.path).glob(f'{test_data.split}/*.musicxml')))
    feature_path_list = test_data.files
    for feature_path in feature_path_list:
        feature_path = str(feature_path)
        xml_path = None
        for test_xml_path in test_xml_list:
            if test_xml_path.name[:-len('.musicxml')] in feature_path:
                xml_path = str(test_xml_path)
                break
       
        test_x, xml_notes, xml_doc, edges, note_locations = xml_utils.get_score_information(xml_path, feature_path, feature_stats,
                                                                                    start_tempo, composer,
                                                                                    vel_pair)
        batch_x = torch.Tensor(test_x)
        num_notes = len(test_x)
        input_y = torch.zeros(1, num_notes, model.config.output_size)

        if type(z) is dict:
            initial_z = z['z']
            qpm_change = z['qpm']
            z = z['key']
            batch_x[:, cons.QPM_PRIMO_IDX] = batch_x[:, cons.QPM_PRIMO_IDX] + qpm_change
        else:
            initial_z = 'zero'
        
        if model.config.is_dependent: # han_note_ar
            batch_x = batch_x.to(device).view(1, -1, hier_model.config.input_size)
            graphs = graph.edges_to_matrix(edges, batch_x.shape[1], model.config)
            model.config.is_teacher_force = False
            # TODO: how to give z for args?
            if type(initial_z) is list:
                hier_z = initial_z[0]
                final_z = initial_z[1]
            else:
                # TODO: what is this for?
                # hier_z = [z] * HIER_MODEL_PARAM.encoder.size
                hier_z = 'zero'
                final_z = initial_z
            hier_input_y = torch.zeros(1, num_notes, hier_model.config.output_size)
            hier_output, _ = utils.run_model_in_steps(batch_x, hier_input_y, args, graphs, note_locations, model=hier_model, device=device, initial_z=hier_z)

            if 'measure' in args.hierCode:
                hierarchy_numbers = [x.measure for x in note_locations]
            else:
                hierarchy_numbers = [x.section for x in note_locations]

            hier_output_spanned = hier_model.span_beat_to_note_num(hier_output, hierarchy_numbers, len(test_x), 0)
            combined_x = torch.cat((batch_x, hier_output_spanned), 2)
            prediction, _ = utils.run_model_in_steps(combined_x, input_y, args, graphs, note_locations, model=model, device=device, initial_z=final_z)
        else:
            if type(initial_z) is list:
                initial_z = initial_z[0]
            batch_x = batch_x.to(device).view(1, -1, model.config.input_size)
            graphs = graph.edges_to_matrix(edges, batch_x.shape[1], model.config)
            prediction, _ = utils.run_model_in_steps(batch_x, input_y, args, graphs, note_locations, model=model, device=device, initial_z=initial_z)

        trill_batch_x = torch.cat((batch_x[:,:,:78], prediction), 2)
        trill_prediction, _ = utils.run_model_in_steps(trill_batch_x, torch.zeros(1, num_notes, 5), args, graphs, note_locations, model=trill_model, device=device)

        prediction = torch.cat((prediction, trill_prediction), 2)
        prediction = scale_model_prediction_to_original(
            args, prediction, feature_stats)

        output_features = prediction_to_feature(prediction)
        output_features = add_note_location_to_features(output_features, note_locations)

        if return_features:
            return output_features

        output_xml_notes = xml_utils.apply_tempo_perform_features(xml_doc, xml_notes, output_features, start_time=1)

        output_midi, midi_pedal = xml_utils.xml_notes_to_midi(output_xml_notes)

        feature_path = Path(feature_path)
        file_name = feature_path.name[:-len('.dat')]
        output_name = args.test_result_path.joinpath(file_name)
        pSP_xml_utils.save_midi_notes_as_piano_midi(output_midi, midi_pedal, str(
            output_name), bool_pedal=False, disklavier=args.disklavier)

        print('midi saved: {}'.format(file_name))

def test(args,
         test_data,
         model,
         trill_model,
         device,
         feature_stats,
         constants):

    hier_model = None
    model_path = args.test_model_path
    hier_model_path = args.test_hierarchy_model_path
    trill_model_path = args.test_trill_model_path

    # load checkpoint and check device
    # if os.path.isfile('prime_' + args.modelCode + args.resume):
    if model_path.exists():
        # load model
        print("=> loading checkpoint '{}'".format(model_path.name))
        
        print('device is ', args.device)
        torch.cuda.set_device(args.device)
        if torch.cuda.is_available():
            def map_location(storage, loc): return storage.cuda()
        else:
            map_location = 'cpu'
        
        checkpoint = torch.load(model_path, map_location=map_location)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path.name, checkpoint['epoch']))

        # load trill model
        #trill_filename = args.trillCode + '_best.pth.tar'
        checkpoint = torch.load(trill_model_path, map_location=map_location)
        trill_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(trill_model_path.name, checkpoint['epoch']))
        
        # load hier_model
        if model.config.is_dependent: # when model: han_note_ar
            hier_model_config = param.load_parameters(args.parameters_folder, args.hierCode + '_param')
            hier_model = modelzoo.HAN_Integrated(hier_model_config, device, constants, step_by_step=True).to(device)
            #hier_filename = 'prime_' + args.hierCode + args.resume
            hier_checkpoint = torch.load(hier_model_path, map_location=map_location)
            hier_model.load_state_dict(hier_checkpoint['state_dict'])
            print("=> high-level model loaded checkpoint '{}' (epoch {})"
                  .format(hier_model_path.name, hier_checkpoint['epoch']))

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    model.config.is_teacher_force = False

    if args.sessMode == 'test':
        random.seed(0)
        load_file_and_generate_performance(
            test_data, args, model, device, constants, args.composer, args.latent, args.startTempo, feature_stats, hier_model=hier_model, trill_model=trill_model)

    elif args.sessMode == 'test_with_perform_z':
        pass


def encode_emotion_data(path_list):
