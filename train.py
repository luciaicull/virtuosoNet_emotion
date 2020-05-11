from pathlib import Path
import os
import random
import math
import copy

import numpy as np
import torch as th
import pickle

from .parser import get_parser
from .utils import categorize_value_to_vector
from . import data_process as dp  # maybe confuse with dynamic programming?
from . import graph
from . import utils
#from . import model_constants as const
from . import model
from torch.utils.tensorboard import SummaryWriter

def sigmoid(x, gain=1):
  # why not np.sigmoid or something?
  return 1 / (1 + math.exp(-gain*x))

class TraningSample():
    def __init__(self, index):
        self.index = index          # song index
        self.slice_indexes = None   # list of start, end note index : [(start, end), (start, end), ...]

def train(args,
          model,
          train_data,
          valid_data,
          device,
          optimizer, 
          bins,
          criterion,
          constants):
    logdir = Path(args.logs_folder)
    train_logdir = logdir.joinpath('train')
    train_logdir.mkdir(exist_ok=True)
    writer = SummaryWriter(train_logdir)
    train_epoch_logdir = logdir.joinpath('train_epoch')
    train_epoch_logdir.mkdir(exist_ok=True)
    train_epoch_writer = SummaryWriter(train_epoch_logdir)
    valid_logdir = logdir.joinpath('valid')
    valid_logdir.mkdir(exist_ok=True)
    valid_writer = SummaryWriter(valid_logdir)


    # isn't this redundant?
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of Network Parameters is ', params)

    best_prime_loss = float("inf")
    best_trill_loss = float("inf")
    start_epoch = 0
    NUM_UPDATED = 0

    if args.resumeTraining and not args.trainTrill:     
        # [Default] 
        #   args.resumeTraining : False
        #   args.trainTrill: False
        # Load trained-model to resume the training process
        if os.path.isfile('prime_' + args.modelCode + args.resume):
            print("=> loading checkpoint '{}'".format(args.modelCode + args.resume))
            # model_codes = ['prime', 'trill']
            filename = 'prime_' + args.modelCode + args.resume
            checkpoint = th.load(filename,  map_location=device)
            best_valid_loss = checkpoint['best_valid_loss']
            model.load_state_dict(checkpoint['state_dict'])
            model.device = device
            optimizer.load_state_dict(checkpoint['optimizer'])
            NUM_UPDATED = checkpoint['training_step']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
            start_epoch = checkpoint['epoch'] - 1
            best_prime_loss = checkpoint['best_valid_loss']
            print('Best valid loss was ', best_prime_loss)
    
    train_xy = train_data
    test_xy = valid_data
    print('number of train performances: ', len(train_xy), 'number of valid perf: ', len(test_xy))
    print('training sample example', train_xy[0]['input_data'][0])

    train_model = model
    count = 0
    # total_step = len(train_loader)
    for epoch in range(start_epoch, args.num_epochs):
        print('current training step is ', NUM_UPDATED)
        tempo_loss_total = []
        vel_loss_total = []
        dev_loss_total = []
        articul_loss_total = []
        pedal_loss_total = []
        trill_loss_total = []
        kld_total = []

        num_perf_data = len(train_xy)
        remaining_samples = []
        for i in range(num_perf_data):
            temp_training_sample = TraningSample(i)
            measure_numbers = [x.measure for x in train_xy[i]['note_location']['data']]
            data_size = len(train_xy[i]['input_data'])
            if model.config.hierarchy_level == 'measure' and not model.config.is_dependent:
                temp_training_sample.slice_indexes = dp.make_slice_with_same_measure_number(data_size,
                                                                                       measure_numbers,
                                                                                       measure_steps=args.time_steps)

            else:
                temp_training_sample.slice_indexes = dp.make_slicing_indexes_by_measure(data_size, measure_numbers,
                                                                                   steps=args.time_steps)
            remaining_samples.append(temp_training_sample)
        print(sum([len(x.slice_indexes) for x in remaining_samples if x.slice_indexes]))

        del_count = 0
        while len(remaining_samples) > 0:
            new_index = random.randrange(0, len(remaining_samples))
            selected_sample = remaining_samples[new_index]
            # train_x = train_xy[selected_sample.index][0]
            # train_y = train_xy[selected_sample.index][1]
            train_x = train_xy[selected_sample.index]['input_data']
            train_y = train_xy[selected_sample.index]['output_data']
            # if args.loss == 'CE':
            #     train_y = categorize_value_to_vector(train_y, bins)
            note_locations = train_xy[selected_sample.index]['note_location']['data']
            align_matched = train_xy[selected_sample.index]['align_matched']['data']
            # TODO: which variable would be corresponds to pedal status?
            # pedal_status = train_xy[selected_sample.index][4]
            pedal_status = train_xy[selected_sample.index]['articulation_loss_weight']['data']
            edges = train_xy[selected_sample.index]['graph']

            num_slice = len(selected_sample.slice_indexes)
            selected_idx = random.randrange(0,num_slice)
            slice_idx = selected_sample.slice_indexes[selected_idx]

            if model.config.is_graph:
                graphs = graph.edges_to_matrix_short(edges, slice_idx, model.config)
            else:
                graphs = None

            key_lists = [0]
            key = 0
            for i in range(args.num_key_augmentation):
                while key in key_lists:
                    key = random.randrange(-5, 7)
                key_lists.append(key)

            for i in range(args.num_key_augmentation+1):
                #try:
                key = key_lists[i]
                temp_train_x = dp.key_augmentation(train_x, key)
                kld_weight = sigmoid((NUM_UPDATED - args.kld_sig) / (args.kld_sig/10)) * args.kld_max

                training_data = {'x': temp_train_x, 'y': train_y, 'graphs': graphs,
                                'note_locations': note_locations,
                                'align_matched': align_matched, 'pedal_status': pedal_status,
                                'slice_idx': slice_idx, 'kld_weight': kld_weight}
                try:
                    tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, kld = \
                        utils.batch_train_run(training_data, model=train_model, args=args, optimizer=optimizer, const=constants)
                except Exception as ex:
                    pass
                #    print(ex)
                #    print(train_xy[selected_sample.index]['perform_path'])
                tempo_loss_total.append(tempo_loss.item())
                vel_loss_total.append(vel_loss.item())
                dev_loss_total.append(dev_loss.item())
                articul_loss_total.append(articul_loss.item())
                pedal_loss_total.append(pedal_loss.item())
                trill_loss_total.append(trill_loss.item())
                kld_total.append(kld.item())
                NUM_UPDATED += 1

                #except:
                #    print(train_xy[selected_sample.index]['perform_path'])
            del selected_sample.slice_indexes[selected_idx]
            if len(selected_sample.slice_indexes) == 0:
                # print('every slice in the sample is trained')
                del remaining_samples[new_index]
                del_count += 1
            #print('sample [{}/{}], Loss - Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}, KLD: {:.4f}'
            #      .format(del_count, len(train_xy), np.mean(tempo_loss_total), np.mean(vel_loss_total),
            #              np.mean(dev_loss_total), np.mean(articul_loss_total), np.mean(pedal_loss_total), np.mean(trill_loss_total), np.mean(kld_total)))
            if del_count % 50 == 0:
                #print('sample [{}/{}], Loss - Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}, KLD: {:.4f}'
                #      .format(del_count, len(train_xy), np.mean(tempo_loss_total), np.mean(vel_loss_total),
                #              np.mean(dev_loss_total), np.mean(articul_loss_total), np.mean(pedal_loss_total), np.mean(trill_loss_total), np.mean(kld_total)))
                writer.add_scalar('tempo_loss', np.mean(
                    tempo_loss_total), global_step=count)
                writer.add_scalar('vel_loss', np.mean(
                    vel_loss_total), global_step=count)
                writer.add_scalar('dev_loss', np.mean(
                    dev_loss_total), global_step=count)
                writer.add_scalar('articulation_loss', np.mean(
                    articul_loss_total), global_step=count)
                writer.add_scalar('pedal_loss', np.mean(
                    pedal_loss_total), global_step=count)
                writer.add_scalar('trill_loss', np.mean(
                    trill_loss_total), global_step=count)
                writer.add_scalar('kld', np.mean(kld_total), global_step=count)
            count += 1    
            # print("Remaining samples: ", sum([len(x.slice_indexes) for x in remaining_samples if x.slice_indexes]))
        print('===========================================================================')
        print('Epoch [{}/{}], Loss - Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}, KLD: {:.4f}'
              .format(epoch + 1, args.num_epochs, np.mean(tempo_loss_total), np.mean(vel_loss_total),
                      np.mean(dev_loss_total), np.mean(articul_loss_total), np.mean(pedal_loss_total), np.mean(trill_loss_total), np.mean(kld_total)))
        print(
            '===========================================================================')
        train_epoch_writer.add_scalar('tempo_loss', np.mean(
            tempo_loss_total), global_step=epoch)
        train_epoch_writer.add_scalar('vel_loss', np.mean(
            vel_loss_total), global_step=epoch)
        train_epoch_writer.add_scalar('dev_loss', np.mean(
            dev_loss_total), global_step=epoch)
        train_epoch_writer.add_scalar('articulation_loss', np.mean(
            articul_loss_total), global_step=epoch)
        train_epoch_writer.add_scalar('pedal_loss', np.mean(
            pedal_loss_total), global_step=epoch)
        train_epoch_writer.add_scalar('trill_loss', np.mean(
            trill_loss_total), global_step=epoch)
        train_epoch_writer.add_scalar(
            'kld', np.mean(kld_total), global_step=epoch)
        ## Validation
        tempo_loss_total =[]
        vel_loss_total =[]
        deviation_loss_total =[]
        articul_loss_total = []
        pedal_loss_total = []
        trill_loss_total = []
        kld_loss_total = []

        for xy_tuple in test_xy:
            test_x = xy_tuple['input_data']
            test_y = xy_tuple['output_data']
            note_locations = xy_tuple['note_location']['data']
            align_matched = xy_tuple['align_matched']['data']
            # TODO: need check
            pedal_status = xy_tuple['articulation_loss_weight']['data']
            edges = xy_tuple['graph']
            graphs = graph.edges_to_matrix(edges, len(test_x), model.config)
            # if args.loss == 'CE':
            #     test_y = categorize_value_to_vector(test_y, bins)

            batch_x, batch_y = utils.handle_data_in_tensor(test_x, test_y, model.config, device)
            batch_x = batch_x.view(1, -1, model.config.input_size)
            batch_y = batch_y.view(1, -1, model.config.output_size)
            # input_y = th.Tensor(prev_feature).view((1, -1, TOTAL_OUTPUT)).to(device)
            align_matched = th.Tensor(align_matched).view(1, -1, 1).to(device)
            pedal_status = th.Tensor(pedal_status).view(1,-1,1).to(device)
            outputs, total_z = utils.run_model_in_steps(batch_x, batch_y, args, graphs, note_locations, model, device)

            # valid_loss = criterion(outputs[:,:,const.NUM_TEMPO_PARAM:-const.num_trill_param], batch_y[:,:,const.NUM_TEMPO_PARAM:-const.num_trill_param], align_matched)
            if model.config.hierarchy_level and not model.config.is_dependent:
                if model.config.hierarchy_level == 'measure':
                    hierarchy_numbers = [x.measure for x in note_locations]
                elif model.config.hierarchy_level == 'beat':
                    hierarchy_numbers = [x.beat for x in note_locations]
                tempo_y = model.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, 0, 0)
                vel_y = model.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, 0, 1)

                tempo_loss = criterion(outputs[:, :, 0:1], tempo_y, model.config)
                vel_loss = criterion(outputs[:, :, 1:2], vel_y, model.config)
                if args.deltaLoss:
                    tempo_out_delta = outputs[:, 1:, 0:1] - outputs[:, :-1, 0:1]
                    tempo_true_delta = tempo_y[:, 1:, :] - tempo_y[:, :-1, :]
                    vel_out_delta = outputs[:, 1:, 1:2] - outputs[:, :-1, 1:2]
                    vel_true_delta = vel_y[:, 1:, :] - vel_y[:, :-1, :]

                    tempo_loss += criterion(tempo_out_delta, tempo_true_delta, model.config) * args.delta_weight
                    vel_loss += criterion(vel_out_delta, vel_true_delta, model.config) * args.delta_weight

                dev_loss = th.zeros(1)
                articul_loss = th.zeros(1)
                pedal_loss = th.zeros(1)
                trill_loss = th.zeros(1)

                for z in total_z:
                    perform_mu, perform_var = z
                    kld_loss = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                    kld_loss_total.append(kld_loss.item())
            elif model.config.is_trill:
                trill_bool = batch_x[:,:, const.is_trill_index_concated] == 1
                trill_bool = trill_bool.float().view(1,-1,1).to(device)
                trill_loss = criterion(outputs, batch_y, model.config, trill_bool)

                tempo_loss = th.zeros(1)
                vel_loss = th.zeros(1)
                dev_loss = th.zeros(1)
                articul_loss = th.zeros(1)
                pedal_loss = th.zeros(1)
                kld_loss = th.zeros(1)
                kld_loss_total.append(kld_loss.item())

            else:
                valid_loss, tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss = utils.cal_loss_by_output_type(outputs, batch_y, align_matched, pedal_status, args, model.config, note_locations, 0)
                for z in total_z:
                    perform_mu, perform_var = z
                    kld_loss = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                    kld_loss_total.append(kld_loss.item())
                trill_loss = th.zeros(1)

            # valid_loss_total.append(valid_loss.item())
            tempo_loss_total.append(tempo_loss.item())
            vel_loss_total.append(vel_loss.item())
            deviation_loss_total.append(dev_loss.item())
            articul_loss_total.append(articul_loss.item())
            pedal_loss_total.append(pedal_loss.item())
            trill_loss_total.append(trill_loss.item())

        mean_tempo_loss = np.mean(tempo_loss_total)
        mean_vel_loss = np.mean(vel_loss_total)
        mean_deviation_loss = np.mean(deviation_loss_total)
        mean_articul_loss = np.mean(articul_loss_total)
        mean_pedal_loss = np.mean(pedal_loss_total)
        mean_trill_loss = np.mean(trill_loss_total)
        mean_kld_loss = np.mean(kld_loss_total)

        mean_valid_loss = (mean_tempo_loss + mean_vel_loss + mean_deviation_loss + mean_articul_loss + mean_pedal_loss * 7 + mean_kld_loss * kld_weight) / (11 + kld_weight)

        valid_writer.add_scalar(
            'tempo_loss', mean_tempo_loss, global_step=epoch)
        valid_writer.add_scalar(
            'vel_loss', mean_vel_loss, global_step=epoch)
        valid_writer.add_scalar(
            'dev_loss', mean_deviation_loss, global_step=epoch)
        valid_writer.add_scalar(
            'articulation_loss', mean_articul_loss, global_step=epoch)
        valid_writer.add_scalar(
            'pedal_loss', mean_pedal_loss, global_step=epoch)
        valid_writer.add_scalar(
            'trill_loss', mean_trill_loss, global_step=epoch)
        valid_writer.add_scalar('kld', mean_kld_loss, global_step=epoch)
        print("Valid Loss= {:.4f} , Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}"
              .format(mean_valid_loss, mean_tempo_loss , mean_vel_loss,
                      mean_deviation_loss, mean_articul_loss, mean_pedal_loss, mean_trill_loss))

        is_best = mean_valid_loss < best_prime_loss
        best_prime_loss = min(mean_valid_loss, best_prime_loss)

        is_best_trill = mean_trill_loss < best_trill_loss
        best_trill_loss = min(mean_trill_loss, best_trill_loss)

        if model.config.is_trill:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_valid_loss': best_trill_loss,
                'optimizer': optimizer.state_dict(),
                'training_step': NUM_UPDATED
            }, is_best_trill, model_name=args.modelCode, folder=str(args.checkpoints_folder), epoch='{0:03d}'.format(epoch))
        else:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_valid_loss': best_prime_loss,
                'optimizer': optimizer.state_dict(),
                'training_step': NUM_UPDATED
            }, is_best, model_name=args.modelCode, folder=str(args.checkpoints_folder), epoch='{0:03d}'.format(epoch))
    #end of epoch
