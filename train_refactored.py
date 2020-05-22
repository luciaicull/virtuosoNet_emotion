from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import math
import torch as th
import torch.nn as nn

import data_process
import utils_refactored as new_utils
import graph



def sigmoid(x, gain=1):
  # why not np.sigmoid or something?
  return 1 / (1 + math.exp(-gain*x))

class TraningSample():
    def __init__(self, index):
        self.index = index          # song index
        # list of start, end note index : [(start, end), (start, end), ...]
        self.slice_indexes = None

def get_accuracy(labels, preds):
    '''
    labels = np.array(labels)
    preds = np.array(preds)
    correct = (labels == preds)
    accuracy = correct.sum() / correct.size
    '''
    label_num = [0,0,0,0,0] # number of label 1~5
    correct_pred_num = [0,0,0,0,0] # number of correct prediction for label 1~5
    for label, pred in zip(labels, preds):
        label_num[label-1] += 1
        if label == pred:
            correct_pred_num[label-1] += 1
    e1_accuracy = correct_pred_num[0] / label_num[0]
    e2_accuracy = correct_pred_num[1] / label_num[1]
    e3_accuracy = correct_pred_num[2] / label_num[2]
    e4_accuracy = correct_pred_num[3] / label_num[3]
    e5_accuracy = correct_pred_num[4] / label_num[4]
    total_accuracy = sum(correct_pred_num) / sum(label_num)

    return e1_accuracy, e2_accuracy, e3_accuracy, e4_accuracy, e5_accuracy, total_accuracy

def train_classifier(args, model,
                     train_data, valid_data,
                     device, optimizer, bins, criterion, constants):
    logdir = Path(args.logs_folder)
    train_epoch_logdir = logdir.joinpath('train_epoch')
    train_epoch_logdir.mkdir(exist_ok=True)
    train_epoch_writer = SummaryWriter(train_epoch_logdir)
    valid_logdir = logdir.joinpath('valid')
    valid_logdir.mkdir(exist_ok=True)
    valid_writer = SummaryWriter(valid_logdir)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of Network Parameters is ', params)

    # criterion is cross entropy loss
    criterion = nn.CrossEntropyLoss()
    
    best_prime_loss = float("inf")
    start_epoch = 0
    NUM_UPDATED = 0

    train_xy = train_data
    valid_xy = valid_data
    print('number of train performances: ', len(train_xy),
          'number of valid perf: ', len(valid_xy))
    print('training sample example', valid_xy[0]['input_data'][0])

    count = 0

    for epoch in range(start_epoch, args.num_epochs):
        print('current training step is ', NUM_UPDATED)
        
        labels = []
        preds = []
        loss_total = []

        # slice song by using measure indices
        num_perf_data = len(train_xy)
        remaining_samples = []
        for i in range(num_perf_data):
            temp_training_sample = TraningSample(i)
            measure_numbers = [x.measure for x in train_xy[i]['note_location']['data']]
            data_size = len(train_xy[i]['input_data'])
            temp_training_sample.slice_indexes = data_process.make_slicing_indexes_by_measure(data_size, measure_numbers,
                                                                                              steps=args.time_steps)
            remaining_samples.append(temp_training_sample)
        print(sum([len(x.slice_indexes) for x in remaining_samples if x.slice_indexes]))

        # start training
        while len(remaining_samples) > 0:
            new_index = random.randrange(0, len(remaining_samples))
            selected_sample = remaining_samples[new_index]

            train_x = train_xy[selected_sample.index]['input_data'] # (note_num, input_feature_num)
            train_y = train_xy[selected_sample.index]['output_data'] 
            train_e1_data = train_xy[selected_sample.index]['e1_perform_data']
            label = train_xy[selected_sample.index]['label'] # integer (emotion number)

            note_locations = train_xy[selected_sample.index]['note_location']['data']
            align_matched = train_xy[selected_sample.index]['align_matched']['data']

            pedal_status = train_xy[selected_sample.index]['articulation_loss_weight']['data']
            edges = train_xy[selected_sample.index]['graph']

            num_slice = len(selected_sample.slice_indexes)
            selected_idx = random.randrange(0, num_slice)
            slice_idx = selected_sample.slice_indexes[selected_idx]

            graphs = None
            '''
            key_lists = [0]
            key = 0
            
            for i in range(args.num_key_augmentation):
                while key in key_lists:
                    key = random.randrange(-5, 7)
                key_lists.append(key)
            
            for i in range(args.num_key_augmentation+1):
                key = key_lists[i]
                temp_train_x = data_process.key_augmentation(train_x, key)
                kld_weight = sigmoid((NUM_UPDATED - args.kld_sig) / (args.kld_sig/10)) * args.kld_max
            '''
            training_data = {'x': train_x, 'y': train_y, 'label': label,
                            'e1_data': train_e1_data,
                            'note_locations': note_locations, 'graphs': graphs,
                            'align_matched': align_matched, 'pedal_status': pedal_status,
                            'slice_idx': slice_idx}
            
            loss, pred = new_utils.batch_train_run_classifier(
                training_data, model=model, args=args, optimizer=optimizer, const=constants, criterion=criterion)
            
            loss_total.append(loss.item())
            
            labels.append(label)
            preds.append(pred)
            
            NUM_UPDATED += 1
            
            del selected_sample.slice_indexes[selected_idx]
            if len(selected_sample.slice_indexes) == 0:
                del remaining_samples[new_index]
        
        e1_accuracy, e2_accuracy, e3_accuracy, e4_accuracy, e5_accuracy, total_accuracy = get_accuracy(labels, preds)
        print('===========================================================================')
        print('Epoch[{}/{}], Loss: {: .4f}, Accuracy- e1: {: .4f}, e2: {: .4f}, e3: {: .4f}, e4: {: .4f}, e5: {: .4f}, total: {: .4f}'.format(
            epoch+1, args.num_epochs, np.mean(loss_total), e1_accuracy, e2_accuracy, e3_accuracy, e4_accuracy, e5_accuracy, total_accuracy))
        print('===========================================================================')

        train_epoch_writer.add_scalar('loss', np.mean(loss_total), global_step=epoch)

        # start validation
        loss_total = []
        labels = []
        preds = []
        for xy_tuple in valid_xy:
            valid_x = xy_tuple['input_data']
            valid_y = xy_tuple['output_data']
            valid_e1_data = xy_tuple['e1_perform_data']
            valid_label = xy_tuple['label']
            note_locations = xy_tuple['note_location']['data']
            align_matched = xy_tuple['align_matched']['data']
            pedal_status = xy_tuple['articulation_loss_weight']['data']

            measure_numbers = [x.measure for x in note_locations]
            slice_indexes = data_process.make_slicing_indexes_by_measure(
                            len(valid_x), measure_numbers, steps=args.valid_steps, overlap=False)
            valid_data = {'x': valid_x, 'y': valid_y, 'label': valid_label,
                          'e1_data': valid_e1_data,
                          'note_locations': note_locations, 'graphs': graphs,
                          'align_matched': align_matched, 'pedal_status': pedal_status,
                          'slice_indexes': slice_indexes}
            
            loss, label, pred = new_utils.batch_eval_run_classifier(
                valid_data, model=model, args=args, optimizer=optimizer, const=constants, criterion=criterion)
            
            loss_total += loss
            labels += label
            preds += pred

        mean_loss = np.mean(loss_total)
        valid_writer.add_scalar('loss', mean_loss, global_step=epoch)
        e1_accuracy, e2_accuracy, e3_accuracy, e4_accuracy, e5_accuracy, total_accuracy = get_accuracy(labels, preds)
        
        print("Valid Loss={: .4f}, Valid Accuracy - e1: {: .4f}, e2: {: .4f}, e3: {: .4f}, e4: {: .4f}, e5: {: .4f}, total: {: .4f}".format(
            mean_loss, e1_accuracy, e2_accuracy, e3_accuracy, e4_accuracy, e5_accuracy, total_accuracy))

        is_best = mean_loss < best_prime_loss
        best_prime_loss = min(mean_loss, best_prime_loss)

        new_utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_valid_loss': best_prime_loss,
            'optimizer': optimizer.state_dict(),
            'training_step': NUM_UPDATED
        }, is_best, model_name=args.modelCode, folder=str(args.checkpoints_folder), epoch='{0:03d}'.format(epoch))
        


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
    train_epoch_logdir = logdir.joinpath('train_epoch')
    train_epoch_logdir.mkdir(exist_ok=True)
    train_epoch_writer = SummaryWriter(train_epoch_logdir)
    valid_logdir = logdir.joinpath('valid')
    valid_logdir.mkdir(exist_ok=True)
    valid_writer = SummaryWriter(valid_logdir)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of Network Parameters is ', params)

    best_prime_loss = float("inf")
    best_trill_loss = float("inf")
    start_epoch = 0
    NUM_UPDATED = 0

    train_xy = train_data
    test_xy = valid_data
    print('number of train performances: ', len(train_xy),
          'number of valid perf: ', len(test_xy))
    print('training sample example', train_xy[0]['input_data'][0])

    train_model = model
    count = 0

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
            measure_numbers = [
                x.measure for x in train_xy[i]['note_location']['data']]
            data_size = len(train_xy[i]['input_data'])
            
            # direct_measure
            if model.config.hierarchy_level == 'measure' and not model.config.is_dependent:
                temp_training_sample.slice_indexes = data_process.make_slice_with_same_measure_number(data_size,
                                                                                            measure_numbers,
                                                                                            measure_steps=args.time_steps)

            # direct_note
            else:
                temp_training_sample.slice_indexes = data_process.make_slicing_indexes_by_measure(data_size, measure_numbers,
                                                                                        steps=args.time_steps)
            remaining_samples.append(temp_training_sample)
        print(sum([len(x.slice_indexes)for x in remaining_samples if x.slice_indexes]))

        while len(remaining_samples) > 0:
            new_index = random.randrange(0, len(remaining_samples))
            selected_sample = remaining_samples[new_index]

            train_x = train_xy[selected_sample.index]['input_data']
            train_y = train_xy[selected_sample.index]['output_data']

            note_locations = train_xy[selected_sample.index]['note_location']['data']
            align_matched = train_xy[selected_sample.index]['align_matched']['data']

            pedal_status = train_xy[selected_sample.index]['articulation_loss_weight']['data']
            edges = train_xy[selected_sample.index]['graph']

            num_slice = len(selected_sample.slice_indexes)
            selected_idx = random.randrange(0, num_slice)
            slice_idx = selected_sample.slice_indexes[selected_idx]

            graphs = None

            key_lists = [0]
            key = 0
            for i in range(args.num_key_augmentation):
                while key in key_lists:
                    key = random.randrange(-5, 7)
                key_lists.append(key)

            for i in range(args.num_key_augmentation+1):
                key = key_lists[i]
                temp_train_x = data_process.key_augmentation(train_x, key)
                kld_weight = sigmoid((NUM_UPDATED - args.kld_sig) / (args.kld_sig/10)) * args.kld_max

                training_data = {'x': temp_train_x, 'y': train_y, 'graphs': graphs,
                                'note_locations': note_locations,
                                'align_matched': align_matched, 'pedal_status': pedal_status,
                                'slice_idx': slice_idx, 'kld_weight': kld_weight}
                tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss = \
                    new_utils.batch_train_run_direct(training_data, model=train_model, args=args, optimizer=optimizer, const=constants)

                tempo_loss_total.append(tempo_loss.item())
                vel_loss_total.append(vel_loss.item())
                dev_loss_total.append(dev_loss.item())
                articul_loss_total.append(articul_loss.item())
                pedal_loss_total.append(pedal_loss.item())
                trill_loss_total.append(trill_loss.item())
                #kld_total.append(kld.item())
                NUM_UPDATED += 1
            
            del selected_sample.slice_indexes[selected_idx]
            
        print('===========================================================================')
        print('Epoch [{}/{}], Loss - Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}, KLD: {:.4f}'
              .format(epoch + 1, args.num_epochs, np.mean(tempo_loss_total), np.mean(vel_loss_total),
                      np.mean(dev_loss_total), np.mean(articul_loss_total), np.mean(pedal_loss_total), np.mean(trill_loss_total), np.mean(kld_total)))
        print('===========================================================================')

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
        tempo_loss_total = []
        vel_loss_total = []
        deviation_loss_total = []
        articul_loss_total = []
        pedal_loss_total = []
        trill_loss_total = []
        kld_loss_total = []

        for xy_tuple in test_xy:
            test_x = xy_tuple['input_data']
            test_y = xy_tuple['output_data']
            note_locations = xy_tuple['note_location']['data']
            align_matched = xy_tuple['align_matched']['data']
            pedal_status = xy_tuple['articulation_loss_weight']['data']
            edges = xy_tuple['graph']
            graphs = graph.edges_to_matrix(edges, len(test_x), model.config)

            batch_x, batch_y = new_utils.handle_data_in_tensor(test_x, test_y, model.config, device, constants)
            batch_x = batch_x.view(1, -1, model.config.input_size)
            batch_y = batch_y.view(1, -1, model.config.output_size)
            
            align_matched = th.Tensor(align_matched).view(1, -1, 1).to(device)
            pedal_status = th.Tensor(pedal_status).view(1,-1,1).to(device)
            outputs, total_z = new_utils.run_model_in_steps(batch_x, batch_y, args, graphs, note_locations, model, device)

            # direct_measure
            if model.config.hierarchy_level and not model.config.is_dependent:
                if model.config.hierarchy_level == 'measure':
                    hierarchy_numbers = [x.measure for x in note_locations]
                elif model.config.hierarchy_level == 'beat':
                    hierarchy_numbers = [x.beat for x in note_locations]
                tempo_y = model.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, 0, 0)
                vel_y = model.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, 0, 1)

                tempo_loss = criterion(outputs[:, :, 0:1], tempo_y, model.config)
                vel_loss = criterion(outputs[:, :, 1:2], vel_y, model.config)
                if args.deltaLoss: #default=False
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
                    kld_loss = -0.5 * \
                        th.sum(1 + perform_var -
                               perform_mu.pow(2) - perform_var.exp())
                    kld_loss_total.append(kld_loss.item())

            # direct_note
            else:
                valid_loss, tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss = utils.cal_loss_by_output_type(outputs, batch_y, align_matched, pedal_status, args, model.config, note_locations, 0, constants)
                for z in total_z:
                    perform_mu, perform_var = z
                    kld_loss = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                    kld_loss_total.append(kld_loss.item())
                trill_loss = th.zeros(1)
            
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

        valid_writer.add_scalar('tempo_loss', mean_tempo_loss, global_step=epoch)
        valid_writer.add_scalar('vel_loss', mean_vel_loss, global_step=epoch)
        valid_writer.add_scalar('dev_loss', mean_deviation_loss, global_step=epoch)
        valid_writer.add_scalar('articulation_loss', mean_articul_loss, global_step=epoch)
        valid_writer.add_scalar('pedal_loss', mean_pedal_loss, global_step=epoch)
        valid_writer.add_scalar('trill_loss', mean_trill_loss, global_step=epoch)
        valid_writer.add_scalar('kld', mean_kld_loss, global_step=epoch)
        print("Valid Loss= {:.4f} , Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}"
              .format(mean_valid_loss, mean_tempo_loss , mean_vel_loss,
                      mean_deviation_loss, mean_articul_loss, mean_pedal_loss, mean_trill_loss))

        is_best = mean_valid_loss < best_prime_loss
        best_prime_loss = min(mean_valid_loss, best_prime_loss)

        is_best_trill = mean_trill_loss < best_trill_loss
        best_trill_loss = min(mean_trill_loss, best_trill_loss)

        new_utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_valid_loss': best_prime_loss,
            'optimizer': optimizer.state_dict(),
            'training_step': NUM_UPDATED
        }, is_best, model_name=args.modelCode, folder=str(args.checkpoints_folder), epoch='{0:03d}'.format(epoch))
    #end of epoch
