import torch as th
import shutil

import data_process


def save_checkpoint(state, is_best, model_name, folder='', epoch=''):
    save_name = folder + '/' + epoch + '_' + model_name + '_checkpoint.pth.tar'
    print('checkpoint saved: ', save_name)
    th.save(state, save_name)
    if is_best:
        best_name = folder + '/' + model_name + '_best.pth.tar'
        shutil.copyfile(save_name, best_name)

def make_tensor(data, args, const, model):
    device = args.device
    
    batch_start, batch_end = data['slice_idx']
    batch_x = th.Tensor(data['x'][batch_start:batch_end]) # (slice length, input_feature_num)
    batch_y = th.Tensor(data['y'][batch_start:batch_end])
    batch_e1_y = th.Tensor(data['e1_data'][batch_start:batch_end])
    batch_x = batch_x.to(device)
    batch_y = batch_y[:, :const.NUM_PRIME_PARAM].to(device)
    batch_e1_y = batch_e1_y.to(device)
    batch_x = batch_x.view((args.batch_size, -1, model.config.input_size))
    batch_y = batch_y.view((args.batch_size, -1, model.config.output_size))
    batch_e1_y = batch_e1_y.view((args.batch_size, -1, model.config.output_size))

    label = th.Tensor([data['label']]).to(device).long()  # integer, emotion number

    align_matched = th.Tensor(data['align_matched'][batch_start:batch_end]).view(
        (args.batch_size, -1, 1)).to(args.device)
    pedal_status = th.Tensor(data['pedal_status'][batch_start:batch_end]).view(
        (args.batch_size, -1, 1)).to(args.device)
    note_locations = data['note_locations']

    return batch_x, batch_y, batch_e1_y, label, align_matched, pedal_status, note_locations

def batch_train_run_classifier(data, model, args, optimizer, const, criterion):
    batch_start, batch_end = data['slice_idx']
    batch_x, batch_y, batch_e1_y, label, align_matched, pedal_status, note_locations = make_tensor(data, args, const, model)

    model_train = model.train()
    output_vector = model_train(batch_x, batch_y, batch_e1_y, note_locations, batch_start)

    loss = criterion(output_vector, label)
    optimizer.zero_grad()
    loss.backward()
    th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    _, predicted_class = th.max(output_vector, 0)

    return loss, predicted_class.item()


def batch_eval_run_classifier(data, model, args, optimizer, const, criterion):


'''
def run_classifer_model_in_steps(input, input_y, label, args, note_locations, model, device, criterion, initial_z=False):
    num_notes = input.shape[1]
    with th.no_grad():
        model_eval = model.eval()
        measure_numbers = [x.measure for x in note_locations]
        slice_indexes = data_process.make_slicing_indexes_by_measure(
            num_notes, measure_numbers, steps=args.valid_steps, overlap=False)

        total_loss = []
        for slice_idx in slice_indexes:
            batch_start, batch_end = slice_idx
            batch_input = input[:, batch_start:batch_end, :].view(1, -1, model.config.input_size)
            batch_input_y = input_y[:, batch_start:batch_end, :].view(1, -1, model.config.output_size)
            output_vector = model_eval(batch_input, batch_input_y, note_locations, batch_start)

            loss = criterion(output_vector, label)
            total_loss.append(loss.item())
        
    return total_loss
'''

def batch_train_run_direct(data, model, args, optimizer, const):
    batch_start, batch_end = data['slice_idx']
    batch_x, batch_y = handle_data_in_tensor(
        data['x'][batch_start:batch_end], data['y'][batch_start:batch_end], model.config, device=args.device, const=const)

    batch_x = batch_x.view((args.batch_size, -1, model.config.input_size))
    batch_y = batch_y.view((args.batch_size, -1, model.config.output_size))
    
    align_matched = th.Tensor(data['align_matched'][batch_start:batch_end]).view(
        (args.batch_size, -1, 1)).to(args.device)
    pedal_status = th.Tensor(data['pedal_status'][batch_start:batch_end]).view(
        (args.batch_size, -1, 1)).to(args.device)
    note_locations = data['note_locations']

    if data['graphs'] is not None:
        edges = data['graphs']
        if edges.shape[1] == batch_end - batch_start:
            edges = edges.to(args.device)
        else:
            edges = edges[:, batch_start:batch_end,
                          batch_start:batch_end].to(args.device)
    else:
        edges = data['graphs']

    model_train = model.train()
    outputs, total_out_list \
        = model_train(batch_x, batch_y, edges, note_locations, batch_start)

    # direct_measure
    if model.config.hierarchy_level in ['measure', 'beat'] and not model.config.is_dependent:
        if model.config.hierarchy_level == 'measure':
            hierarchy_numbers = [x.measure for x in note_locations]
        elif model.config.hierarchy_level == 'beat':
            hierarchy_numbers = [x.beat for x in note_locations]
        
        tempo_in_hierarchy = model.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, batch_start, 0)
        dynamics_in_hierarchy = model.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, batch_start, 1)
        tempo_loss = criterion(outputs[:, :, 0:1], tempo_in_hierarchy, model.config)
        vel_loss = criterion(outputs[:, :, 1:2], dynamics_in_hierarchy, model.config)
        if args.deltaLoss and outputs.shape[1] > 1:
            vel_out_delta = outputs[:, 1:, 1:2] - outputs[:, :-1, 1:2]
            vel_true_delta = dynamics_in_hierarchy[:,
                                                   1:, :] - dynamics_in_hierarchy[:, :-1, :]

            vel_loss += criterion(vel_out_delta, vel_true_delta,
                                  model.config) * args.delta_weight
            vel_loss /= 1 + args.delta_weight
        total_loss = tempo_loss + vel_loss
    # direct_note
    else:
        total_loss, tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss =\
            cal_loss_by_output_type(outputs, batch_y, align_matched, pedal_status, args,
                                    model.config, note_locations, batch_start, const)

    optimizer.zero_grad()
    total_loss.backward()
    th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    # direct_measure
    if model.config.hierarchy_level in ['measure', 'beat'] and not model.config.is_dependent:
        return tempo_loss, vel_loss, th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1)
    # direct_note
    else:  
        return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, th.zeros(1)


def handle_data_in_tensor(x, y, model_config, device, const, hierarchy_test=False):
    x = th.Tensor(x)
    y = th.Tensor(y)
    if model_config.hierarchy_level == 'measure':  # direct_measure, direct_note
        hierarchy_output = y[:, const.MEAS_TEMPO_IDX:const.MEAS_TEMPO_IDX+2]
    elif model_config.hierarchy_level == 'beat':
        hierarchy_output = y[:, const.BEAT_TEMPO_IDX:const.BEAT_TEMPO_IDX+2]

    # direct_measure
    if model_config.hierarchy_level in ['measure', 'beat'] and not model_config.is_dependent:
        y = hierarchy_output
    # direct_note
    elif model_config.is_dependent:  
        x = th.cat((x, hierarchy_output), 1)
        y = y[:, :const.NUM_PRIME_PARAM]
    
    return x.to(device), y.to(device)


def criterion(pred, target, model_config, aligned_status=1):
    if isinstance(aligned_status, int):
        data_size = pred.shape[-2] * pred.shape[-1]
    else:
        data_size = th.sum(aligned_status).item() * pred.shape[-1]
        if data_size == 0:
            data_size = 1

    if target.shape != pred.shape:
        print('Error: The shape of the target and prediction for the loss calculation is different')
        print(target.shape, pred.shape)
        # return th.zeros(1).to(DEVICE)
        return th.zeros(1)

    return th.sum(((target - pred) ** 2) * aligned_status) / data_size


def cal_loss_by_output_type(output, target_y, align_matched, pedal_weight, args, model_config, note_locations, batch_start, const):
    tempo_loss = cal_tempo_loss_in_beat(
        output, target_y, note_locations, batch_start, args, model_config, const)
    vel_loss = criterion(output[:, :, const.VEL_PARAM_IDX:const.VEL_PARAM_IDX + 1],
                         target_y[:, :, const.VEL_PARAM_IDX:const.VEL_PARAM_IDX + 1], model_config,  align_matched)
    dev_loss = criterion(output[:, :, const.DEV_PARAM_IDX:const.DEV_PARAM_IDX + 1],
                         target_y[:, :, const.DEV_PARAM_IDX:const.DEV_PARAM_IDX + 1], model_config,  align_matched)
    articul_loss = criterion(output[:, :, const.ARTICULATION_PARAM_IDX:const.ARTICULATION_PARAM_IDX + 1],
                             target_y[:, :, const.ARTICULATION_PARAM_IDX:const.ARTICULATION_PARAM_IDX + 1], model_config, align_matched)
    pedal_loss = criterion(output[:, :, const.PEDAL_PARAM_IDX:const.PEDAL_PARAM_IDX + const.PEDAL_PARAM_LEN],
                           target_y[:, :, const.PEDAL_PARAM_IDX:const.PEDAL_PARAM_IDX + const.PEDAL_PARAM_LEN], model_config, pedal_weight)
    total_loss = (tempo_loss + vel_loss + dev_loss +
                  articul_loss + pedal_loss * 7) / 11

    return total_loss, tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss


def cal_tempo_loss_in_beat(pred_x, true_x, note_locations, start_index, args, model_config, const):
    previous_beat = -1

    num_notes = pred_x.shape[1]
    start_beat = note_locations[start_index].beat
    num_beats = note_locations[num_notes+start_index-1].beat - start_beat + 1

    pred_beat_tempo = th.zeros(
        [num_beats, const.OUTPUT_TEMPO_PARAM_LEN]).to(args.device)
    true_beat_tempo = th.zeros(
        [num_beats, const.OUTPUT_TEMPO_PARAM_LEN]).to(args.device)
    for i in range(num_notes):
        current_beat = note_locations[i+start_index].beat
        if current_beat > previous_beat:
            previous_beat = current_beat
            if model_config.is_baseline:
                for j in range(i, num_notes):
                    if note_locations[j+start_index].beat > current_beat:
                        break
                if not i == j:
                    pred_beat_tempo[current_beat - start_beat] = th.mean(
                        pred_x[0, i:j, const.OUTPUT_TEMPO_INDEX])
                    true_beat_tempo[current_beat - start_beat] = th.mean(
                        true_x[0, i:j, const.OUTPUT_TEMPO_INDEX])
            else:
                pred_beat_tempo[current_beat-start_beat] = pred_x[0, i,
                                                                  const.OUTPUT_TEMPO_INDEX:const.OUTPUT_TEMPO_INDEX + const.OUTPUT_TEMPO_PARAM_LEN]
                true_beat_tempo[current_beat-start_beat] = true_x[0, i,
                                                                  const.OUTPUT_TEMPO_INDEX:const.OUTPUT_TEMPO_INDEX + const.OUTPUT_TEMPO_PARAM_LEN]

    tempo_loss = criterion(pred_beat_tempo, true_beat_tempo, model_config)
    if args.deltaLoss and pred_beat_tempo.shape[0] > 1:
        prediction_delta = pred_beat_tempo[1:] - pred_beat_tempo[:-1]
        true_delta = true_beat_tempo[1:] - true_beat_tempo[:-1]
        delta_loss = criterion(prediction_delta, true_delta, model_config)

        tempo_loss = (tempo_loss + delta_loss *
                      args.delta_weight) / (1 + args.delta_weight)

    return tempo_loss


def run_model_in_steps(input, input_y, args, edges, note_locations, model, device, initial_z=False):
    num_notes = input.shape[1]
    with th.no_grad():  # no need to track history in validation
        model_eval = model.eval()
        total_output = []
        
