import sys
from pathlib import Path

from torch import distributed, nn
from torch.nn.parallel.distributed import DistributedDataParallel
import torch as th
import _pickle as cPickle
import random

from .parser import get_parser, get_name
from . import model as modelzoo
from . import model_parameters as param
from . import train
from . import test
from . import utils
from .dataset import PerformDataset
from . import model_constants


def main():
    parser = get_parser()
    random.seed(0)
    args = parser.parse_args()
    name = args.modelCode

    data_path = args.data_path
    train_data = PerformDataset(data_path, split='train')
    valid_data = PerformDataset(data_path, split='valid')
    test_data = PerformDataset(data_path, split='test')

    criterion = utils.criterion

    with open(data_path+'dataset_info.dat', 'rb') as f:
        u = cPickle.Unpickler(f)
        dataset_info = u.load()
    feature_stats = dataset_info['stats']
    index_dict = dataset_info['index_dict']
    constants = model_constants.initialize_model_constants(index_dict)
    
    print(f"Experiment {name}")

    if args.device is None:
        device = "cpu"
        if th.cuda.is_available():
            device = "cuda"
    else:
        device = args.device

    th.manual_seed(args.seed)
    
    if args.world_size > 1:
        if device != "cuda" and args.rank == 0:
            print("Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        th.cuda.set_device(args.rank % th.cuda.device_count())
        distributed.init_process_group(backend="nccl",
                                       init_method="tcp://" + args.master,
                                       rank=args.rank,
                                       world_size=args.world_size)


    if args.sessMode == 'train' and not args.resumeTraining:
        model_config = param.initialize_model_parameters_by_code(args, constants)
        model_config.training_args = args
        param.save_parameters(model_config, args.parameters_folder, args.modelCode + '_param')
    elif args.resumeTraining: # default = False
        model_config = param.load_parameters(args.parameters_folder, args.modelCode + '_param')
    else: 
        model_config = param.load_parameters(args.parameters_folder,
            args.modelCode + '_param')  # default = han_ar
        TrillNET_Param = param.load_parameters(args.parameters_folder,
            args.trillCode + '_param')  # default = trill_default
        TRILL_MODEL = modelzoo.TrillRNN(TrillNET_Param, device, constants).to(device)

    if 'han' in args.modelCode:
        if 'ar' in args.modelCode:
            step_by_step = True
        else:
            step_by_step = False
        MODEL = modelzoo.HAN_Integrated(model_config, device, constants, step_by_step).to(device)
    elif 'trill' in args.modelCode:
        MODEL = modelzoo.TrillRNN(model_config, device, constants).to(device)
    else:
        print('Error: Unclassified model code')

    optimizer = th.optim.Adam(
        MODEL.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    

    if args.sessMode == 'train' and not args.resumeTraining:
        train.train(args,
                    MODEL,
                    train_data,
                    valid_data,
                    device,
                    optimizer, 
                    None,  # TODO: bins: what should be?
                    criterion,
                    constants)

    if args.sessMode == 'test' :
        test.test(args,
                  test_data,
                  MODEL,
                  TRILL_MODEL,
                  device,
                  feature_stats)


if __name__ == "__main__":
    main()
