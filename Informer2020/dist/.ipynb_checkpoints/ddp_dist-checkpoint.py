########################################################
####### 1. SageMaker Distributed Data Parallel  ########
#######  - Import Package and Initialization    ########
########################################################
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

    
#######################################################
def dist_init(args):
    mp.spawn(fn, nprocs=args.num_gpus, args=(args, ))

def dist_set(args):
    args.world_size = len(args.hosts) * args.num_gpus
    if args.local_rank is not None:
        args.rank = args.num_gpus * args.host_num + \
            args.local_rank  # total rank in all hosts

    dist.init_process_group(backend=args.backend,
                            rank=args.rank,
                            world_size=args.world_size)
    print(f"args.local_rank : {args.local_rank }")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    
    args.batch_size //= args.world_size
    args.batch_size = max(args.batch_size, 1)
    return args


def dist_model(model):
    model = DDP(model)
    return model
