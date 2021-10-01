########################################################
####### 1. SageMaker Distributed Data Parallel  ########
#######  - Import Package and Initialization    ########
########################################################
import torch
try:
    import smdistributed.dataparallel.torch.distributed as smdp
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as smdpDDP

    if not smdp.is_initialized():
        smdp.init_process_group()
except:
    print("NO SageMaker DP")
    pass
    
    
#######################################################


def dist_set(args):
    args.world_size = smdp.get_world_size()
    args.rank = smdp.get_rank()
    args.local_rank = smdp.get_local_rank()
    print(f"args.local_rank : {args.local_rank }")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    
    args.batch_size //= args.world_size
    args.batch_size = max(args.batch_size, 1)
    return args


def dist_model(model):
    model = smdpDDP(model, broadcast_buffers=False)
#     model = smdpDDP(model)
    return model
    

def barrier():
    return smdp.barrier()