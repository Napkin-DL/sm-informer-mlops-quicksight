from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

print("****************************************")

import numpy as np
import json
import os
import csv

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import pandas as pd
import subprocess

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def model_fn(model_dir):
    global args
    
    res = dir_info(model_dir)
    print(f"********** res_model_dir : {res}")

    for (root, dirs, files) in os.walk(model_dir):
        for file in files:
            if file in ['training_config.npy']:
                config_file = os.path.join(root, file)

    config = np.load(config_file, allow_pickle=True)
    args = config.tolist()
    
    model_dict = {
        'informer':Informer,
        'informerstack':InformerStack,
    }

    if args.model=='informer' or args.model=='informerstack':
        e_layers = args.e_layers if args.model=='informer' else args.s_layers
        model = model_dict[args.model](
            args.enc_in,
            args.dec_in, 
            args.c_out, 
            args.seq_len, 
            args.label_len,
            args.pred_len, 
            args.factor,
            args.d_model, 
            args.n_heads, 
            e_layers, # self.args.e_layers,
            args.d_layers, 
            args.d_ff,
            args.dropout, 
            args.attn,
            args.embed,
            args.freq,
            args.activation,
            args.output_attention,
            args.distil,
            args.mix,
        ).float()
    
    print(f" ************* model_dir : {model_dir}, args.setting : {args.setting}")
    
    with open(os.path.join(model_dir, args.setting, "checkpoint.pth"), 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    print("Informer Model loaded")
    model.eval()
    model = model.to(device)
    return model


def dir_info(path):
    for (root, dirs, files) in os.walk(path):
        print(f"root : {root}, dir : {dirs}, files : {files}")


def input_fn(data_dir, input_content_type):
    res = dir_info(data_dir)
    print(f"********** res_input_fn : {res}")
    f = open("predict_data.txt", 'w')
    f.write(data_dir)
    f.close()

    res = None
    return res

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(res, model):
    default_bucket = os.environ['default_bucket']
    print(f"default_bucket : {default_bucket}")
    predict_dataset = f"s3://{default_bucket}/predict_dataset/"
    cmd = ["aws", "s3", "cp", "predict_data.txt", predict_dataset]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()    
       
    data_dict = {
        'ETTh1':Dataset_ETT_hour,
        'ETTh2':Dataset_ETT_hour,
        'ETTm1':Dataset_ETT_minute,
        'ETTm2':Dataset_ETT_minute,
        'WTH':Dataset_Custom,
        'ECL':Dataset_Custom,
        'Solar':Dataset_Custom,
        'custom':Dataset_Custom,
    }
    Data = data_dict[args.data]
    timeenc = 0 if args.embed!='timeF' else 1

    shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
    Data = Dataset_Pred

    pred_data = Data(
        root_path="./",
        data_path="predict_data.txt",
        flag="pred",
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        inverse=args.inverse,
        timeenc=timeenc,
        freq=freq,
        cols=args.cols
    )

    pred_loader = DataLoader(
        pred_data,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        sampler=None,
        drop_last=drop_last)      

    #     pred_data = res['data_set']
    #     pred_loader = res['data_loader']
    preds = []
    trues = []
    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, r_begin) in enumerate(pred_loader):
#         print(batch_x.shape,batch_y.shape,batch_x_mark.shape,batch_y_mark.shape,r_begin)
        pred, true = _process_one_batch(
            pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark, model)

        pred = pred_data.scaler.inverse_transform(pred)
    #     true = pred_data.scaler.inverse_transform(true)
        preds.append(pred.detach().cpu().numpy())
    #     trues.append(true.detach().cpu().numpy())
        if i == 0:
            start_point = r_begin.item()

        end_point = r_begin.item()+pred_data.label_len

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])    

    prediction = preds[0,:,-1]

    # trues = np.array(trues)
    # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])    

    # gt = trues[0,:,-1]

    pred_timestamp = pred_data.df_stamp
    pred_prediction = pred_data.prediction_data

    pred_dataset = pd.concat([pred_timestamp, pred_prediction], axis=1)
    pred_dataset = pred_dataset.reset_index()
    pred_dataset.drop('index', axis=1, inplace=True)

    start=pred_dataset.shape[0]-args.pred_len
    pred_result=pred_dataset.loc[start:].reset_index()
    final_result = pd.concat([pred_result, pd.DataFrame(prediction, columns=['Prediction'])], axis=1)
    final_result.drop('index', axis=1, inplace=True)
    
#     print(f"final_result : {final_result}")
    final_result.to_csv("./prediction_result.csv")
    
    
    result_repo = f"s3://{default_bucket}/poc_informer/batch_result/"
    cmd = ["aws", "s3", "cp", "prediction_result.csv", result_repo]
    print(f"Syncing files from prediction_result.csv to {result_repo}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()  
    
    return prediction


def _process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, model):
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    # decoder input
    if args.padding==0:
        dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
    elif args.padding==1:
        dec_inp = torch.ones([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
    dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float().to(device)
    # encoder - decoder
    if args.use_amp:
        with torch.cuda.amp.autocast():
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    else:
        if args.output_attention:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    if args.inverse:
        outputs = dataset_object.inverse_transform(outputs)
    f_dim = -1 if args.features=='MS' else 0
    batch_y = batch_y[:,-args.pred_len:,f_dim:].to(device)

    return outputs, batch_y