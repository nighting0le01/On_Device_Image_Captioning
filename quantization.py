import os
import torch
import argparse
from argparse import Namespace
from utils.args_utils import str2list, str2bool

from time import time
import json


from data.vizwiz_dataset import VizWizDataset
from data.vizwiz_dataloader import VizWizDataLoader
from utils import language_utils
from utils.language_utils import compute_num_pads as compute_num_pads
from utils.image_utils import preprocess_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)




    parser.add_argument('--save_model_path', type=str, default='/usr0/home/nvaikunt/On_Device_Image_Captioning/pretrained_weights/4_th.pth')
    parser.add_argument('--eval_beam_sizes', type=str2list, default=[3])
    parser.add_argument('--image_folder', type=str, default="./VizWizData")
    parser.add_argument('--vocab_path', type=str, default="./vocab/coco_vocab_idx_dict.json")

    # parser.add_argument('--pretrain_checkpoint', type=str, default="/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/pretrained_weightscheckpoint_2023-10-12-13:36:34_epoch4it1968bs8_xe_.pth")
    parser.add_argument('--vizwiz', type=str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_accum', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1)

    parser.add_argument('--save_path', type=str, default="/usr0/home/nvaikunt/On_Device_Image_Captioning/pretrained_weights") #default='./github_ignore_material/saves/')
    parser.add_argument('--static', type=str2bool, default=True) #default='./github_ignore_material/saves/')
    parser.add_argument('--calibration_steps', type=int, default=1000)
    parser.add_argument('--static_qconfig', type=str, default="x86")
    
    
    
    args = parser.parse_args()
    args.ddp_sync_port = str(args.ddp_sync_port)

    assert (args.eval_parallel_batch_size % args.num_gpus == 0), \
        "num gpus must be multiple of the requested parallel batch size"

    print("is_ensemble: " + str(args.is_ensemble))
    print("eval parallel batch_size: " + str(args.eval_parallel_batch_size))
    print("ddp_sync_port: " + str(args.ddp_sync_port))
    print("save_model_path: " + str(args.save_model_path))




    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)

    model_args = Namespace(model_dim=args.model_dim,
                           N_enc=args.N_enc,
                           N_dec=args.N_dec,
                           dropout=0.0,
                           drop_args=drop_args,
                           vizwiz = args.vizwiz,
                           image_folder = args.image_folder,
                           param_config = args.param_config
                           )

    print(model_args.param_config)
    
    if args.vizwiz: 
         if os.path.isfile(args.vocab_path):
            with open("./vocab/coco_vocab_idx_dict.json", "r") as vocab_json: 
                coco_vocab_idx_dict = json.load(vocab_json)
         else: 
             coco_vocab_idx_dict = None
         # Currently testing with val_split, normally should set to 1 with train being True
         split = 2
         dataset = VizWizDataset(split, train=False,val = True,coco_vocab_dict=coco_vocab_idx_dict, 
                                 vizwiz_annotations_dir="/usr0/home/nvaikunt/On_Device_Image_Captioning/VizWizData/annotations")
