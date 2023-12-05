import os
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping
import argparse
from argparse import Namespace
from utils.args_utils import str2list, str2bool
import random

from time import time
import json
from torch.ao.quantization import QConfigMapping
from data.vizwiz_dataset import VizWizDataset
from data.vizwiz_dataloader import VizWizDataLoader

from models.End_ExpansionNet_v2 import (
    End_ExpansionNet_v2_Encoder,
    End_ExpansionNet_v2_Decoder,
    E2E_ExpansionNet_Captioner,
    E2E_ExpansionNet_Captioner,
)

from utils import language_utils
from utils.language_utils import compute_num_pads, tokens2description
from utils.image_utils import preprocess_image
from utils.quantization_utils import (
    calibrate_enc_dec,
    prepare_model,
    quantize_model,
    quantize_encoder_decoder,
    print_size_of_model,
)

encoder_modules = [
    "swin_transf",
    "encoders",
    "input_embedder_dropout",
    "input_linear",
    "out_enc_dropout",
    "pos_encoder",
    "enc_reduce_group",
    "enc_reduce_norm",
    "out_embedder",
]
decoder_modules = [
    "decoders",
    "log_softmax",
    "out_embedder",
    "out_dec_dropout",
    "dec_reduce_group",
    "pos_encoder",
    "dec_reduce_group",
    "dec_reduce_norm",
    "vocab_linear",
]


def filter_state_dict(state_dict, list_to_include):
    new_state_dict = {}
    for key in state_dict.keys():
        valid_key = False
        for entry in list_to_include:
            if entry in key:
                valid_key = True
        if valid_key:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def load_models(
    model_args: Namespace,
    dataset,
    model_max_len: int,
    img_size: int = 384,
    device: str = "cpu",
):
    encoder_model = End_ExpansionNet_v2_Encoder(
        swin_img_size=img_size,
        swin_patch_size=4,
        swin_in_chans=3,
        swin_embed_dim=192,
        swin_depths=[2, 2, 18, 2],
        swin_num_heads=[6, 12, 24, 48],
        swin_window_size=12,
        swin_mlp_ratio=4.0,
        swin_qkv_bias=True,
        swin_qk_scale=None,
        swin_drop_rate=0.0,
        swin_attn_drop_rate=0.0,
        swin_drop_path_rate=0.1,
        swin_norm_layer=torch.nn.LayerNorm,
        swin_ape=False,
        swin_patch_norm=True,
        swin_use_checkpoint=False,
        final_swin_dim=1536,
        d_model=model_args.model_dim,
        N_enc=model_args.N_enc,
        N_dec=model_args.N_dec,
        num_heads=8,
        ff=2048,
        num_exp_enc_list=[32, 64, 128, 256, 512],
        num_exp_dec=16,
        output_word2idx=dataset.caption_word2idx_dict,
        output_idx2word=dataset.caption_idx2word_list,
        max_seq_len=model_max_len,
        drop_args=model_args.drop_args,
        rank=device,
    )
    decoder_model = End_ExpansionNet_v2_Decoder(
        d_model=512,
        N_enc=3,
        N_dec=3,
        num_heads=8,
        ff=2048,
        num_exp_enc_list=[32, 64, 128, 256, 512],
        num_exp_dec=16,
        output_word2idx=dataset.caption_word2idx_dict,
        output_idx2word=dataset.caption_idx2word_list,
        max_seq_len=model_max_len,
        drop_args=drop_args,
        rank=device,
    )

    return encoder_model, decoder_model


def demo_quantized_model(encoder, decoder, demo_image_path, idx2word, sos_idx, eos_idx, device="cpu"):
    #demo_image_path = "./demo_material/micheal.jpg"
    img_size = 384
    demo_image = preprocess_image(demo_image_path, img_size)

    beam_search_arg_defaults = {
        "beam_size": 5,
        "beam_max_seq_len": 40,
        "sample_or_max": "sample",
        "how_many_outputs": 1,
        "sos_idx": sos_idx,
        "eos_idx": eos_idx,
    }
    # encoder.to(device)
    # decoder.to(device)

    captioner = E2E_ExpansionNet_Captioner(
        beam_search_arg_defaults,
        split_encoder=True,
        encoder=encoder,
        decoder=decoder,
        rank=device,
    )
    with torch.no_grad():
        pred, _ = captioner(
            enc_x=demo_image.to(device), enc_x_num_pads=[0], mode="beam_search"
        )

    pred = tokens2description(
        pred[0][0], idx2word, sos_idx, eos_idx
    )
    print(" \n\tDescription: " + pred + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning")
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--N_enc", type=int, default=3)
    parser.add_argument("--N_dec", type=int, default=3)

    parser.add_argument(
        "--save_model_path",
        type=str,
        default="/usr0/home/nvaikunt/On_Device_Image_Captioning/pretrained_weights/base/4_th.pth",
    )
    parser.add_argument("--eval_beam_sizes", type=str2list, default=[3])
    parser.add_argument("--image_folder", type=str, default="./VizWizData")
    parser.add_argument(
        "--vocab_path", type=str, default="./vocab/coco_vocab_idx_dict.json"
    )

    # parser.add_argument('--pretrain_checkpoint', type=str, default="/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/pretrained_weightscheckpoint_2023-10-12-13:36:34_epoch4it1968bs8_xe_.pth")
    parser.add_argument("--vizwiz", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_accum", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument(
        "--save_path",
        type=str,
        default="/usr0/home/nvaikunt/On_Device_Image_Captioning/pretrained_weights",
    )  # default='./github_ignore_material/saves/')
    parser.add_argument(
        "--static", type=str2bool, default=False
    )  # default='./github_ignore_material/saves/')
    parser.add_argument(
        "--qat", type=str2bool, default=False
    )  # default='./github_ignore_material/saves/')
    parser.add_argument("--calibration_steps", type=int, default=1000)
    parser.add_argument("--static_qconfig", type=str, default="x86")
    parser.add_argument("--demo", type=str2bool, default=False)

    args = parser.parse_args()

    print("save_model_path: " + str(args.save_model_path))

    drop_args = Namespace(enc=0.0, dec=0.0, enc_input=0.0, dec_input=0.0, other=0.0)

    model_args = Namespace(
        model_dim=args.model_dim,
        N_enc=args.N_enc,
        N_dec=args.N_dec,
        dropout=0.0,
        drop_args=drop_args,
        vizwiz=args.vizwiz,
        image_folder=args.image_folder,
    )

    quant_args = Namespace(
        static=args.static,
        calibration_steps=args.calibration_steps,
        static_qconfig_str=args.static_qconfig,
    )

    dataset = None
    if args.vizwiz:
        if os.path.isfile(args.vocab_path):
            with open("./vocab/coco_vocab_idx_dict.json", "r") as vocab_json:
                coco_vocab_idx_dict = json.load(vocab_json)
        else:
            coco_vocab_idx_dict = None
        # Currently testing with val_split, normally should set to 1 with train being True
        split = 2
        dataset = VizWizDataset(
            split,
            train=False,
            val=True,
            coco_vocab_dict=coco_vocab_idx_dict,
            vizwiz_annotations_dir="/usr0/home/nvaikunt/On_Device_Image_Captioning/VizWizData/annotations",
        )
    ckpt_path = args.save_model_path
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    state_dict = checkpoint["model_state_dict"]
    del checkpoint

    model_max_len = dataset.max_seq_len + 20
    img_size = 384
    # device = "cpu"
    device = args.device
    beam_search_arg_defaults = {
        "sos_idx": dataset.get_sos_token_idx(),
        "eos_idx": dataset.get_eos_token_idx(),
        "beam_size": 5,
        "beam_max_seq_len": model_max_len,
        "sample_or_max": "max",
        "how_many_outputs": 1,
    }
    encoder_model, decoder_model = load_models(
        model_args, dataset, model_max_len, img_size=img_size, device=device
    )

    encoder_state_dict = filter_state_dict(state_dict, encoder_modules)
    decoder_state_dict = filter_state_dict(state_dict, decoder_modules)
    encoder_model.load_state_dict(encoder_state_dict)
    decoder_model.load_state_dict(decoder_state_dict)

    encoder_model.to(device)
    decoder_model.to(device)
    image_folder = args.image_folder
    array_of_init_seeds = [random.random() for _ in range(1 * 2)]
    data_loader = VizWizDataLoader(
        vizwiz_dataset=dataset,
        batch_size=4,
        num_procs=1,
        array_of_init_seeds=array_of_init_seeds,
        dataloader_mode="caption_wise",
        resize_image_size=img_size,
        rank=device,
        image_folder=image_folder,
        verbose=True,
    )
    if quant_args.static:
        static_qconfig_str = quant_args.static_qconfig_str
        qconfig_mapping = get_default_qconfig_mapping(static_qconfig_str)
    else:
        qconfig_mapping = QConfigMapping().set_global(
            torch.ao.quantization.default_dynamic_qconfig
        )

    quantized_encoder, quantized_decoder = quantize_encoder_decoder(
        encoder_model,
        decoder_model,
        data_loader,
        3,
        qconfig_mapping,
        device,
        static=quant_args.static,
    )
    # Save models
    orig_file_name = ckpt_path.split("/")[-1]
    if args.static:
        model_type = "static"
    else:
        model_type = "dynamic"
    encoder_save_file = f"{model_type}_quantized_encoder_{orig_file_name}"
    decoder_save_file = f"{model_type}_quantized_decoder_{orig_file_name}"
    torch.save(
        quantized_encoder.state_dict(), os.path.join(args.save_path, encoder_save_file)
    )
    torch.save(
        quantized_decoder.state_dict(), os.path.join(args.save_path, decoder_save_file)
    )

    # Print Info
    print_size_of_model(encoder_model)
    print_size_of_model(decoder_model)
    print_size_of_model(quantized_encoder)
    print_size_of_model(quantized_decoder)

    if args.demo:
        demo_quantized_model(
            quantized_encoder,
            quantized_decoder,
            sos_idx=dataset.get_sos_token_idx(),
            eos_idx=dataset.get_eos_token_idx(),
        )
