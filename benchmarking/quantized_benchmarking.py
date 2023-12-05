import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
import pickle
import cv2
from argparse import Namespace
import sys
import random

from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.qconfig import default_embedding_qat_qconfig
from torch.ao.quantization import get_default_qconfig_mapping, QConfigMapping, get_default_qat_qconfig_mapping

sys.path.append("/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning")
from models.End_ExpansionNet_v2 import (
    End_ExpansionNet_v2,
    End_ExpansionNet_v2_Encoder,
    End_ExpansionNet_v2_Decoder,
    E2E_ExpansionNet_Captioner,
)

from utils.image_utils import preprocess_image
from utils.language_utils import tokens2description
from utils.quantization_utils import (
    print_size_of_model,
    quantize_encoder_decoder,
    prepare_model,
    quantize_model,
)


# from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
from thop import profile


def compute_FLOPS(encoder, decoder, img_size, sos_idx, eos_idx, beam_size, max_seq_len):
    """
    Current: https://github.com/Lyken17/pytorch-OpCounter

    ## To try DeepSpeed as its better supported:https://www.deepspeed.ai/tutorials/flops-profiler/#example-bert

    """

    class Wrapper(nn.Module):
        def forward(self, inputs):
            return self.captioner(enc_x=inputs, enc_x_num_pads=[0], mode="beam_search")

    input_data = torch.randn(1, 3, img_size, img_size)
    encoder.eval()
    decoder.eval()
    model_wrapped = Wrapper()
    beam_search_kwargs = {
        "beam_size": beam_size,
        "beam_max_seq_len": max_seq_len,
        "sample_or_max": "max",
        "how_many_outputs": 1,
        "sos_idx": sos_idx,
        "eos_idx": eos_idx,
    }
    model_wrapped.captioner = E2E_ExpansionNet_Captioner(
        beam_search_kwargs,
        split_encoder=True,
        encoder=encoder,
        decoder=decoder,
        rank="cpu",
    )
    # input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model_wrapped, inputs=(input_data,))
    print(flops)


def compute_parameters(model, verbose=False):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            if verbose: 
                print(f"Layer: {name}, Parameters: {num_params}")
            total_params += num_params
    print(f"Total Trainable Parameters: {total_params}")


def compute_inference_Latency(
    encoder,
    decoder,
    num_runs,
    img_size,
    coco_tokens,
    sos_idx,
    eos_idx,
    beam_size,
    max_seq_len,
    plots_path,
):
    encoder.eval()
    decoder.eval()
    inference_times = []
    runs = num_runs
    beam_search_kwargs = {
        "beam_size": beam_size,
        "beam_max_seq_len": max_seq_len,
        "sample_or_max": "max",
        "how_many_outputs": 1,
        "sos_idx": sos_idx,
        "eos_idx": eos_idx,
    }
    captioner = E2E_ExpansionNet_Captioner(
        beam_search_kwargs,
        split_encoder=True,
        encoder=encoder,
        decoder=decoder,
        rank="cpu",
    )
    for run in range(runs):
        input_data = torch.randn(1, 3, img_size, img_size).to("cpu")

        t0 = time.perf_counter()
        with torch.no_grad():
            pred, _ = captioner(
                enc_x=input_data, enc_x_num_pads=[0], mode="beam_search"
            )
        end_time = time.perf_counter()
        pred = tokens2description(
            pred[0][0], coco_tokens["idx2word_list"], sos_idx, eos_idx
        )
        inference_times.append(end_time - t0)
        print(f"{run+1}/{runs} runs completed ")
    print(inference_times)
    avg_inference_time = sum(inference_times) / len(inference_times)
    print("Inference time:", avg_inference_time)
    mean_time = np.mean(inference_times)
    variance_time = np.var(inference_times)
    plt.plot(inference_times)
    plt.xlabel("Inference Run")
    plt.ylabel("Inference Time (seconds)")
    plt.title("Inference Time per Run")
    plt.text(5, max(inference_times) * 0.8, f"Mean Time: {mean_time:.10f} seconds")
    plt.text(5, max(inference_times) * 0.7, f"Variance: {variance_time:.10f}")
    plt.savefig(f"{plots_path}/inference_latency.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser("ExpansionNet Benchmarking")
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--compute_train_time",
        action="store_true",
        default=False,
        help="To compute_train_time",
    )
    parser.add_argument(
        "--compute_inference_time",
        action="store_true",
        default=True,
        help="To compute_train_time",
    )
    parser.add_argument(
        "--compute_FLOPS", action="store_true", default=False, help="To Compute FLOPS"
    )
    parser.add_argument(
        "--compute_params",
        action="store_true",
        default=True,
        help="To Compute parameters",
    )
    parser.add_argument(
        "--img_size", type=int, default=384, help=" Image size for Swin Transformer"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed ")
    parser.add_argument(
        "--plot_results",
        action="store_true",
        default=False,
        help="To plot results or not",
    )

    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--N_enc", type=int, default=3)
    parser.add_argument("--N_dec", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=74)
    parser.add_argument(
        "--encoder_load_path",
        type=str,
        default="/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/pretrained_weights/base_QAKD_experiment/checkpoint_base_e4_encoder_.pth",
    )
    parser.add_argument(
        "--decoder_load_path",
        type=str,
        default="/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/pretrained_weights/base_QAKD_experiment/checkpoint_base_e4_decoder_.pth",
    )
    parser.add_argument(
        "--image_paths",
        type=str,
        default=[
            "./demo_material/tatin.jpg",
            "./demo_material/micheal.jpg",
            "./demo_material/napoleon.jpg",
            "./demo_material/cat_girl.jpg",
        ],
        nargs="+",
    )
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument(
        "--plots_path",
        type=str,
        default="./benchmarking/plots",
    )
    parser.add_argument(
        "--model_type", type=str, default="qat", help="Model Type to Load"
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    drop_args = Namespace(enc=0.0, dec=0.0, enc_input=0.0, dec_input=0.0, other=0.0)
    model_args = Namespace(
        model_dim=args.model_dim,
        N_enc=args.N_enc,
        N_dec=args.N_dec,
        dropout=0.0,
        drop_args=drop_args,
    )

    with open("./demo_material/demo_coco_tokens.pickle", "rb") as f:
        coco_tokens = pickle.load(f)
        sos_idx = coco_tokens["word2idx_dict"][coco_tokens["sos_str"]]
        eos_idx = coco_tokens["word2idx_dict"][coco_tokens["eos_str"]]

    encoder_model = End_ExpansionNet_v2_Encoder(
        swin_img_size=args.img_size,
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
        output_word2idx=coco_tokens["word2idx_dict"],
        output_idx2word=coco_tokens["idx2word_list"],
        max_seq_len=args.max_seq_len,
        drop_args=model_args.drop_args,
        rank="cpu",
    )
    decoder_model = End_ExpansionNet_v2_Decoder(
        d_model=512,
        N_enc=3,
        N_dec=3,
        num_heads=8,
        ff=2048,
        num_exp_enc_list=[32, 64, 128, 256, 512],
        num_exp_dec=16,
        output_word2idx=coco_tokens["word2idx_dict"],
        output_idx2word=coco_tokens["idx2word_list"],
        max_seq_len=args.max_seq_len,
        drop_args=model_args.drop_args,
        rank="cpu",
    )

    # Get quantized model structures
    model_type = args.model_type
    if model_type == "static":
        static_qconfig_str = "x86"
        qconfig_mapping = get_default_qconfig_mapping(static_qconfig_str)
    elif model_type == "qat": 
        static_qconfig_str = "x86"
        qconfig_mapping = get_default_qat_qconfig_mapping(
                static_qconfig_str, version=1
        )
            # qconfig_mapping = get_default_qconfig_mapping(static_qconfig_str)
        qconfig_mapping.set_object_type(
                torch.nn.Embedding, default_embedding_qat_qconfig
        )
    else:
        qconfig_mapping = QConfigMapping().set_global(
            torch.ao.quantization.default_dynamic_qconfig
        )
    is_qat = False
    if model_type == "qat":
        is_qat = True
    example_input = (
        torch.randn(1, 3, args.img_size, args.img_size),
        torch.randint(1, 100, (1, 15)),
        [0],
        [0],
    )
    if is_qat: 
        prepared_encoder = prepare_model(encoder_model, example_input, qconfig_mapping, qat=is_qat)
        print("Prepared Encoder Object ...")
        prepared_decoder = prepare_model(decoder_model, example_input, qconfig_mapping, qat=is_qat)
        print("Prepared Decoder Object ...")
        prepared_encoder.load_state_dict(torch.load(args.encoder_load_path, map_location="cpu")["model_state_dict"])
        print("Prepared Encoder Weights loaded ...")
        prepared_decoder.load_state_dict(torch.load(args.decoder_load_path, map_location="cpu")["model_state_dict"])
        print("Prepared Decoder Weights loaded ...")
        encoder_model = quantize_model(prepared_encoder)
        print("Quantized Encoder!")
        decoder_model = quantize_model(prepared_decoder)
        print("Quantized Decoder!")
        encoder_model.eval()
        decoder_model.eval()
    else: 
        prepared_encoder = prepare_model(encoder_model, example_input, qconfig_mapping)
        prepared_decoder = prepare_model(decoder_model, example_input, qconfig_mapping)
        encoder_model = quantize_model(prepared_encoder)
        decoder_model = quantize_model(prepared_decoder)
        encoder_model.load_state_dict(torch.load(args.encoder_load_path))
        print("Encoder loaded ...")
        decoder_model.load_state_dict(torch.load(args.decoder_load_path))
        print("Decoder loaded ...")

    # encoder_model.to("cuda")
    # decoder_model.to("cuda")
    if args.compute_params:
        print("Computing Encoder Params")
        compute_parameters(encoder_model)
        print("Computing Decoder Params")
        compute_parameters(decoder_model)
        print("Add for Total Params")

        print("Printing Model Sizes on Disk")
        print("Encoder Size:")
        print_size_of_model(encoder_model)
        print("Decoder Size:")
        print_size_of_model(decoder_model)

    # if args.compute_FLOPS:
    #     print("Computing FLOPS")
    #     compute_FLOPS(
    #         encoder_model,
    #         decoder_model,
    #         args.img_size,
    #         sos_idx=sos_idx,
    #         eos_idx=eos_idx,
    #         beam_size=args.beam_size,
    #         max_seq_len=args.max_seq_len,
    #     )

    if args.compute_inference_time:
        print("Computing Average Inference Time")
        compute_inference_Latency(
            encoder_model,
            decoder_model,
            num_runs=100,
            img_size=args.img_size,
            coco_tokens=coco_tokens,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            beam_size=args.beam_size,
            max_seq_len=args.max_seq_len,
            plots_path=args.plots_path,
        )
        return


if __name__ == "__main__":
    main()
