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

sys.path.append("/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning")
from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.image_utils import preprocess_image
from utils.language_utils import tokens2description

# from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
from thop import profile


def compute_FLOPS(model, img_size, sos_idx, eos_idx, beam_size, max_seq_len):
    """
    Current: https://github.com/Lyken17/pytorch-OpCounter

    ## To try DeepSpeed as its better supported:https://www.deepspeed.ai/tutorials/flops-profiler/#example-bert

    """

    class Wrapper(nn.Module):
        def forward(self, inputs):
            beam_search_kwargs = {
                "beam_size": beam_size,
                "beam_max_seq_len": max_seq_len,
                "sample_or_max": "max",
                "how_many_outputs": 1,
                "sos_idx": sos_idx,
                "eos_idx": eos_idx,
            }
            return self.model(
                enc_x=input_data,
                enc_x_num_pads=[0],
                mode="beam_search",
                **beam_search_kwargs,
            )

    input_data = torch.randn(1, 3, img_size, img_size)
    model.eval()
    model_wrapped = Wrapper()
    model_wrapped.model = model
    # input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model_wrapped, inputs=(input_data,))
    print(flops)


def compute_parameters(model):
    print(model)
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(f"Layer: {name}, Parameters: {num_params}")
            total_params += num_params
    print(f"Total Trainable Parameters: {total_params}")


def compute_inference_Latency(
    model,
    num_runs,
    img_size,
    coco_tokens,
    sos_idx,
    eos_idx,
    beam_size,
    max_seq_len,
    plots_path,
    device,
):
    model = model.to(device)
    model.eval()
    inference_times = []
    runs = num_runs
    for run in range(runs):
        input_data = torch.randn(1, 3, img_size, img_size).to(device)
        beam_search_kwargs = {
            "beam_size": beam_size,
            "beam_max_seq_len": max_seq_len,
            "sample_or_max": "max",
            "how_many_outputs": 1,
            "sos_idx": sos_idx,
            "eos_idx": eos_idx,
        }
        t0 = time.perf_counter()
        with torch.no_grad():
            pred, _ = model(
                enc_x=input_data,
                enc_x_num_pads=[0],
                mode="beam_search",
                **beam_search_kwargs,
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
        default=False,
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
        "--load_path",
        type=str,
        default="On_Device_Image_Captioning/pretrained_weights/rf_model.pth",
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
        default="On_Device_Image_Captioning/benchmarking/plots",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
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

    with open(
        "On_Device_Image_Captioning/demo_material/demo_coco_tokens.pickle", "rb"
    ) as f:
        coco_tokens = pickle.load(f)
        sos_idx = coco_tokens["word2idx_dict"][coco_tokens["sos_str"]]
        eos_idx = coco_tokens["word2idx_dict"][coco_tokens["eos_str"]]

    model = End_ExpansionNet_v2(
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
        swin_drop_path_rate=0.0,
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
        rank=args.device,
    )

    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded ...")

    if args.compute_params:
        print("Computing params")
        compute_parameters(model)

    if args.compute_FLOPS:
        print("Computing FLOPS")
        compute_FLOPS(
            model,
            args.img_size,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            beam_size=args.beam_size,
            max_seq_len=args.max_seq_len,
        )

    if args.compute_inference_time:
        print("Computing Average Inference Time")
        print(model)
        compute_inference_Latency(
            model=model,
            num_runs=100,
            img_size=args.img_size,
            coco_tokens=coco_tokens,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            beam_size=args.beam_size,
            max_seq_len=args.max_seq_len,
            plots_path=args.plots_path,
            device=args.device,
        )
        return


if __name__ == "__main__":
    main()
