import os
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
from copy import deepcopy
import torch.nn.utils.prune as prune

sys.path.append("/usr0/home/nvaikunt/On_Device_Image_Captioning")
print(sys.path)
from legacy_models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.image_utils import preprocess_image
from utils.language_utils import tokens2description
from utils.quantization_utils import print_size_of_model

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


def compute_parameters(model, verbose=False):
    if verbose: 
        print(model)
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            if verbose: 
                print(f"Layer: {name}, Parameters: {num_params}")
            total_params += num_params
    print(f"Total Trainable Parameters: {total_params}")
    return total_params


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


def calculate_sparsity(buffer):
    total_zeros = 0
    total_params = 0
    for x, y in buffer:
        curr_zeros = float(torch.sum(y == 0))
        curr_params = torch.flatten(y).shape[0]
        total_zeros += curr_zeros
        total_params += curr_params
        sp = (curr_zeros / curr_params) * 100
        print(f"Sparsity in {x} is {sp:.2f}%")
    print(
        f"Overall Sparsity of the pruned parameters is {(total_zeros/total_params)*100:.2f}%"
    )


def calculate_pruned_size(model, checkpoint_name, weight_names):
    for i in range(0, len(weight_names)):
        prune.remove(weight_names[i][0], "weight")
    sd = model.state_dict()
    for item in sd:
        sd[item] = model.state_dict()[item].to_sparse()
    torch.save(sd, f"On_Device_Image_Captioning/pruned_weights/{checkpoint_name}.pth")
    print(
        f'Size of the pruned model: {os.path.getsize(f"On_Device_Image_Captioning/pruned_weights/{checkpoint_name}.pth")/1e6} MB'
    )


def prune_model(model, n_prune, checkpoint_name):
    c_model = deepcopy(model)
    weight_names = [
        (m[1], "weight")
        for m in c_model.named_modules()
        if len(list(m[1].children())) == 0
        and not isinstance(
            m[1],
            (nn.Dropout, nn.Sigmoid, nn.GELU, nn.Identity, nn.Softmax, nn.LogSoftmax),
        )
    ]
    for _ in range(n_prune):
        prune.global_unstructured(
            weight_names, pruning_method=prune.L1Unstructured, amount=0.33
        )
    buffer = list(c_model.named_buffers())
    calculate_sparsity(buffer)
    calculate_pruned_size(c_model, checkpoint_name, weight_names)


def prune_attn_params(q, k, v, heads_to_prune, total_heads):
    in_channels = v.shape[-1]
    q_heads = q.reshape(total_heads, -1, in_channels)
    k_heads = k.reshape(total_heads, -1, in_channels)
    v_heads = v.reshape(total_heads, -1, in_channels)
    importances = torch.norm(v_heads, dim=(1, 2))
    # print(f"Norms: {importances}")
    _, bottom_idx = torch.topk(-importances, k=heads_to_prune)
    # print(f"Heads to cut: {bottom_idx}")
    q_heads[bottom_idx] = 0
    k_heads[bottom_idx] = 0
    v_heads[bottom_idx] = 0

    return q_heads.reshape(-1, in_channels), k_heads.reshape(-1, in_channels), v_heads.reshape(-1, in_channels)


def structured_head_pruning(state_dict, num_heads=[6, 12, 24, 48], prune_pct=(1 / 3), prune_only = None):
    num_params_pruned = 0
    for name, param in state_dict.items():
        if "attn.qkv.weight" in name:

            # print(name)
            # print(f"Orig_Shape: {param.shape}")
            layer_num = name.split(".")[2]
            in_channels = param.shape[-1]
            if prune_only is None or prune_only == layer_num: 
                q, k, v = param.reshape(3, in_channels, in_channels)
                total_heads = num_heads[int(layer_num)]
                heads_to_prune = int(total_heads * prune_pct)
                # print(f"Heads to Prune, Total Heads: {heads_to_prune},{total_heads}")
                # print(f"Query Matrix Shape {q.shape}")

                pruned_q, pruned_k, pruned_v = prune_attn_params(q, k, v, heads_to_prune, total_heads)
                pruned_qkv = torch.cat((pruned_q, pruned_k, pruned_v), dim=0)
                # print(f"New_Shape: {pruned_qkv.shape}")
                num_params_pruned += prune_pct * param.shape[0] * param.shape[1]
                state_dict[name] = pruned_qkv
    print(f"Total_Pruned: {num_params_pruned}")
    return state_dict, num_params_pruned

def structured_pruning_stats(pruned_state_dict, pruned_params, total_params, ckpt_name="pruned_rf.pth"):
    print(f"Sparsity of the model is {pruned_params / total_params}")
    copied_state_dict = {}
    for key in pruned_state_dict: 
        if "attn.qkv.weight" in key:
            copied_state_dict[key] = pruned_state_dict[key].to_sparse()
        else: 
            copied_state_dict[key] = pruned_state_dict[key]
    torch.save(copied_state_dict, ckpt_name)
    print(
        f'Size of the pruned model: {os.path.getsize(ckpt_name)/1e6} MB'
    )
    os.remove(ckpt_name)
    

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
        default=False,
        help="To Compute parameters",
    )
    parser.add_argument(
        "--prune", action="store_true", default=False, help="To Prune the model"
    )
    parser.add_argument(
        "--structured_prune", action="store_true", default=False, help="Structured Pruning and Stats"
    )
    parser.add_argument(
        "--prune_count", type=int, default=1, help="No. of times to prune the model"
    )
    parser.add_argument(
        "--load_pruned_model",
        action="store_true",
        default=False,
        help="To load the sparsed pruned weights in the model",
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
        default="On_Device_Image_Captioning/pretrained_weights/base/4_th.pth",
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
        "../On_Device_Image_Captioning/demo_material/demo_coco_tokens.pickle", "rb"
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

    if args.load_pruned_model:
        sparse_checkpoint = torch.load(args.load_path)
        model.load_state_dict(
            {
                k: (v if v.layout == torch.strided else v.to_dense())
                for k, v in sparse_checkpoint.items()
            }
        )
        print("Model loaded with pruned weights ...")
    else:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded ...")
        if args.structured_prune: 
            total_params = compute_parameters(model)
            print_size_of_model(model)
            pruned_param_dict, num_pruned = structured_head_pruning(checkpoint["model_state_dict"], prune_pct=(1/3))
            structured_pruning_stats(pruned_state_dict=pruned_param_dict, pruned_params=num_pruned, total_params=total_params)

    if args.prune:
        print("Pruning")
        prune_model(model, args.prune_count, f"prune_{args.prune_count}")
    

    if args.compute_params:
        print("Computing params")
        compute_parameters(model, verbose=True)

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
