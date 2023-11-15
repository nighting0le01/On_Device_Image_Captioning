import os
import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping
import argparse
from argparse import Namespace
from utils.args_utils import str2list, str2bool
import random
from torch2trt import torch2trt
from time import time
import json
from torch.ao.quantization import QConfigMapping
from data.vizwiz_dataset import VizWizDataset
from data.vizwiz_dataloader import VizWizDataLoader
import deeplearning.trt.fx2trt.converter.converters
from torch.fx.experimental.fx2trt.fx2trt import InputTensorSpec, TRTInterpreter
from torch_tensorrt.fx import TRTModule

from models.End_ExpansionNet_v2 import (
    End_ExpansionNet_v2_Encoder,
    End_ExpansionNet_v2_Decoder,
    E2E_ExpansionNet_Captioner
)

from utils import language_utils
from utils.language_utils import compute_num_pads, tokens2description
from utils.image_utils import preprocess_image
from utils.quantization_utils import (
    calibrate_enc_dec,
    prepare_model,
    quantize_model,
    quantize_encoder_decoder,
    print_size_of_model
)

    
def convert2TRT(encoder_model,decoder_model,
                img_size, sos_idx, eos_idx, 
                device,beam_search_arg_defaults,
                dataset):

    demo_image_path = "./demo_material/micheal.jpg"
    demo_image = preprocess_image(demo_image_path, img_size)
    
    
    example_input = [(
    torch.randn(1, 3, img_size, img_size), 
    torch.randint(1, 100, (1, 15)),
    [0],
    [0])]
    acc_mod_encoder = acc_tracer.trace(encoder_model, example_input)
    acc_mod_decoder = acc_tracer.trace(decoder_model, example_input)

    inputs = [example_input]
    input_specs = InputTensorSpec.from_tensors(inputs)
    interpreter_encoder = TRTInterpreter(
    acc_mod_encoder, input_specs, explicit_batch_dimension=True
    )
    interpreter_decoder = TRTInterpreter(
    acc_mod_decoder, input_specs, explicit_batch_dimension=True
    )
    trt_interpreter_result_enc = interpreter_encoder.run(
                            max_batch_size=1,
                            max_workspace_size=1 << 25,
                            sparse_weights=False,
                            force_fp32_output=False,
                            strict_type_constraints=False,
                            algorithm_selector=None,
                            timing_cache=None,
                            profiling_verbosity=None,
                        )
    trt_interpreter_result_dec = interpreter_decoder.run(
                            max_batch_size=1,
                            max_workspace_size=1 << 25,
                            sparse_weights=False,
                            force_fp32_output=False,
                            strict_type_constraints=False,
                            algorithm_selector=None,
                            timing_cache=None,
                            profiling_verbosity=None,
                        )
    
    mod_enc = TRTModule(
            trt_interpreter_result_enc.engine,
            trt_interpreter_result_enc.input_names,
            trt_interpreter_result_enc.output_names)
    mod_dec = TRTModule(
            trt_interpreter_result_dec.engine,
            trt_interpreter_result_dec.input_names,
            trt_interpreter_result_dec.output_names)
    # Just like all other PyTorch modules
    inputs = [(
    torch.randn(1, 3, img_size, img_size), 
    torch.randint(1, 100, (1, 15)),
    [0],
    [0])]
    
    # outputs_enc = mod_enc(*inputs)
    # torch.save(mod_enc, "mod_enc_trt.pt")
    # reload_trt_mod_enc = torch.load("mod_enc_trt.pt")
    # reload_model_output_enc = reload_trt_mod_enc(*inputs)
    
    # outputs_dec = mod_dec(*inputs)
    # torch.save(mod_dec, "mod_dec_trt.pt")
    # reload_trt_mod_dec = torch.load("mod_dec_trt.pt")
    # reload_model_output_dec = reload_trt_mod_dec(*inputs)
    
    captioner = E2E_ExpansionNet_Captioner(beam_search_arg_defaults, split_encoder=True, encoder=mod_enc,
                                            decoder=mod_dec, rank=device)
    with torch.no_grad():
        pred, _ = captioner(enc_x=demo_image.to(device),
                            enc_x_num_pads=[0], mode="beam_search")

    pred = tokens2description(pred[0][0], dataset.caption_idx2word_list, sos_idx, eos_idx)
    print(' \n\tDescription: ' + pred + '\n')



def main():
    parser = argparse.ArgumentParser("ExpansionNet Quantization Testing")

    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--N_enc", type=int, default=3)
    parser.add_argument("--N_dec", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=74)
    parser.add_argument("--seed", type=int, default=42, help="random seed ")
    parser.add_argument(
        "--encoder_load_path",
        type=str,
        default="./pretrained_weights/dynamic_quantized_encoder_rf_model.pth",
    )
    parser.add_argument(
        "--decoder_load_path",
        type=str,
        default="./pretrained_weights/dynamic_quantized_decoder_rf_model.pth",
    )
    parser.add_argument("--image_folder", type=str, default="./VizWizData")
    parser.add_argument(
        "--vocab_path", type=str, default="./vocab/coco_vocab_idx_dict.json"
    )
    parser.add_argument("--vizwiz", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument(
        "--img_size", type=int, default=384, help=" Image size for Swin Transformer"
    )
    parser.add_argument("--device", type=str, default="cpu")
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
        output_word2idx=dataset.caption_word2idx_dict,
        output_idx2word=dataset.caption_idx2word_list,
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
        output_word2idx=dataset.caption_word2idx_dict,
        output_idx2word=dataset.caption_idx2word_list,
        max_seq_len=args.max_seq_len,
        drop_args=model_args.drop_args,
        rank="cpu",
    )


    # Get quantized model structures
    model_type = args.encoder_load_path.split("/")[-1].split("_")[0]
    if model_type == "static":
        static_qconfig_str = "x86"
        qconfig_mapping = get_default_qconfig_mapping(static_qconfig_str)
    else:
 
        qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)
    
    example_input = (
        torch.randn(1, 3, args.img_size, args.img_size), 
        torch.randint(1, 100, (1, 15)),
        [0],
        [0]
    )
    prepared_encoder = prepare_model(encoder_model, example_input, qconfig_mapping)
    prepared_decoder = prepare_model(decoder_model, example_input, qconfig_mapping)
    encoder_model = quantize_model(prepared_encoder)
    decoder_model = quantize_model(prepared_decoder)
    encoder_model.load_state_dict(torch.load(args.encoder_load_path))
    print("Encoder loaded ...")
    decoder_model.load_state_dict(torch.load(args.decoder_load_path))
    print("Decoder loaded ...")


    image_folder = args.image_folder
    array_of_init_seeds = [random.random() for _ in range(1 * 2)]
    data_loader = VizWizDataLoader(vizwiz_dataset=dataset,
                                   batch_size=4,
                                   num_procs=1,
                                   array_of_init_seeds=array_of_init_seeds,
                                   dataloader_mode='caption_wise',
                                   resize_image_size=args.img_size,
                                   rank=args.device,
                                   image_folder=image_folder,
                                   verbose=True)
    model_max_len = dataset.max_seq_len + 20
    print("DataLoader initialized ...")
    beam_search_arg_defaults = {'sos_idx': dataset.get_sos_token_idx(),
                                'eos_idx': dataset.get_eos_token_idx(),
                                'beam_size': 5,
                                'beam_max_seq_len': model_max_len,
                                'sample_or_max': 'max',
                                'how_many_outputs': 1, }
    convert2TRT(encoder_model=encoder_model,decoder_model=decoder_model,
                img_size=args.img_size,sos_idx=dataset.get_sos_token_idx(), 
                eos_idx=dataset.get_eos_token_idx(), device=args.device,beam_search_arg_defaults =beam_search_arg_defaults,dataset = dataset)


if __name__ == "__main__":
    main()

    
    
