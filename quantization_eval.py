import torch
import math
from argparse import Namespace
import json
import random
import sys
import argparse
from utils.args_utils import str2list, str2bool
import pickle
from tqdm import tqdm
from time import time
from utils import language_utils
from utils.language_utils import compute_num_pads as compute_num_pads
from eval.eval import COCOEvalCap
from data.vizwiz_dataset import VizWizDataset
from data.vizwiz_dataloader import VizWizDataLoader
from models.End_ExpansionNet_v2 import (
    End_ExpansionNet_v2,
    End_ExpansionNet_v2_Encoder,
    End_ExpansionNet_v2_Decoder,
    E2E_ExpansionNet_Captioner,
)
from utils.quantization_utils import (
    print_size_of_model,
    quantize_encoder_decoder,
    prepare_model,
    quantize_model,
)
from quantization import demo_quantized_model
from torch.ao.quantization.qconfig import default_embedding_qat_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping, QConfigMapping, get_default_qat_qconfig_mapping
import os


def compute_quantized_evaluation_loss(
    loss_function,
    encoder,
    decoder,
    data_set,
    data_loader,
    num_samples,
    sub_batch_size,
    dataset_split,
    rank=0,
    verbose=False,
):
    encoder.eval()
    decoder.eval()

    sb_size = sub_batch_size

    tot_loss = 0
    num_sub_batch = math.ceil(num_samples / sb_size)
    tot_num_tokens = 0
    for sb_it in range(num_sub_batch):
        from_idx = sb_it * sb_size
        to_idx = min((sb_it + 1) * sb_size, num_samples)

        (
            sub_batch_input_x,
            sub_batch_target_y,
            sub_batch_input_x_num_pads,
            sub_batch_target_y_num_pads,
        ) = data_loader.get_batch_samples(
            img_idx_batch_list=list(range(from_idx, to_idx)),
            dataset_split=dataset_split,
        )
        sub_batch_input_x = sub_batch_input_x.to(rank)
        sub_batch_target_y = sub_batch_target_y.to(rank)

        sub_batch_input_x = sub_batch_input_x
        sub_batch_target_y = sub_batch_target_y
        tot_num_tokens += sub_batch_target_y.size(1) * sub_batch_target_y.size(0) - sum(
            sub_batch_target_y_num_pads
        )
        encoder.apply_softmax = False
        decoder.apply_softmax = False
        cross_enc_out = encoder(
            enc_x=sub_batch_input_x,
            dec_x=sub_batch_target_y[:, :-1],
            enc_x_num_pads=sub_batch_input_x_num_pads,
            dec_x_num_pads=sub_batch_target_y_num_pads,
        )

        pred = decoder(
            enc_x=cross_enc_out,
            dec_x=sub_batch_target_y[:, :-1],
            enc_x_num_pads=sub_batch_input_x_num_pads,
            dec_x_num_pads=sub_batch_target_y_num_pads,
        )
        encoder.apply_softmax = True
        decoder.apply_softmax = True
        tot_loss += loss_function(
            pred,
            sub_batch_target_y[:, 1:],
            data_set.get_pad_token_idx(),
            divide_by_non_zeros=False,
        ).item()
        del sub_batch_input_x, sub_batch_target_y, pred
        torch.cuda.empty_cache()
    tot_loss /= tot_num_tokens
    if verbose and rank == 0:
        print("Validation Loss on " + str(num_samples) + " samples: " + str(tot_loss))

    return tot_loss


def evaluate_quantized_model(
    encoder,
    decoder,
    y_idx2word_list,
    beam_size,
    max_seq_len,
    sos_idx,
    eos_idx,
    rank,
    batch_size,
    indexes=[0],
    data_loader=None,
    dataset_split=VizWizDataset.ValidationSet_ID,
    use_images_instead_of_features=True,
    verbose=True,
    stanford_model_path="./eval/get_stanford_models.sh",
):
    start_time = time()

    sub_list_predictions = []
    validate_y = []
    num_samples = len(indexes)
    encoder.eval()
    decoder.eval()
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
        rank=rank,
    )
    with torch.no_grad():
        num_iter_sub_batches = math.ceil(len(indexes) / batch_size)
        sb_size = batch_size
        for sb_it in tqdm(range(num_iter_sub_batches)):
            last_iter = sb_it == num_iter_sub_batches - 1
            if last_iter:
                from_idx = sb_it * sb_size
                to_idx = num_samples
            else:
                from_idx = sb_it * sb_size
                to_idx = (sb_it + 1) * sb_size
            print(from_idx, to_idx)
            if use_images_instead_of_features:
                sub_batch_x = [
                    data_loader.get_images_by_idx(
                        i, dataset_split=dataset_split
                    ).unsqueeze(0)
                    for i in list(range(from_idx, to_idx))
                ]
                sub_batch_x = torch.cat(sub_batch_x).to(rank)
                sub_batch_x_num_pads = [0] * sub_batch_x.size(0)
            else:
                sub_batch_x = [
                    data_loader.get_bboxes_by_idx(i, dataset_split=dataset_split)
                    for i in list(range(from_idx, to_idx))
                ]
                sub_batch_x = torch.nn.utils.rnn.pad_sequence(
                    sub_batch_x, batch_first=True
                ).to(rank)
                sub_batch_x_num_pads = compute_num_pads(sub_batch_x)

            validate_y += [
                data_loader.get_captions_by_idx(i, dataset_split=dataset_split)
                for i in list(range(from_idx, to_idx))
            ]

            output_words, _ = captioner(
                enc_x=sub_batch_x,
                enc_x_num_pads=sub_batch_x_num_pads,
                mode="beam_search",
            )

            output_words = [output_words[i][0] for i in range(len(output_words))]

            pred_sentence = language_utils.convert_allsentences_idx2word(
                output_words, y_idx2word_list
            )
            for sentence in pred_sentence:
                sub_list_predictions.append(
                    " ".join(sentence[1:-1])
                )  # remove EOS and SOS
            # print(sub_list_predictions[-1], validate_y[-1])
            del sub_batch_x, sub_batch_x_num_pads, output_words
    encoder.train()
    decoder.train()
    if (rank == 0 or rank == "cpu") and verbose:
        # dirty code to leave the evaluation code untouched
        list_predictions = [sub_predictions for sub_predictions in sub_list_predictions]
        list_list_references = [
            [validate_y[i][j] for j in range(len(validate_y[i]))]
            for i in range(len(validate_y))
        ]

        gts_dict = dict()
        for i in range(len(list_list_references)):
            gts_dict[i] = [
                {"image_id": i, "caption": list_list_references[i][j]}
                for j in range(len(list_list_references[i]))
            ]

        pred_dict = dict()
        for i in range(len(list_predictions)):
            pred_dict[i] = [{"image_id": i, "caption": list_predictions[i]}]

        coco_eval = COCOEvalCap(
            gts_dict,
            pred_dict,
            list(range(len(list_predictions))),
            get_stanford_models_path=stanford_model_path,
        )
        score_results = coco_eval.evaluate(
            bleu=True, rouge=True, cider=True, spice=True, meteor=True, verbose=False
        )
        elapsed_ticks = time() - start_time
        print(
            "Evaluation Phase over "
            + str(len(validate_y))
            + " BeamSize: "
            + str(beam_size)
            + "  elapsed: "
            + str(int(elapsed_ticks / 60))
            + " m "
            + str(int(elapsed_ticks % 60))
            + " s"
        )
        print(score_results)

    if rank == 0:
        return pred_dict, gts_dict

    return None, None


def evaluate_quantized_model_on_set(
    encoder,
    decoder,
    caption_idx2word_list,
    sos_idx,
    eos_idx,
    num_samples,
    data_loader,
    dataset_split,
    eval_max_len,
    rank,
    batch_size,
    beam_size=5,
    stanford_model_path="./eval/get_stanford_models.sh",
    use_images_instead_of_features=True,
    get_predictions=False,
    is_vizwiz=False,
):
    with torch.no_grad():
        encoder.eval()
        decoder.eval

        pred_dict, gts_dict = evaluate_quantized_model(
            encoder,
            decoder,
            y_idx2word_list=caption_idx2word_list,
            beam_size=beam_size,
            max_seq_len=eval_max_len,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            rank=rank,
            batch_size=batch_size,
            indexes=list(range(num_samples)),
            data_loader=data_loader,
            dataset_split=dataset_split,
            use_images_instead_of_features=use_images_instead_of_features,
            verbose=True,
            stanford_model_path=stanford_model_path,
        )

    if get_predictions:
        return pred_dict, gts_dict

    return None, None


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
        default="/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/pretrained_weights/encoder_expt5.pth",
    )
    parser.add_argument(
        "--decoder_load_path",
        type=str,
        default="/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/pretrained_weights/decoder_expt5.pth",
    )
    parser.add_argument("--image_folder", type=str, default="/home/arpitsah/Desktop/Fall-2023/odml/vizWiz")
    parser.add_argument(
        "--vocab_path", type=str, default="/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/vocab/coco_vocab_idx_dict.json"
    )
    parser.add_argument("--vizwiz", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument(
        "--img_size", type=int, default=384, help=" Image size for Swin Transformer"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--model_type", type=str, default="qat", help="Model Type to Load"
    )
    parser.add_argument("--demo", type=str2bool, default=True)
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
            with open("/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/vocab/coco_vocab_idx_dict.json", "r") as vocab_json:
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
            vizwiz_annotations_dir="/home/arpitsah/Desktop/Fall-2023/odml/vizWiz/annotations",
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
    # with open("prepared_decoder.txt", "w+") as f:
    #     print(prepared_decoder, file=f)
    # sys.exit()

    if args.demo: 
        demo_images = os.listdir("./vizwiz_demo")
        for file in demo_images: 
            print(file)
            path = os.path.join("./vizwiz_demo", file)
            demo_quantized_model(encoder_model, decoder_model, path, dataset.caption_idx2word_list, sos_idx=dataset.get_sos_token_idx(),
                eos_idx=dataset.get_eos_token_idx())
    else: 
        image_folder = args.image_folder
        array_of_init_seeds = [random.random() for _ in range(1 * 2)]
        data_loader = VizWizDataLoader(
            vizwiz_dataset=dataset,
            batch_size=4,
            num_procs=1,
            array_of_init_seeds=array_of_init_seeds,
            dataloader_mode="caption_wise",
            resize_image_size=args.img_size,
            rank=args.device,
            image_folder=image_folder,
            verbose=True,
        )
        model_max_len = dataset.max_seq_len + 20
        print("DataLoader initialized ...")
        evaluate_quantized_model_on_set(
            encoder_model,
            decoder_model,
            dataset.caption_idx2word_list,
            dataset.get_sos_token_idx(),
            dataset.get_eos_token_idx(),
            dataset.val_num_images,
            data_loader,
            VizWizDataset.ValidationSet_ID,
            model_max_len,
            args.device,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
        )


if __name__ == "__main__":
    main()
