import argparse
import os
import sys
import random
from argparse import Namespace
from time import time
import json

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.ao.quantization import (
    QConfigMapping,
    get_default_qat_qconfig_mapping,
    QConfig,
    get_default_qconfig_mapping,
)
from torch.ao.quantization.quantize_fx import convert_fx
from torch.ao.quantization.qconfig import (
    default_embedding_qat_qconfig,
    default_qat_qconfig_v2,
    default_qat_qconfig,
)
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    NoopObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.fake_quantize import default_embedding_fake_quant
from torch.nn.parallel import DistributedDataParallel as DDP
from data.coco_dataset import CocoDatasetKarpathy
from data.coco_dataloader import CocoDataLoader
from data.vizwiz_dataset import VizWizDataset
from data.vizwiz_dataloader import VizWizDataLoader
from test import compute_evaluation_loss, evaluate_model_on_set
from losses.loss import LabelSmoothingLoss, kl_loss
from losses.reward import ReinforceCiderReward
from optims.radam import RAdam
from utils import language_utils
from utils.args_utils import (
    str2bool,
    str2list,
    scheduler_type_choice,
    optim_type_choice,
)
from utils.saving_utils import (
    load_most_recent_checkpoint,
    save_last_checkpoint,
    partially_load_state_dict,
)
from models.End_ExpansionNet_v2 import E2E_ExpansionNet_Captioner

torch.autograd.set_detect_anomaly(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import functools
from utils.quantization_utils import prepare_model
from quantization import encoder_modules, decoder_modules, filter_state_dict
from quantization_eval import (
    evaluate_quantized_model_on_set,
    compute_quantized_evaluation_loss,
)

print = functools.partial(print, flush=True)


def convert_time_as_hhmmss(ticks):
    return str(int(ticks / 60)) + " m " + str(int(ticks) % 60) + " s"


def train(
    rank,
    train_args,
    path_args,
    ddp_model,
    dataset,
    data_loader,
    optimizer,
    sched,
    max_len,
    ddp_sync_port,
):
    num_sampled_captions = 5
    if not train_args.reinforce:
        loss_function = LabelSmoothingLoss(smoothing_coeff=0.1, rank=rank)
        loss_function.to(rank)
    else:  # 'rf'
        running_logprobs = 0
        running_reward = 0
        running_reward_base = 0

        training_references = dataset.get_all_images_captions(
            CocoDatasetKarpathy.TrainSet_ID
        )
        reinforce_reward = ReinforceCiderReward(
            training_references, dataset.get_eos_token_str(), num_sampled_captions, rank
        )

    algorithm_start_time = time()
    saving_timer_start = time()
    time_to_save = False
    kd_alpha = .9
    running_loss = 0
    running_teacher_loss = 0
    running_student_loss = 0
    running_time = 0
    already_trained_steps = (
        data_loader.get_num_batches() * data_loader.get_epoch_it()
        + data_loader.get_batch_it()
    )
    prev_print_iter = already_trained_steps
    num_iter = data_loader.get_num_batches() * train_args.num_epochs
    num_batches = data_loader.get_num_batches()
    print(f"NUM ITER: {num_iter}")
    sampling_search_kwargs = {
        "sample_max_seq_len": train_args.scst_max_len,
        "how_many_outputs": num_sampled_captions,
        "sos_idx": dataset.get_sos_token_idx(),
        "eos_idx": dataset.get_eos_token_idx(),
    }
    if train_args.quantized:
        if train_args.kd:
            ddp_encoder, ddp_decoder, ddp_teacher = ddp_model
            ddp_teacher.eval()

        else: 
            ddp_encoder, ddp_decoder = ddp_model

    for it in range(already_trained_steps, num_iter):
        iter_timer_start = time()
        if train_args.quantized:
            ddp_encoder.train()
            ddp_decoder.train()
            if train_args.kd and train_args.phase_2:
                ddp_teacher.train()
        else:
            ddp_model.train()

        if not train_args.reinforce:
            (
                batch_input_x,
                batch_target_y,
                batch_input_x_num_pads,
                batch_target_y_num_pads,
                batch_img_idx,
            ) = data_loader.get_next_batch(
                verbose=True
                * (
                    ((it + 1) % train_args.print_every_iter == 0)
                    or (it + 1) % data_loader.get_num_batches() == 0
                ),
                get_also_image_idxes=True,
            )

            batch_input_x = batch_input_x.to(rank)
            batch_target_y = batch_target_y.to(rank)
            # create a list of sub-batches so tensors can be deleted right-away after being used
            if train_args.quantized:
                # import pdb; pdb.set_trace()

                cross_enc_out = ddp_encoder(
                    enc_x=batch_input_x,
                    dec_x=batch_target_y[:, :-1],
                    enc_x_num_pads=batch_input_x_num_pads,
                    dec_x_num_pads=batch_target_y_num_pads,
                )
                # print(it)
                # print(f"First Pass!")
     
                pred_logprobs = ddp_decoder(
                    enc_x=cross_enc_out,
                    dec_x=batch_target_y[:, :-1],
                    enc_x_num_pads=batch_input_x_num_pads,
                    dec_x_num_pads=batch_target_y_num_pads,
                )

                if train_args.kd:
                    if train_args.phase_2:
                         teacher_logprobs = ddp_teacher(
                            enc_x=batch_input_x,
                            dec_x=batch_target_y[:, :-1],
                            enc_x_num_pads=batch_input_x_num_pads,
                            dec_x_num_pads=batch_target_y_num_pads,
                        )
                    else: 
                        with torch.no_grad():
                            teacher_logprobs = ddp_teacher(
                                enc_x=batch_input_x,
                                dec_x=batch_target_y[:, :-1],
                                enc_x_num_pads=batch_input_x_num_pads,
                                dec_x_num_pads=batch_target_y_num_pads,
                            )
                    kd_loss_student = kl_loss(pred_logprobs, teacher_logprobs, temperature=4)
                    running_student_loss += kd_loss_student.item()
                    if train_args.phase_2:
                        ce_teacher_loss = loss_function(teacher_logprobs, batch_target_y[:, 1:],
                                                         dataset.get_pad_token_idx())
                        kd_teacher_loss = kl_loss(teacher_logprobs, pred_logprobs, temperature=3)
                        teacher_loss = (1 - kd_alpha) * ce_teacher_loss + kd_alpha * kd_teacher_loss
                        teacher_loss.backward(retain_graph=True) 
                        # teacher_loss.backward()
                        running_teacher_loss += teacher_loss.item()


            else:
                pred_logprobs = ddp_model(
                    enc_x=batch_input_x,
                    dec_x=batch_target_y[:, :-1],
                    enc_x_num_pads=batch_input_x_num_pads,
                    dec_x_num_pads=batch_target_y_num_pads,
                )
            # print(f"Log Probs {pred_logprobs}")
            loss = loss_function(
                pred_logprobs, batch_target_y[:, 1:], dataset.get_pad_token_idx()
            )
            if train_args.kd:
                loss = (1 - kd_alpha) * loss + kd_alpha * kd_loss_student
            # print(f"LOSS IS {loss}")
            running_loss += loss.item()

            loss.backward()
        else:  # rf mode
            (
                batch_input_x,
                batch_target_y,
                batch_input_x_num_pads,
                batch_img_idx,
            ) = data_loader.get_next_batch(
                verbose=True
                * (
                    ((it + 1) % train_args.print_every_iter == 0)
                    or (it + 1) % data_loader.get_num_batches() == 0
                ),
                get_also_image_idxes=True,
            )

            batch_input_x = batch_input_x.to(rank)
            if train_args.quantized:
                captioner = E2E_ExpansionNet_Captioner(
                    sampling_search_kwargs,
                    split_encoder=True,
                    encoder=ddp_encoder,
                    decoder=ddp_decoder,
                    rank=rank,
                    apply_log_softmax=True,
                    train=True,
                )
            else:
                captioner = E2E_ExpansionNet_Captioner(
                    sampling_search_kwargs,
                    split_encoder=False,
                    model=ddp_model,
                    rank=rank,
                    apply_log_softmax=True,
                    train=True,
                )
            all_images_pred_idx, all_images_logprob = captioner(
                enc_x=batch_input_x,
                enc_x_num_pads=batch_input_x_num_pads,
                mode="sampling",
            )

            all_images_pred_caption = [
                language_utils.convert_allsentences_idx2word(
                    one_image_pred_idx, dataset.caption_idx2word_list
                )
                for one_image_pred_idx in all_images_pred_idx
            ]

            reward_loss, reward, reward_base = reinforce_reward.compute_reward(
                all_images_pred_caption=all_images_pred_caption,
                all_images_logprob=all_images_logprob,
                all_images_idx=batch_img_idx,
            )

            running_logprobs += all_images_logprob.sum().item() / len(
                torch.nonzero(all_images_logprob, as_tuple=False)
            )
            running_reward += reward.sum().item() / len(reward.flatten())
            running_reward_base += reward_base.sum().item() / len(reward_base.flatten())
            running_loss += reward_loss.item()
            # reward_loss.backward()
            reward_loss.backward(retain_graph=True)

        if it % train_args.num_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        sched.step()

        current_rl = sched.get_last_lr()[0]

        running_time += time() - iter_timer_start
        if (it + 1) % train_args.print_every_iter == 0:
            if not train_args.reinforce:
                avg_loss = running_loss / (it + 1 - prev_print_iter)
                if train_args.kd: 
                    avg_student_kd_loss = running_student_loss /  (it + 1 - prev_print_iter)
                else: 
                    avg_student_kd_loss = 0
                avg_student_ce_loss = avg_loss - avg_student_kd_loss    
                if train_args.phase_2: 
                    avg_teacher_loss = running_teacher_loss / (it + 1 - prev_print_iter)
                else: 
                    avg_teacher_loss = 0
                tot_elapsed_time = time() - algorithm_start_time
                avg_time_time_per_iter = running_time / (it + 1 - prev_print_iter)
                print(
                    "[GPU:"
                    + str(rank)
                    + "] "
                    + str(round(((it + 1) / num_iter) * 100, 3))
                    + " % it: "
                    + str(it + 1)
                    + " lr: "
                    + str(round(current_rl, 12))
                    + " n.acc: "
                    + str(train_args.num_accum)
                    + " avg total loss: "
                    + str(round(avg_loss, 3))
                    + " avg student ce loss: "
                    + str(round( avg_student_ce_loss, 3))
                    + " avg student kd loss: "
                    + str(round( avg_student_kd_loss, 3))
                    + " avg teacher loss: "
                    + str(round(avg_teacher_loss, 3))
                    + " elapsed: "
                    + convert_time_as_hhmmss(tot_elapsed_time)
                    + " sec/iter: "
                    + str(round(avg_time_time_per_iter, 3))
                )
                running_loss = 0
                if train_args.kd: 
                    running_student_loss = 0
                if train_args.phase_2: 
                    running_teacher_loss = 0
                running_time = 0
                prev_print_iter = it + 1
            else:
                avg_loss = running_loss / (it + 1 - prev_print_iter)
                tot_elapsed_time = time() - algorithm_start_time
                avg_time_time_per_iter = running_time / (it + 1 - prev_print_iter)
                avg_logprobs = running_logprobs / (it + 1 - prev_print_iter)
                avg_reward = running_reward / (it + 1 - prev_print_iter)
                avg_reward_base = running_reward_base / (it + 1 - prev_print_iter)
                print(
                    "[GPU:"
                    + str(rank)
                    + "] "
                    + str(round(((it + 1) / num_iter) * 100, 3))
                    + " % it: "
                    + str(it + 1)
                    + " lr: "
                    + str(round(current_rl, 12))
                    + " n.acc: "
                    + str(train_args.num_accum)
                    + " avg rew loss: "
                    + str(round(avg_loss, 3))
                    + " elapsed: "
                    + convert_time_as_hhmmss(tot_elapsed_time)
                    + " sec/iter: "
                    + str(round(avg_time_time_per_iter, 3))
                    + "\n"
                    " avg reward: "
                    + str(round(avg_reward, 5))
                    + " avg base: "
                    + str(round(avg_reward_base, 5))
                    + " avg logprobs: "
                    + str(round(avg_logprobs, 5))
                )
                running_loss = 0
                running_time = 0
                running_logprobs = 0
                running_reward = 0
                running_reward_base = 0
                prev_print_iter = it + 1

        if (
            it + 1
        ) % train_args.eval_every_iter == 0:  # ((it + 1) % data_loader.get_num_batches() == 0) or
            if not train_args.reinforce:
                if train_args.quantized:
                    compute_quantized_evaluation_loss(
                        loss_function,
                        ddp_encoder,
                        ddp_decoder,
                        dataset,
                        data_loader,
                        dataset.val_num_images,
                        sub_batch_size=train_args.eval_parallel_batch_size,
                        dataset_split=dataset.ValidationSet_ID,
                        rank=rank,
                        verbose=True,
                    )
                else:
                    compute_evaluation_loss(
                        loss_function,
                        ddp_model,
                        dataset,
                        data_loader,
                        dataset.val_num_images,
                        sub_batch_size=train_args.eval_parallel_batch_size,
                        dataset_split=dataset.ValidationSet_ID,
                        rank=rank,
                        verbose=True,
                    )

            if rank == 0:
                print("Evaluation on Validation Set")
            if train_args.quantized:
                evaluate_quantized_model_on_set(
                    ddp_encoder,
                    ddp_decoder,
                    dataset.caption_idx2word_list,
                    dataset.get_sos_token_idx(),
                    dataset.get_eos_token_idx(),
                    dataset.val_num_images,
                    data_loader,
                    VizWizDataset.ValidationSet_ID,
                    max_len,
                    rank,
                    batch_size=args.batch_size,
                    beam_size=args.beam_size,
                )
            else:
                evaluate_model_on_set(
                    ddp_model,
                    dataset.caption_idx2word_list,
                    dataset.get_sos_token_idx(),
                    dataset.get_eos_token_idx(),
                    dataset.val_num_images,
                    data_loader,
                    dataset.ValidationSet_ID,
                    max_len,
                    rank,
                    ddp_sync_port,
                    parallel_batches=train_args.eval_parallel_batch_size,
                    use_images_instead_of_features=train_args.is_end_to_end,
                    beam_sizes=train_args.eval_beam_sizes,
                    is_vizwiz=train_args.vizwiz,
                )
            time_to_save = True

        # saving
        if (it + 1) % num_batches == 0:
            time_to_save = True
        elapsed_minutes = (time() - saving_timer_start) / 60
        if (
            time_to_save
            or elapsed_minutes > train_args.save_every_minutes
            or ((it + 1) == num_iter)
        ):
            saving_timer_start = time()
            time_to_save = False
            if rank == 0:
                if train_args.quantized:
                    save_last_checkpoint(
                        ddp_encoder.module,
                        optimizer,
                        sched,
                        data_loader,
                        path_args.save_path,
                        num_max_checkpoints=train_args.how_many_checkpoints,
                        additional_info="rf" if train_args.reinforce else "xe",
                        encoder=True
                    )
                    save_last_checkpoint(
                        ddp_decoder.module,
                        optimizer,
                        sched,
                        data_loader,
                        path_args.save_path,
                        num_max_checkpoints=train_args.how_many_checkpoints,
                        additional_info="rf" if train_args.reinforce else "xe",
                        decoder=True
                    )
                    if train_args.phase_2:
                        save_last_checkpoint(
                            ddp_teacher.module,
                            optimizer,
                            sched,
                            data_loader,
                            path_args.save_path,
                            num_max_checkpoints=train_args.how_many_checkpoints,
                            additional_info="rf" if train_args.reinforce else "xe",
                            teacher=True
                        )
                else: 
                    save_last_checkpoint(
                        ddp_model.module,
                        optimizer,
                        sched,
                        data_loader,
                        path_args.save_path,
                        num_max_checkpoints=train_args.how_many_checkpoints,
                        additional_info="rf" if train_args.reinforce else "xe",
                    )
               
            


def load_state_dict_filtered(model, checkpoint, filter_prefixes="enc"):
    pretrained_state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for key, value in pretrained_state_dict.items():
        if "swin_transf.patch_embed.proj.weight" in key:
            new_state_dict[key] = torch.nn.init.kaiming_uniform(
                torch.empty((192, 3, 3, 3))
            )
            continue
        if filter_prefixes == "dec":
            if "decoders.2" in key:
                new_key = key.replace("decoders.2", "decoders.1")
                new_state_dict[new_key] = value
                continue
            elif "dec_reduce_group.weight" in key:
                split_index = value.shape[-1] // 3
                first_part = value[:, :split_index]
                last_part = value[:, -split_index:]
                value = torch.hstack((first_part, last_part))
                new_state_dict[key] = value
                continue

        if "encoders.2" in key:
            new_key = key.replace("encoders.2", "encoders.1")
            new_state_dict[new_key] = value
            continue
        elif "enc_reduce_group.weight" in key:
            split_index = value.shape[-1] // 3
            first_part = value[:, :split_index]
            last_part = value[:, -split_index:]
            value = torch.hstack((first_part, last_part))
            new_state_dict[key] = value
            continue
        else:
            new_key = key
            new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)


def load_base_state_dict(model, checkpoint):
    new_state_dict = {}
 
    for key, value in checkpoint.items():
        """
        if "swin_transf.patch_embed.proj.weight" in key:
            new_state_dict[key] = torch.nn.init.kaiming_uniform(
                torch.empty((192, 3, 3, 3))
            )"""

        new_state_dict[key] = value 
    model.load_state_dict(new_state_dict)


def distributed_train(
    rank,
    world_size,
    model_args,
    optim_args,
    dataset,
    array_of_init_seeds,
    model_max_len,
    train_args,
    path_args,
):
    print("GPU: " + str(rank) + "] Process " + str(rank) + " working...")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = train_args.ddp_sync_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if model_args.param_config == 1:
        model_args.N_enc = 2

    elif model_args.param_config == 2:
        model_args.N_enc = 2
        model_args.N_dec = 2

    # img_size = 288
    img_size = 384
    if train_args.kd and not train_args.quantized:
        raise ValueError("Can only run kd with quantization code flow!")
    # if train_args.kd and train_args.quantized_checkpoint:
    #     raise ValueError("Can only run kd on first pass!")
    if train_args.is_end_to_end:
        from models.End_ExpansionNet_v2 import (
            End_ExpansionNet_v2,
            End_ExpansionNet_v2_Encoder,
            End_ExpansionNet_v2_Decoder,
        )

        if train_args.quantized:
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
                eps=1e-9,
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
                max_seq_len=model_max_len,
                drop_args=model_args.drop_args,
                eps=1e-9,
                rank="cpu",
            )
        if not train_args.quantized or (train_args.quantized and train_args.kd):
            model = End_ExpansionNet_v2(
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
                rank=rank,
                apply_log_softmax=train_args.reinforce,
            )

    else:
        from models.ExpansionNet_v2 import ExpansionNet_v2

        model = ExpansionNet_v2(
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
            img_feature_dim=1536,
            rank=rank,
        )
    if train_args.quantized: ##continue here
        if train_args.quantization_type == "static":
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

        example_input = (
            torch.randn(1, 3, img_size, img_size, device="cpu"),
            torch.randint(1, 100, (1, 15), device="cpu"),
            [0],
            [0],
        )
        # TODO: Loading and Preparing Logic (look at quantization.py)
        if train_args.quantized_checkpoint:
            print("Checkpoint already prepared for quantization...")
            prepared_encoder = prepare_model(
                encoder_model, example_input, qconfig_mapping, device="cpu", qat=True
            )
            print("Encoder prepared ...")
            prepared_decoder = prepare_model(
                decoder_model, example_input, qconfig_mapping, device="cpu", qat=True
            )
            print("Decoder prepared ...")
            prepared_encoder.load_state_dict(torch.load(path_args.encoder_load_path)["model_state_dict"])
            print("Encoder loaded ...")
            prepared_decoder.load_state_dict(torch.load(path_args.decoder_load_path)["model_state_dict"])
            print("Decoder loaded ...")

        else:
            print("Loading encoder / decoder from unprepared full model")
            state_dict = torch.load(path_args.pretrain_checkpoint)["model_state_dict"]
            encoder_state_dict = filter_state_dict(state_dict, encoder_modules)
            decoder_state_dict = filter_state_dict(state_dict, decoder_modules)

            encoder_model.load_state_dict(encoder_state_dict)
            print("Encoder loaded ...")
            decoder_model.load_state_dict(decoder_state_dict)
            print("Decoder loaded ...")
            print("Preparing Encoder and Decoder Model")
            encoder_model.to("cpu")
            decoder_model.to("cpu")
            prepared_encoder = prepare_model(
                encoder_model, example_input, qconfig_mapping, device="cpu", qat=True
            )
            print("Encoder prepared ...")
            prepared_decoder = prepare_model(
                decoder_model, example_input, qconfig_mapping, device="cpu", qat=True
            )
            print("Decoder prepared ...")
        if train_args.kd:
            checkpoint = torch.load(path_args.teacher_checkpoint)
            load_base_state_dict(model, checkpoint["model_state_dict"])
            model.to(rank)
        else:
            del encoder_model
            del decoder_model
        prepared_encoder.to(rank)
        prepared_decoder.to(rank)

    else:
        checkpoint = torch.load(path_args.pretrain_checkpoint)
        if model_args.param_config == 0:
            load_base_state_dict(model, checkpoint["model_state_dict"])
            print("Baseline Model loaded ...")

        elif model_args.param_config == 1:
            load_state_dict_filtered(model, checkpoint, "enc")
            print(" Model with 2 Encoder Layers loaded ...")

        elif model_args.param_config == 2:
            load_state_dict_filtered(model, checkpoint, "dec")
            print(" Model with 2 Encoder & 2 Decoder Layers  loaded ...")
    if train_args.quantized:
        ddp_encoder = DDP(prepared_encoder, device_ids=[rank])
        ddp_decoder = DDP(prepared_decoder, device_ids=[rank])
        if train_args.kd:
            ddp_teacher = DDP(model, device_ids=[rank])
        # ddp_encoder = prepared_encoder
        # ddp_decoder = prepared_decoder
    else:
        model.to(rank)
        ddp_model = DDP(model, device_ids=[rank])
    if train_args.vizwiz:
        print("VizWiz Dataloader in use")
        if train_args.reinforce:
            print("Reinforcement learning Mode")
            data_loader = VizWizDataLoader(
                vizwiz_dataset=dataset,
                batch_size=train_args.batch_size,
                num_procs=world_size,
                array_of_init_seeds=array_of_init_seeds,
                dataloader_mode="image_wise",
                resize_image_size=img_size if train_args.is_end_to_end else None,
                rank=rank,
                image_folder=path_args.image_folder,
                verbose=True,
            )
        else:
            print("Cross Entropy Learning Mode")
            data_loader = VizWizDataLoader(
                vizwiz_dataset=dataset,
                batch_size=train_args.batch_size,
                num_procs=world_size,
                array_of_init_seeds=array_of_init_seeds,
                dataloader_mode="caption_wise",
                resize_image_size=img_size if train_args.is_end_to_end else None,
                rank=rank,
                image_folder=path_args.image_folder,
                verbose=True,
            )
    else:
        if train_args.reinforce:
            print("Reinforcement learning Mode")
            data_loader = CocoDataLoader(
                coco_dataset=dataset,
                batch_size=train_args.batch_size,
                num_procs=world_size,
                array_of_init_seeds=array_of_init_seeds,
                dataloader_mode="image_wise",
                resize_image_size=img_size if train_args.is_end_to_end else None,
                rank=rank,
                verbose=True,
            )
        else:
            print("Cross Entropy learning mode")
            data_loader = CocoDataLoader(
                coco_dataset=dataset,
                batch_size=train_args.batch_size,
                num_procs=world_size,
                array_of_init_seeds=array_of_init_seeds,
                dataloader_mode="caption_wise",
                resize_image_size=img_size if train_args.is_end_to_end else None,
                rank=rank,
                verbose=True,
            )

    base_lr = 1.0
    if train_args.quantized:
        params = list(ddp_encoder.parameters()) + list(ddp_decoder.parameters())
    else:
        params = list(ddp_model.parameters())
    if optim_args.optim_type == "radam":
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, params),
            lr=base_lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, params), lr=base_lr)

    if optim_args.sched_type == "annealing":
        sched_func = (
            lambda it: (min(it, optim_args.warmup_iters) / optim_args.warmup_iters)
            * optim_args.lr
            * (
                0.8
                ** (
                    it
                    // (optim_args.anneal_every_epoch * data_loader.get_num_batches())
                )
            )
        )
    else:  # optim_args.sched_type == 'custom_warmup_anneal':
        num_batches = data_loader.get_num_batches()
        sched_func = lambda it: max(
            (it >= optim_args.warmup_iters) * optim_args.min_lr,
            (optim_args.lr / (max(optim_args.warmup_iters - it, 1)))
            * (
                pow(
                    optim_args.anneal_coeff,
                    it // (num_batches * optim_args.anneal_every_epoch),
                )
            ),
        )

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_func)

    if path_args.backbone_save_path != "" or path_args.body_save_path != "":
        if train_args.is_end_to_end:
            map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
            checkpoint = torch.load(
                path_args.backbone_save_path, map_location=map_location
            )
            if "model" in checkpoint.keys():
                partially_load_state_dict(model.swin_transf, checkpoint["model"])
            elif "model_state_dict" in checkpoint.keys():
                partially_load_state_dict(model, checkpoint["model_state_dict"])
            print("Backbone loaded...", end=" ")
            map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
            checkpoint = torch.load(path_args.body_save_path, map_location=map_location)
            partially_load_state_dict(model, checkpoint["model_state_dict"])
            print("Body loaded")
        else:
            if train_args.partial_load:
                map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
                checkpoint = torch.load(
                    path_args.body_save_path, map_location=map_location
                )
                partially_load_state_dict(model, checkpoint["model_state_dict"])
                print("Partial load done.")
    else:
        change_from_xe_to_rf = False
        changed_batch_size = data_loader.get_batch_size() != train_args.batch_size
        if changed_batch_size or change_from_xe_to_rf:
            if changed_batch_size:
                print(
                    "New requested batch size differ from previous checkpoint", end=" "
                )
                print("- Proceed to reset training session keeping pre-trained weights")
                data_loader.change_batch_size(
                    batch_size=train_args.batch_size, verbose=True
                )
            else:  # change_from_xe_to_rf
                print(
                    "Passing from XE training to RL - Optimizer and data loader states are resetted."
                )
                data_loader.set_epoch_it(epoch=0, verbose=True)

            if optim_args.optim_type == "radam":
                optimizer = RAdam(
                    filter(lambda p: p.requires_grad, ddp_model.parameters()),
                    lr=1,
                    betas=(0.9, 0.98),
                    eps=1e-9,
                )
            else:
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=1
                )

            sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_func)
    if train_args.quantized:
        if train_args.kd:
            ddp_model = (ddp_encoder, ddp_decoder, ddp_teacher)
        else:
            ddp_model = (ddp_encoder, ddp_decoder)
    train(
        rank,
        train_args,
        path_args,
        ddp_model,
        dataset,
        data_loader,
        optimizer,
        sched,
        model_max_len if not train_args.reinforce else train_args.scst_max_len,
        train_args.ddp_sync_port,
    )

    print("[GPU: " + str(rank) + " ] Closing...")
    dist.destroy_process_group()


def spawn_train_processes(model_args, optim_args, dataset, train_args, path_args):
    max_sequence_length = dataset.max_seq_len + 20
    print("Max sequence length: " + str(max_sequence_length))
    print("y vocabulary size: " + str(len(dataset.caption_word2idx_dict)))

    world_size = torch.cuda.device_count()
    print("Using - ", world_size, " processes / GPUs!")
    assert (
        train_args.num_gpus <= world_size
    ), "requested num gpus higher than the number of available gpus "
    print("Requested num GPUs: " + str(train_args.num_gpus))

    array_of_init_seeds = [random.random() for _ in range(train_args.num_epochs * 2)]
    distributed_train(
        0,
        train_args.num_gpus,
        model_args,
        optim_args,
        dataset,
        array_of_init_seeds,
        max_sequence_length,
        train_args,
        path_args,
    )
    """
    mp.spawn(
        distributed_train,
        args=(
            train_args.num_gpus,
            model_args,
            optim_args,
            dataset,
            array_of_init_seeds,
            max_sequence_length,
            train_args,
            path_args,
        ),
        nprocs=train_args.num_gpus,
        join=True,
    )"""


def set_seeds():
    seed = args.seed
    print("seed: " + str(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning")
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--N_enc", type=int, default=3)
    parser.add_argument("--N_dec", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--enc_drop", type=float, default=0.1)
    parser.add_argument("--dec_drop", type=float, default=0.1)
    parser.add_argument("--enc_input_drop", type=float, default=0.1)
    parser.add_argument("--dec_input_drop", type=float, default=0.1)
    parser.add_argument("--drop_other", type=float, default=0.1)

    parser.add_argument("--optim_type", type=optim_type_choice, default="adam")
    parser.add_argument("--sched_type", type=scheduler_type_choice, default="annealing")

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=5e-7)
    parser.add_argument("--warmup_iters", type=int, default=4000)
    parser.add_argument("--anneal_coeff", type=float, default=0.8)
    parser.add_argument("--anneal_every_epoch", type=float, default=3.0)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_accum", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ddp_sync_port", type=int, default=12324)
    parser.add_argument(
        "--save_path",
        type=str,
        default="./pretrained_weights/",
    )  # default='./github_ignore_material/saves/')
    parser.add_argument("--save_every_minutes", type=int, default=1500)
    parser.add_argument("--how_many_checkpoints", type=int, default=3)
    parser.add_argument("--print_every_iter", type=int, default=10)

    parser.add_argument("--eval_every_iter", type=int, default=999999)
    parser.add_argument("--eval_parallel_batch_size", type=int, default=8)
    parser.add_argument("--eval_beam_sizes", type=str2list, default=[3])

    parser.add_argument("--reinforce", type=str2bool, default=False)
    parser.add_argument("--vizwiz", type=str2bool, default=True)
    parser.add_argument("--quantized", type=str2bool, default=True) # just q--ph1
    parser.add_argument("--kd", type=str2bool, default=True) # all three is ph2, only kd and q is p
    parser.add_argument("--phase_2", type=str2bool, default=False)
    parser.add_argument("--quantization_type", type=str, default="static")
    parser.add_argument("--quantized_checkpoint", type=str2bool, default=True)

    parser.add_argument("--scst_max_len", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=2)

    parser.add_argument(
        "--image_folder",
        type=str,
        default="/home/arpitsah/Desktop/Fall-2023/odml/vizWiz",
    )
    parser.add_argument(
        "--captions_path", type=str, default="./github_ignore_material/raw_data/"
    )
    parser.add_argument(
        "--vocab_path", type=str, default="./vocab/coco_vocab_idx_dict.json"
    )
    parser.add_argument("--partial_load", type=str2bool, default=False)
    parser.add_argument("--backbone_save_path", type=str, default="")
    parser.add_argument("--body_save_path", type=str, default="")
    parser.add_argument("--is_end_to_end", type=str2bool, default=True)

    parser.add_argument(
        "--images_path", type=str, default="./github_ignore_material/raw_data/"
    )
    parser.add_argument("--preproc_images_hdf5_filepath", type=str, default=None)
    parser.add_argument(
        "--features_path", type=str, default="./github_ignore_material/raw_data/"
    )
    parser.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default="/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/pretrained_weights/4_th.pth",
    )
    parser.add_argument(
        "--encoder_load_path",
        type=str,
        default="/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/pretrained_weights/QAKD_full_stage_2/checkpoint_2023-12-04-05:34:21_epoch1it7875bs2_xeencoder_.pth",
    )
    parser.add_argument(
        "--decoder_load_path",
        type=str,
        default="/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/pretrained_weights/QAKD_full_stage_2/checkpoint_2023-12-04-05:34:25_epoch1it7875bs2_xedecoder_.pth",
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default="/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/pretrained_weights/QAKD_full_stage_2/checkpoint_2023-12-04-05:34:28_epoch1it7875bs2_xeteacher_.pth",
    )

    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--param_config",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Choose a mode: \n"
        "0 - Baseline\n"
        "1 - Remove layer in Encoder (Enc_dec)\n"
        "2 - Remove layer from Encoder and Decoder (Enc_deco_dec)",
    )
    args = parser.parse_args()
    args.ddp_sync_port = str(args.ddp_sync_port)

    # Seed setting ---------------------------------------------
    set_seeds()
    drop_args = Namespace(
        enc=args.enc_drop,
        dec=args.dec_drop,
        enc_input=args.enc_input_drop,
        dec_input=args.dec_input_drop,
        other=args.drop_other,
    )

    model_args = Namespace(
        model_dim=args.model_dim,
        N_enc=args.N_enc,
        N_dec=args.N_dec,
        dropout=args.dropout,
        drop_args=drop_args,
        param_config=args.param_config,
    )
    optim_args = Namespace(
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        anneal_coeff=args.anneal_coeff,
        anneal_every_epoch=args.anneal_every_epoch,
        optim_type=args.optim_type,
        sched_type=args.sched_type,
    )

    path_args = Namespace(
        save_path=args.save_path,
        images_path=args.images_path,
        image_folder=args.image_folder,
        captions_path=args.captions_path,
        vocab_path=args.vocab_path,
        features_path=args.features_path,
        backbone_save_path=args.backbone_save_path,
        body_save_path=args.body_save_path,
        preproc_images_hdf5_filepath=args.preproc_images_hdf5_filepath,
        pretrain_checkpoint=args.pretrain_checkpoint,
        encoder_load_path=args.encoder_load_path,
        decoder_load_path=args.decoder_load_path,
        teacher_checkpoint=args.teacher_checkpoint
    )

    train_args = Namespace(
        is_end_to_end=args.is_end_to_end,
        batch_size=args.batch_size,
        num_accum=args.num_accum,
        num_gpus=args.num_gpus,
        ddp_sync_port=args.ddp_sync_port,
        save_every_minutes=args.save_every_minutes,
        how_many_checkpoints=args.how_many_checkpoints,
        print_every_iter=args.print_every_iter,
        eval_every_iter=args.eval_every_iter,
        eval_parallel_batch_size=args.eval_parallel_batch_size,
        eval_beam_sizes=args.eval_beam_sizes,
        reinforce=args.reinforce,
        num_epochs=args.num_epochs,
        partial_load=args.partial_load,
        scst_max_len=args.scst_max_len,
        vizwiz=args.vizwiz,
        quantized=args.quantized,
        quantization_type=args.quantization_type,
        quantized_checkpoint=args.quantized_checkpoint,
        kd=args.kd,
        phase_2=args.phase_2,
    )

    print("train batch_size: " + str(args.batch_size))
    print("num_accum: " + str(args.num_accum))
    print("ddp_sync_port: " + str(args.ddp_sync_port))
    print("save_path: " + str(args.save_path))
    print("num_gpus: " + str(args.num_gpus))

    if train_args.vizwiz:
        if os.path.isfile(path_args.vocab_path):
            with open("/home/arpitsah/Desktop/Fall-2023/odml/On_Device_Image_Captioning/vocab/coco_vocab_idx_dict.json", "r") as vocab_json:
                coco_vocab_idx_dict = json.load(vocab_json)
        else:
            coco_vocab_idx_dict = None
        # Currently testing with val_split, normally should set to 1 with train being True
        split = 1
        dataset = VizWizDataset(
            split,
            train=True,
            coco_vocab_dict=coco_vocab_idx_dict,
            vizwiz_annotations_dir=f"{path_args.image_folder}/annotations",
        )
    else:
        dataset = CocoDatasetKarpathy(
            images_path=path_args.images_path,
            coco_annotations_path=path_args.captions_path + "dataset_coco.json",
            train2014_bboxes_path=path_args.captions_path + "train2014_instances.json",
            val2014_bboxes_path=path_args.captions_path + "val2014_instances.json",
            preproc_images_hdf5_filepath=path_args.preproc_images_hdf5_filepath
            if train_args.is_end_to_end
            else None,
            precalc_features_hdf5_filepath=None
            if train_args.is_end_to_end
            else path_args.features_path,
            limited_num_train_images=None,
            limited_num_val_images=5000,
        )

    # train base model
    spawn_train_processes(
        model_args=model_args,
        optim_args=optim_args,
        dataset=dataset,
        train_args=train_args,
        path_args=path_args,
    )
