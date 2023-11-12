import torch
import os
from tqdm import tqdm
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, " \t", "Size (MB):", size / 1e6)
    os.remove("temp.p")
    return size


def calibrate(model, data_loader, iters=30, device="cpu"):
    model.eval()
    model.to(device)
    devices = {p.device for p in model.parameters()} | {
        p.device for p in model.buffers()
    }
    print(devices)
    # iters = len(data_loader.dataset)
    with torch.no_grad():
        for ix in tqdm(range(iters)):
            batch = data_loader.get_batch_samples(2, [ix])
            (
                batch_input_x,
                batch_target_y,
                batch_input_x_num_pads,
                batch_target_y_num_pads,
            ) = batch
            batch_input_x = batch_input_x.to(device)
            batch_target_y = batch_target_y.to(device)
            model(
                enc_x=batch_input_x,
                dec_x=batch_target_y[:, :-1],
                enc_x_num_pads=batch_input_x_num_pads,
                dec_x_num_pads=batch_target_y_num_pads,
            )


def prepare_model(model_to_quantize, example_inputs, qconfig_mapping, device="cpu", qat=False):
    model_to_quantize.eval()
    prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
    return prepared_model


def calibrate_enc_dec(encoder, decoder, data_loader, num_iters, device="cpu"):
    encoder.eval().to(device)
    decoder.eval().to(device)

    # iters = len(data_loader.dataset)
    with torch.no_grad():
        for ix in tqdm(range(num_iters)):
            batch = data_loader.get_batch_samples(2, [ix])
            (
                batch_input_x,
                batch_target_y,
                batch_input_x_num_pads,
                batch_target_y_num_pads,
            ) = batch
            batch_input_x = batch_input_x.to(device)
            batch_target_y = batch_target_y.to(device)
            cross_enc_out = encoder(
                enc_x=batch_input_x,
                dec_x=batch_target_y[:, :-1],
                enc_x_num_pads=batch_input_x_num_pads,
                dec_x_num_pads=batch_target_y_num_pads,
            )
            decoder(
                enc_x=cross_enc_out,
                dec_x=batch_target_y[:, :-1],
                enc_x_num_pads=batch_input_x_num_pads,
                dec_x_num_pads=batch_target_y_num_pads,
            )


def quantize_model(prepared_model, device="cpu"):
    prepared_model.to("cpu").eval()
    quantized_model = convert_fx(prepared_model)
    quantized_model.to(device)
    return quantized_model


def quantize_encoder_decoder(
    encoder, decoder, data_loader, num_iters, qconfig_mapping, device="cpu", static=True, 
    qat=False
):
    example_inputs = list(data_loader.get_batch_samples(2, [0]))
    prepared_encoder = prepare_model(encoder, example_inputs, qconfig_mapping, device, qat=qat)
    prepared_decoder = prepare_model(decoder, example_inputs, qconfig_mapping, device, qat=qat)
    if static:
        calibrate_enc_dec(
            prepared_encoder, prepared_decoder, data_loader, num_iters, device
        )
    quantized_encoder = quantize_model(prepared_encoder)
    quantized_decoder = quantize_model(prepared_decoder)
    return quantized_encoder, quantized_decoder
