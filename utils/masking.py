import torch


@torch.fx.wrap
def assign_zeros(mask, seq_len, offset, dim):
    if dim == 1:
        mask[:, (seq_len - offset) :, :] = 0
    else:
        mask[:, :, (seq_len - offset) :] = 0


@torch.fx.wrap
def fill_zeros(mask, batch_size, seq_len, offsets, dim):
    for batch_idx in range(batch_size):
        if dim == 1:
            mask[batch_idx, (seq_len - offsets[batch_idx]) :, :] = 0
        else:
            mask[batch_idx, :, (seq_len - offsets[batch_idx]) :] = 0
    return mask


def create_pad_mask(mask_size, pad_row, pad_column, rank=0):
    batch_size, output_seq_len, input_seq_len = mask_size
    mask = torch.tensor((), dtype=torch.float32).to(rank)
    mask = mask.new_ones((1, 1, 1))
    mask = (
        mask.repeat_interleave(batch_size, dim=0)
        .repeat_interleave(output_seq_len, dim=1)
        .repeat_interleave(input_seq_len, dim=2)
    )

    mask = fill_zeros(mask, batch_size, input_seq_len, pad_column, dim=2)
    mask = fill_zeros(mask, batch_size, output_seq_len, pad_row, dim=1)
    return mask


def create_no_peak_and_pad_mask(mask_size, num_pads, rank=0):
    batch_size, seq_len, seq_len = mask_size
    ones = torch.tensor((), dtype=torch.float32).to(rank)
    ones = ones.new_ones((1, 1))
    ones = ones.repeat_interleave(seq_len, dim=0).repeat_interleave(seq_len, dim=1)
    mask = (
        torch.tril(ones, diagonal=0).unsqueeze(0).repeat_interleave(batch_size, dim=0)
    )
    mask = fill_zeros(mask, batch_size, seq_len, num_pads, dim=2)
    mask = fill_zeros(mask, batch_size, seq_len, num_pads, dim=1)
    return mask
