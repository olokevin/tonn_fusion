import torch

def concatenate_one(inputs):

    batch_size = inputs[0].shape[0]
    device = inputs[0].device
    dtype = inputs[0].dtype

    concatenated_inputs = [torch.cat((x, torch.ones((batch_size, 1), device=device, dtype=dtype)), dim=1) \
        for x in inputs]

    return concatenated_inputs

def get_log_prior_coeff(log_prior_coeff, epoch, warm_up_epochs, no_log_prior_epochs):

    coeff = log_prior_coeff * (epoch - no_log_prior_epochs) / warm_up_epochs
    return min(max(coeff, 0), 1)