import torch
import ipdb


def batch_pad_(batch_tokens, pad_token, device, max_num_tokens=None):
    """
    In-place
    """
    if max_num_tokens is None:
        max_num_tokens = max(len(sub_tokens) for sub_tokens in batch_tokens)
    mask = torch.ones(len(batch_tokens), max_num_tokens, device=device)
    for sub_tokens, submask in zip(batch_tokens, mask):
        if len(sub_tokens) < max_num_tokens:
            submask[len(sub_tokens):max_num_tokens] = 0.
            sub_tokens.extend([pad_token]*(max_num_tokens - len(sub_tokens)))
    return mask


def batch_pad_lookup_(batch_tokens, vocab, device, max_num_tokens=None):
    """
    Inputs: batch_tokens: list of list
    Outputs: batch_ids [m, s]
             mask [m, s]
         are torch.tensor objects
    """
    mask = batch_pad_(batch_tokens, vocab.pad_token, device, max_num_tokens)
    batch_ids = torch.tensor(
            [[vocab[x] for x in sub_tokens] for sub_tokens in batch_tokens],
            device=device
            )
    return batch_ids, mask


