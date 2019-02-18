import numpy as np
import ipdb


def batch_pad_(batch_tokens, pad_token, max_num_tokens=None):
    """
    In-place
    Output: mask [m, s]
    """
    if max_num_tokens is None:
        max_num_tokens = max(len(sub_tokens) for sub_tokens in batch_tokens)
    mask = np.ones((len(batch_tokens), max_num_tokens), dtype=np.float32)
    for sub_tokens, submask in zip(batch_tokens, mask):
        if len(sub_tokens) < max_num_tokens:
            submask[len(sub_tokens):max_num_tokens] = 0.
            sub_tokens.extend([pad_token]*(max_num_tokens - len(sub_tokens)))
        elif len(sub_tokens) > max_num_tokens:
            while len(sub_tokens)!=max_num_tokens:
                sub_tokens.pop()
    return mask


def batch_pad_lookup_(batch_tokens, vocab, max_num_tokens=None):
    """
    Inputs: batch_tokens: list of list
    Outputs: batch_ids [m, s] np.int64
             mask [m, s] np.float32
         are np.ndarray objects
    """
    # [m, s]
    mask = batch_pad_(batch_tokens, vocab.pad_token, max_num_tokens)
    ids = []
    for sub_tokens in batch_tokens:
        ids.append([])
        for x in sub_tokens:
            x_id = vocab[x]
            ids[-1].append(x_id)
    # TODO CHECK np type
    batch_ids = np.array(ids)
    return batch_ids, mask


def batch_pad_oov_lookup_(batch_tokens, vocab, oov2randidx_dict=None, max_num_tokens=None):
    """
    Inputs: batch_tokens: list of list
    Outputs: batch_ids [m, s] np.int64
             mask [m, s] np.float32
         are np.ndarray objects
    """
    # [m, s]
    mask = batch_pad_(batch_tokens, vocab.pad_token, max_num_tokens)
    ids = []
    oov_mask = []
    # oov_idx & oov2randidx_dict make sure oov tokens in different place can
    # have same "rand" embeddings (E.g. goal, dom)
    oov_idxs = []
    for sub_tokens in batch_tokens:
        ids.append([])
        oov_mask.append([])
        oov_idxs.append([])
        for x in sub_tokens:
            x_id = vocab[x]
            ids[-1].append(x_id)
            if x_id != vocab.unk_id:
                # not out of vocab
                oov_mask[-1].append(0)
            else:
                # OOV 
                oov_mask[-1].append(1)
                if x not in oov2randidx_dict:
                    num_oov_items = len(oov2randidx_dict)
                    oov2randidx_dict[x] = num_oov_items
                oov_idxs[-1].append(oov2randidx_dict[x])
    # TODO CHECK np type
    batch_ids = np.array(ids, dtype=np.int64)
    # [m, max_num_tokens]
    oov_mask = np.array(oov_mask)
    oov_ids = np.array(oov_idxs, dtype=np.int64)
    return batch_ids, mask, oov_mask, oov_ids



