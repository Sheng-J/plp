import plp.token as ptoken
import plp.vocab as pvocab
import plp.serializers.txt as ptxt


class SeqDocument(object):
    @classmethod
    def create_len_slided_seq_doc(cls, doc, fixed_len):
        assert not doc.is_flag_token_applied

        def fixed_len_gen(token_iter):
            seq_list = []
            try:
                for _ in range(fixed_len):
                    seq_list.append(next(token_iter))
            except:
                return
            while True:
                yield tuple(seq_list), fixed_len
                try:
                    seq_list.pop(0)
                    seq_list.append(next(token_iter))
                except:
                    break
        return cls(doc, fixed_len_gen)

    @classmethod
    def create_len_separated_seq_doc(cls, doc, fixed_len):
        assert not doc.is_flag_token_applied

        def fixed_len_gen(token_iter):
            while True:
                seq_list = []
                try:
                    for _ in range(fixed_len):
                        seq_list.append(next(token_iter))
                except:
                    break
                yield tuple(seq_list), fixed_len
            if doc.token_type == "word_type":
                pad_token = pvocab.PAD
            else:
                pad_token = pvocab.PAD_ID
            num_pad = fixed_len - len(seq_list)
            final_seq = tuple(seq_list + [pad_token]*num_pad)
            actual_seq_len = len(seq_list)
            yield final_seq, actual_seq_len
        return cls(doc, fixed_len_gen)

    @classmethod
    def create_flag_separated_seq_doc(cls, doc):
        assert doc.is_flag_token_applied

        def flag_gen(token_iter):
            seq_list = []
            while True:
                try:
                    item = next(token_iter)
                    if item in doc.applied_flag_tokens:
                        yield seq_list, item
                        seq_list = []
                    else:
                        seq_list.append(item)
                except StopIteration:
                    break
            if len(seq_list):
                yield seq_list, "<final>"
        return cls(doc, flag_gen)

    def save_as_txt(self, seq_txt_path, flag_txt_path):
        ptxt.seq_doc_save(seq_txt_path, flag_txt_path, iter(self))

    def __init__(self, doc, seq_gen_f):
        doc.lock()
        self._doc = doc
        self._seq_gen_f = seq_gen_f
        self._seq_flag_gen_fs = []

    def get_max_len_dict(self):
        len_dict = {"max_length": {}, "max_cont": {}}
        prev_label, label_cont_count = None, None
        for seq, label in self:
            if len(seq) > len_dict["max_length"].get(label, 0):
                len_dict["max_length"][label] = len(seq)

            if prev_label is not None:
                if prev_label != label:
                    if label_cont_count > len_dict["max_cont"].get(prev_label, 0):
                        len_dict["max_cont"][prev_label] = label_cont_count
                    prev_label = label
                    label_cont_count = 1
                else:
                    label_cont_count += 1
            else:
                prev_label = label
                label_cont_count = 1
        if label_cont_count > len_dict["max_cont"].get(prev_label, 0):
            len_dict["max_cont"][prev_label] = label_cont_count
        return len_dict

    def __iter__(self):
        seq_flag_iter = self._seq_gen_f(iter(self._doc))
        for seq_flag_gen_f in self._seq_flag_gen_fs:
            seq_flag_iter = seq_flag_gen_f(seq_flag_iter)
        return seq_flag_iter

    def merge_seq_by_flag_tokens(self, flag_tokens):
        def merge_seq_gen(seq_flag_iter):
            cumu_seq = []
            for (seq, flag) in seq_flag_iter:
                cumu_seq += seq
                if flag in flag_tokens:
                    yield cumu_seq, flag
                    cumu_seq = []
        self._seq_flag_gen_fs.append(merge_seq_gen)

    def replace_flag_tokens(self, flag_gen_f):
        def replaced_flag_gen(seq_flag_iter):
            for (seq, _), new_flag in zip(seq_flag_iter, flag_gen_f()):
                yield seq, new_flag
        self._seq_flag_gen_fs.append(replaced_flag_gen)

    def transform_flags(self, flag_token_transformers):
        max_num_left, max_num_right = ptoken.get_transformers_max_num_tokens(
            flag_token_transformers
        )

        def transform_flags_gen(seq_flag_iter):
            left_seq_list, right_seq_list = [], []
            left_flag_list, right_flag_list = [], []
            try:
                seq, center = next(seq_flag_iter)
            except StopIteration:
                return
            while center is not None:
                new_center = None
                if len(left_flag_list) == max_num_left and len(right_flag_list) == max_num_right:
                    for transformer in flag_token_transformers:
                        new_center = transformer[left_flag_list, center, right_flag_list,]
                        # if one successfully transformed the flag, stop checking the rest
                        if new_center is not None:
                            break
                else:
                    for transformer in flag_token_transformers:
                        if transformer.is_applicable(len(left_flag_list), len(right_flag_list)):
                            new_center = transformer[left_flag_list, center, right_flag_list]
                            # if one successfully transformed the flag, stop checking the rest
                            if new_center is not None:
                                break
                    # If none of the transformer was applicable or transforming the token
                    # Expand right flag list until max
                    if new_center is None:
                        if len(right_flag_list) < max_num_right:
                            try:
                                right_seq, right_flag = next(seq_flag_iter)
                                right_seq_list.append(right_seq)
                                right_flag_list.append(right_flag)
                                continue
                            except StopIteration:
                                pass
                if new_center is None:
                    yield seq, center
                else:
                    yield seq, new_center
                seq, center = ptoken.shift_context_center_tokens(
                    (
                        [left_seq_list, left_flag_list],
                        [seq, center],
                        [right_seq_list, right_flag_list]
                    ),
                    seq_flag_iter, max_num_left
                )
                # seq, center will have None if right len is zero and no more elements are iter
        self._seq_flag_gen_fs.append(transform_flags_gen)

    def __getattr__(self, attr):
        if attr in ["token_type", "embed_size", "get_label"]:
            return getattr(self._doc, attr)
        else:
            raise ValueError("SeqDocument doesn't have attr " + attr)

