import os
import plp.token as ptoken
import plp.vocab as pvocab
import plp.serializers.txt as ptxt
import pdb
import glob


class Document(object):
    @classmethod
    def create_from_txt(cls, txt_path, token_type, gen_eol_type,
                        vocab_reader=None, isolating_tokens=None):
        """
        isolating_tokens: E.g. "hi\tmy" where tab needs to be identified as 
        special flag token later
        """
        ptoken.assert_type_doc_valid(token_type)
        if not os.path.exists(txt_path):
            raise IOError(txt_path + " file not found")
        flag_tokens = []
        if gen_eol_type == "yield_eol":
            doc_gen_f = ptxt.doc_gen_f_yield_eol(txt_path, token_type, isolating_tokens)
        elif gen_eol_type == "ignore_eol":
            doc_gen_f = ptxt.doc_gen_f_ignore_eol(txt_path, token_type, isolating_tokens)
        elif gen_eol_type == "keep_eol_nl":
            doc_gen_f = ptxt.doc_gen_f_keep_eol_nl(txt_path, token_type, isolating_tokens)
            flag_tokens.append("\n")
        else:
            raise ValueError("Non existing end of line type")
        return cls(doc_gen_f, token_type, flag_tokens, vocab_reader, txt_path)
    
    def save_as_txt(self, txt_path, num_tokens_per_line=None):
        if num_tokens_per_line is None:
            ptxt.doc_save(txt_path, iter(self))
        else:
            ptxt.doc_save_by_line(txt_path, iter(self), num_tokens_per_line, self.token_type)
    
    @classmethod
    def create_from_docs(cls, *docs):
        if len(docs) == 0:
            return None
        token_type = docs[0].token_type

        def merged_iter_f():
            for doc_ in docs:
                assert token_type == doc_.token_type
                for item in iter(doc_):
                    yield item
        return cls(merged_iter_f, token_type)  # TODO handle vocab

    def __init__(self, src_gen_f, token_type,
                 flag_tokens=None, vocab_reader=None, src_path=None):
        self._src_gen_f = src_gen_f
        self._src_path = src_path
        self._f_name = os.path.basename(src_path) if src_path else None
        self._gen_fs = []
        self._token_type = token_type
        self._vocab_reader = vocab_reader
        self._label_dict = {}
        self._doc_len = None
        self._applied_flag_tokens = flag_tokens if flag_tokens else []
        self._is_locked = False

    def __iter__(self):
        token_iter = self._src_gen_f()
        for gen_f in self._gen_fs:
            token_iter = gen_f(token_iter)
        return token_iter

    def __len__(self):
        if not self._doc_len:
            self._doc_len = sum(1 for _ in iter(self))
        return self._doc_len

    @property
    def token_type(self):
        return self._token_type

    @property
    def src_path(self):
        return self._src_path

    @property
    def f_name(self):
        return self._f_name

    @property
    def applied_flag_tokens(self):
        return self._applied_flag_tokens

    @property
    def is_flag_token_applied(self):
        return len(self.applied_flag_tokens) > 0

    @property
    def embed_size(self):
        if self.token_type != "embed_type":
            return None
        else:
            return self._vocab_reader.embed_size

    @property
    def pad_embedding(self):
        return self._vocab_reader.id2embed_lookup(pvocab.PAD_ID)

    def lock(self):
        self._is_locked = True

    def _assert_not_locked(self, op):
        if self._is_locked:
            raise ValueError("Cannot do this op: " + op + " when locked")

    def record_new_flag_tokens(self, *new_tokens):
        for token in new_tokens:
            if token not in self.applied_flag_tokens:
                self._applied_flag_tokens.append(token)
            else:
                print("Warning: already recorded token " +
                      token + " as one of applied flags")

    def del_prev_flag_tokens(self, *unapplied_tokens):
        for token in unapplied_tokens:
            if token in self.applied_flag_tokens:
                self._applied_flag_tokens.remove(token)
            else:
                print("Warning: cannot del token " +
                      token + " not recorded as applied")

    def set_vocab_reader(self, vocab_reader):
        self._vocab_reader = vocab_reader

    def set_label(self, key, val):
        self._label_dict[key] = val

    def get_label(self, key):
        return self._label_dict.get(key, None)

    def get_iters_with_annotations(self, annotation_transformers):
        max_num_left, max_num_right = ptoken.get_transformers_max_num_tokens(
            annotation_transformers
        )
        left, right = [], []
        token_iter = iter(self)
        try:
            center = next(token_iter)
        except StopIteration:
            return
        while center is not None:
            annotation = None
            if len(left) == max_num_left and len(right) == max_num_right:
                for transformer in annotation_transformers:
                    annotation = transformer[left, center, right]
                    if annotation is not None:
                        break
            else:
                for transformer in annotation_transformers:
                    if transformer.is_applicable(len(left), len(right)):
                        annotation = transformer[left, center, right]
                        if annotation is not None:
                            break
                if annotation is None:
                    if len(right) < max_num_right:
                        try:
                            right.append(next(token_iter))
                            continue
                        except StopIteration:
                            pass
            yield center, annotation
            center = ptoken.shift_context_center_tokens(
                (left, center, right),
                token_iter, max_num_left)

    ############################################
    # State Changing methods, Wrap by f_gen_fs #
    ############################################
    def toggle_word_id(self):
        assert self._vocab_reader is not None
        self._assert_not_locked("toggle_word_id")
        if self._token_type == "word_type":
            self._gen_fs.append(ptoken.create_word2id_gen_f(
                self._vocab_reader, self._applied_flag_tokens))
            self._token_type = "id_type"
        elif self._token_type == "id_type":
            self._gen_fs.append(ptoken.create_id2word_gen_f(
                self._vocab_reader, self._applied_flag_tokens))
            self._token_type = "word_type"
        else:
            raise ValueError("Curr token type does not support toggle word/id")

    def convert_embed(self):
        assert self._vocab_reader is not None
        self._assert_not_locked("convert embed")
        if self._token_type == "word_type":
            self._gen_fs.append(ptoken.create_word2embed_gen_f(
                self._vocab_reader, self._applied_flag_tokens))
            self._token_type = "embed_type"
        elif self._token_type == "id_type":
            self._gen_fs.append(ptoken.create_id2embed_gen_f(
                self._vocab_reader, self._applied_flag_tokens))
            self._token_type = "embed_type"
        else:
            raise ValueError("Curr token type does not support toggle word/id")
    
    def strip_tokens(self):
        assert self.token_type == "word_type"
        self._assert_not_locked("strip_tokens")

        def strip_tokens_gen(token_iter):
            for token in token_iter:
                if token not in self.applied_flag_tokens:
                    yield token.strip()
                else:
                    yield token
        self._gen_fs.append(strip_tokens_gen)

    def skip_tokens(self, bool_token_transformers):
        max_num_left, max_num_right = ptoken.get_transformers_max_num_tokens(
            bool_token_transformers
        )
        self._assert_not_locked("skip_tokens")

        def skip_tokens_gen(token_iter):
            left, right = [], []
            try:
                center = next(token_iter)
            except StopIteration:
                return
            while center is not None:
                skip_flag = False
                if len(left) == max_num_left and len(right) == max_num_right:
                    for transformer in bool_token_transformers:
                        skip_flag = transformer[left, center, right]
                        if skip_flag:
                            break
                    if not skip_flag:
                        yield center
                else:
                    for transformer in bool_token_transformers:
                        if transformer.is_applicable(len(left), len(right)):
                            skip_flag = transformer[left, center, right]
                            if skip_flag:
                                break
                    if not skip_flag:
                        if len(right) < max_num_right:
                            try:
                                right.append(next(token_iter))
                                continue
                            except StopIteration:
                                pass
                        yield center
                center = ptoken.shift_context_center_tokens(
                    (left, center, right),
                    token_iter, max_num_left)
        self._gen_fs.append(skip_tokens_gen)

    def transform_tokens(self, token_transformers):
        max_num_left, max_num_right = ptoken.get_transformers_max_num_tokens(
            token_transformers
        )
        self._assert_not_locked("transform_tokens")

        def transform_tokens_gen(token_iter):
            left, right = [], []
            try:
                center = next(token_iter)
            except StopIteration:
                return
            while center is not None:
                new_tokens = None
                if len(left) == max_num_left and len(right) == max_num_right:
                    for transformer in token_transformers:
                        new_tokens = transformer[left, center, right]
                        if new_tokens is not None:
                            break
                else:
                    for transformer in token_transformers:
                        if transformer.is_applicable(len(left), len(right)):
                            new_tokens = transformer[left, center, right]
                            if new_tokens is not None:
                                break
                    if new_tokens is None:
                        if len(right) < max_num_right:
                            try:
                                right.append(next(token_iter))
                                continue
                            except StopIteration:
                                pass
                if new_tokens is None:
                    yield center
                else:
                    for new_token in new_tokens:
                        yield new_token
                # pdb.set_trace()
                center = ptoken.shift_context_center_tokens(
                    (left, center, right),
                    token_iter, max_num_left)
        self._gen_fs.append(transform_tokens_gen)

    def mask_unk(self):
        assert self._vocab_reader is not None
        self._assert_not_locked("mask_unk")

        def mask_unk_gen_f(token_iter):
            if self.token_type == "word_type":
                unk_signature = pvocab.UNK
                unk_check_f = self._vocab_reader.check_word_exist
            else:
                raise NotImplementedError("Not implemented yet")
            for token in token_iter:
                if not unk_check_f(token):
                    yield unk_signature
                else:
                    yield token
        self._gen_fs.append(mask_unk_gen_f)


def sort_docs_by_len(docs):
    doc_len_tuples = [(doc, len(doc)) for doc in docs]
    return sorted(doc_len_tuples, key=lambda x: x[1])


def batch_docs_by_len(batch_size, batch_seq_len, docs):
    sorted_doc_len_tuples = sort_docs_by_len(docs)
    # pdb.set_trace()
    doc_len_iter = iter(sorted_doc_len_tuples)
    items = []
    item_lens = []
    while True:
        try:
            doc, doc_len = next(doc_len_iter)
            items.append(doc)
            item_lens.append(doc_len)
        except StopIteration:
            break

        if (item_lens[-1]-1)//batch_seq_len != (item_lens[0]-1)//batch_seq_len:
            yield items[:-1], item_lens[:-1]
            items = [items[-1]]
            item_lens = [item_lens[-1]]

        if len(items) == batch_size:
            yield items, item_lens
            items = []
            item_lens = []


def create_from_txt_dir(txt_dir, token_type, gen_eol_type,
                        vocab_reader=None, isolating_tokens=None):
    docs = []
    for f_path in sorted(glob.iglob(os.path.join(txt_dir, "*.txt"))):
        doc = Document.create_from_txt(
                f_path, "word_type", gen_eol_type, vocab_reader, isolating_tokens
                )
        docs.append(doc)
    return docs


def gen_from_txt_dir(txt_dir, token_type, gen_eol_type,
                      vocab_reader=None, isolating_tokens=None):
    for f_path in sorted(glob.iglob(os.path.join(txt_dir, "*.txt"))):
        doc = Document.create_from_txt(
                f_path, "word_type", gen_eol_type, vocab_reader, isolating_tokens
                )
        yield doc







