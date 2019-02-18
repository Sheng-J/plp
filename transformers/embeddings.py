from plp.transformers.interface import DocTransformer, ListStat
import plp.token as ptoken
import plp.utils as putils
import plp.vocab as pvocab


class Word2vecTransformer(DocTransformer):
    def __init__(self, window_size, round_iter=False):
        self._window_size = window_size
        self._round_iter = round_iter

    def get_lists_stats(self, doc, *docs):
        token_type = doc.token_type
        if token_type != "word_type" and token_type != "id_type":
            raise NotImplementedError("not implemented type")
        DocTransformer.assert_docs_token_type(token_type, *docs)
        return(
            ListStat("center", token_type, 1),
            ListStat("context", token_type, 1)
        )

    def transform_docs(self, doc, *docs):
        token_type = doc.token_type
        if token_type != "word_type" and token_type != "id_type":
            raise NotImplementedError("not implemented type")
        unk_token = pvocab.UNK if token_type == "word_type" else pvocab.UNK_ID
        DocTransformer.assert_docs_token_type(token_type, *docs)
        docs = [doc] + list(docs)
        if self._round_iter:
            word2vec_iters = [self._word2vec_gen(iter(doc), unk_token) for doc in docs]
            for center, context in putils.iterator.merged_round_iter(*word2vec_iters):
                yield center, context
        else:
            for doc in docs:
                for center, context in self._word2vec_gen(iter(doc), unk_token):
                    yield center, context


    def _word2vec_gen(self, token_iter, unk_token):
        try:
            center_word = next(token_iter)
        except StopIteration:
            return

        left_context_words = []
        right_context_words = []
        for _ in range(self._window_size):
            try:
                right_context_words.append(next(token_iter))
            except StopIteration:
                break
        while center_word is not None:
            context_words = left_context_words + right_context_words
            if center_word != unk_token:
                for context_word in context_words:
                    if context_word != unk_token:
                        yield center_word, context_word
            center_word = ptoken.shift_context_center_tokens(
                (left_context_words, center_word, right_context_words),
                token_iter,
                self._window_size
            )

    def transform_seq_docs(self, seq_doc, *seq_docs):
        token_type = seq_doc.token_type
        if token_type != "word_type" and token_type != "id_type":
            raise NotImplementedError("not implemented type")
        unk_token = pvocab.UNK if token_type == "word_type" else pvocab.UNK_ID
        DocTransformer.assert_docs_token_type(token_type, *seq_docs)
        seq_docs = [seq_doc] + list(seq_docs)
        for seq_doc in seq_docs:
            for seq, _ in iter(seq_doc):
                for center, context in self._word2vec_gen(iter(seq), unk_token):
                    yield center, context

    def estimate_docs_transformed_size(self, doc, *docs):
        len_sum = 0 
        docs = [doc] + list(docs)
        for doc in docs:
            len_sum += (2 * self._window_size) * (len(doc) - 1) 
        return len_sum

    def estimate_seq_docs_transformed_size(self, *docs):
        raise NotImplementedError("Not supported")


class Sca2wordTransformer(DocTransformer):
    def __init__(self, val_token_type, window_size, each_num_examples, vocab_reader, u_w_ret_id=True):
        self._window_size = window_size
        self._each_num_examples = each_num_examples
        self._val_token_type = val_token_type
        self._vocab_reader = vocab_reader
        self._u_w_ret_id = u_w_ret_id

    def get_lists_stats(self, doc, *docs):
        token_type = doc.token_type
        if token_type != "word_type":
            raise NotImplementedError("not implemented type")
        DocTransformer.assert_docs_token_type(token_type, *docs)
        return(
            ListStat("u_a", "id_type", self._window_size),
            ListStat("v_a", self._val_token_type, 1),
            ListStat("w_a", "id_type", self._window_size),
            ListStat("u_b", "id_type", self._window_size),
            ListStat("v_b", self._val_token_type, 1),
            ListStat("w_b", "id_type", self._window_size)
        )

    def transform_docs(self, doc, *docs):
        token_type = doc.token_type
        if token_type != "word_type":
            raise NotImplementedError("not implemented type")
        docs = [doc] + list(docs)
        DocTransformer.assert_docs_token_type(token_type, *docs)
        # sca2word_iters = [self._sca2word_gen(doc) for doc in docs]
        vocab = self._vocab_reader
        for doc in docs:
            for u_a, v_a, w_a, u_b, v_b, w_b in self._sca2word_gen(doc):
                u_a = [vocab[x] for x in u_a]
                w_a = [vocab[x] for x in w_a]
                u_b = [vocab[x] for x in u_b]
                w_b = [vocab[x] for x in w_b]
                yield u_a, v_a, w_a, u_b, v_b, w_b
        # for u_a, v_a, w_a, u_b, v_b, w_b in merged_round_iter(*sca2word_iters):
        #     yield u_a, v_a, w_a, u_b, v_b, w_b

    def transform_docs2(self, doc, *docs):
        token_type = doc.token_type
        if token_type != "word_type":
            raise NotImplementedError("not implemented type")
        docs = [doc] + list(docs)
        DocTransformer.assert_docs_token_type(token_type, *docs)
        # sca2word_iters = [self._sca2word_gen(doc) for doc in docs]
        vocab = self._vocab_reader
        for doc in docs:
            for (u, v, w), is_sca in self.u_v_w_gen_with_label(doc):
                if is_sca:
                    yield u, float(v), w

    def find_next_u_v_w(self, doc_iter):
        try:
            u = [next(doc_iter) for _ in range(self._window_size)]
            v = next(doc_iter)
            w = [next(doc_iter) for _ in range(self._window_size)]
            u_v_w = [u, v, w]
        except StopIteration:
            return None
        while True:
            # if is_num(w):
            #    pdb.set_trace()
            if self.is_u_v_w(u_v_w):
                return u_v_w
            try:
                u_v_w[0].pop(0)
                u_v_w[0].append(u_v_w[1])
                u_v_w[1] = u_v_w[2].pop(0)
                u_v_w[2].append(next(doc_iter))
            except StopIteration:
                return None

    def u_v_w_gen_with_label(self, doc):
        """
        When the is_sca label is True,
        the context u, w are converted to their indices
        """
        doc_iter = iter(doc)
        u = [next(doc_iter) for _ in range(self._window_size)]
        u = []
        for i in range(self._window_size):
            item = next(doc_iter)
            u.append(item)
            yield (None, item, None), False
        v = next(doc_iter)
        w = [next(doc_iter) for _ in range(self._window_size)]
        u_v_w = [u, v, w]

        is_end = False
        while not is_end:
            u_v_w_ids = self.u_v_w_word2id(u_v_w)
            yield u_v_w_ids, self.is_u_v_w(u_v_w)
            u_v_w[0].pop(0)
            u_v_w[0].append(u_v_w[1])
            u_v_w[1] = u_v_w[2].pop(0)
            try: 
                u_v_w[2].append(next(doc_iter))
            except StopIteration:
                yield u_v_w, False
                is_end = True
        for item in u_v_w[2]:
            yield (None, item, None), False

    def u_v_w_word2id(self, u_v_w):
        u = [self._vocab_reader[x] for x in u_v_w[0]]
        w = [self._vocab_reader[x] for x in u_v_w[2]]
        v = u_v_w[1]
        return (u, v, w)


    def _sca2word_gen(self, doc):
        doc_gen = iter(doc)
        u_v_w_a = self.find_next_u_v_w(doc_gen)
        if u_v_w_a is None:
            return
            # raise ValueError("Not even a single example")
        comparisons = [self.find_next_u_v_w(doc_gen) for _ in range(self._each_num_examples)]
        count = 0
        while True:
            if comparisons[0] is None:
                # print("found " + str(count) + " examples")
                break
            for u_v_w_b in comparisons:
                if not u_v_w_b:
                    break
                u_a, v_a, w_a = u_v_w_a
                u_b, v_b, w_b = u_v_w_b
                count += 1
                if self._val_token_type == "value_int_type":
                    yield u_a, int(float(v_a)), w_a, u_b, int(float(v_b)), w_b
                elif self._val_token_type == "value_float_type":
                    yield u_a, float(v_a), w_a, u_b, float(v_b), w_b
                else:
                    raise ValueError("Unsupported Value token type")

            u_v_w_a = comparisons.pop(0)
            comparisons.append(self.find_next_u_v_w(doc_gen))

    @staticmethod
    def is_num(token):
        return token.lstrip('-').replace('.', '', 1).isdigit()

    def is_u_v_w(self, u_v_w):
        u, v, w = u_v_w
        if not self.is_num(v):
            return False
        for u_, w_ in zip(u, w):
            # Not training on <unk> token!
            if self.is_num(u_) or self.is_num(w_)\
                    or (not self._vocab_reader.check_word_exist(u_))\
                    or (not self._vocab_reader.check_word_exist(w_)):
                return False
        return True

    def estimate_docs_transformed_size(self, *docs):
        len_sum = 0
        for _ in self.transform_docs(*docs):
            len_sum += 1
        return len_sum

    def estimate_docs_single_transformed_size(self, *docs):
        len_sum = 0
        for _ in self.transform_docs2(*docs):
            len_sum += 1
        return len_sum

    def estimate_seq_docs_transformed_size(self, *docs):
        raise NotImplementedError("Not supported")

    def transform_seq_docs(self, seq_doc, *seq_docs):
        raise NotImplementedError("Not supported")

