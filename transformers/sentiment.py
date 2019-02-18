from plp.transformers.interface import DocTransformer, ListStat
import plp.token as ptoken
import plp.seq as pseq
import plp.vocab as pvocab
import plp.utils.iterator as piterator
import plp.doc as pdoc
import pdb


class DocLabelsTransformer(DocTransformer):
    """
    With a list of documents, the transformer puts the doc into rnn batch
    train format, with each batch containing bool flag for the end of the doc
    If seq_len is provided, all the seqs will be truncated or padded to that len
    but the src_len returns the actual length
    """
    def __init__(self, batch_size, seq_len, max_doc_len=None):
        self._batch_size = batch_size
        self._seq_len = seq_len 
        self._max_doc_len = max_doc_len

    def get_lists_stats(self, doc, *docs):
        token_type = doc.token_type
        DocTransformer.assert_docs_token_type(token_type, *docs)
        return (
                ListStat("src", token_type, self._seq_len, is_seq=True),
                ListStat("src_len", "value_int_type", 1),
                ListStat("label", "id_type", 1),
                ListStat("eod_flag", "value_int_type", 1)
                )

    def transform_docs(self, doc, *docs):
        pad_token = self._validate(doc, *docs)
        docs = [doc] + list(docs)
        for batched_docs in piterator.batched_items_iter(self._batch_size, *docs):
            seq_iters = [iter(pseq.SeqDocument.create_len_separated_seq_doc(doc_, self._seq_len))
                         for doc_ in batched_docs]
            labels = [doc_.get_label("label") for doc_ in batched_docs]
            eod_flags = [0 for _ in range(len(batched_docs))]
            i = 0 
            next_seq, next_seq_len = next(seq_iters[i])
            next_label = labels[i]
            while True:
                seq, label, seq_len = next_seq, next_label, next_seq_len
                i = (i+1) % self._batch_size
                try:
                    next_seq, next_seq_len = next(seq_iters[i]) 
                    next_label = labels[i]
                    yield seq, seq_len, label, 0
                except StopIteration:
                    next_seq, next_seq_len = [pad_token] * self._seq_len, 0
                    next_label = labels[i]
                    eod_flags[i] = 1
                    all_eod = True
                    for flag in eod_flags:
                        if flag == 0:
                            all_eod = False
                    if all_eod:
                        yield seq, seq_len, label, 1
                        break
                    else:
                        yield seq, seq_len, label, 0

    def batch_transform_docs(self, doc, *docs, yield_doc=False):
        pad_token = self._validate(doc, *docs)
        docs = [doc] + list(docs)
        for batched_docs, lens in pdoc.batch_docs_by_len(self._batch_size, self._seq_len, docs):
            #if len(batched_docs) != 32:
            #    pdb.set_trace()
            #    print()
            seq_iters = [iter(pseq.SeqDocument.create_len_separated_seq_doc(doc_, self._seq_len))
                         for doc_ in batched_docs]
            labels = [doc_.get_label("label") for doc_ in batched_docs]
            batch_cont_count = 0

            next_batch_src = None
            next_batch_src_len = None
            while True:
                batch_src = next_batch_src
                batch_src_len = next_batch_src_len
                next_batch_src = []
                next_batch_src_len = []
                try:
                    for seq_iter in seq_iters:
                        seq, seq_len = next(seq_iter)
                        next_batch_src.append(seq)
                        next_batch_src_len.append(seq_len)
                except StopIteration:
                    if batch_src is None:
                        raise ValueError("No examples found")
                    if not yield_doc:
                        yield batch_src, batch_src_len, labels, 1
                    else:
                        yield batch_src, batch_src_len, labels, 1, batched_docs
                    break
                if batch_src is None:
                    batch_src = next_batch_src
                    batch_src_len = next_batch_src_len
                batch_cont_count += 1
                if self._max_doc_len is not None:
                    if batch_cont_count * self._seq_len >= self._max_doc_len:
                        if not yield_doc:
                            yield batch_src, batch_src_len, labels, 1
                        else:
                            yield batch_src, batch_src_len, labels, 1, batched_docs
                        break
                if next_batch_src_len[0] == 0:
                    if not yield_doc:
                        yield batch_src, batch_src_len, labels, 1 
                    else:
                        yield batch_src, batch_src_len, labels, 1, batched_docs
                    break

                if not yield_doc:
                    yield batch_src, batch_src_len, labels, 0
                else:
                    yield batch_src, batch_src_len, labels, 0, batched_docs

    def _validate(self, doc, *docs):
        assert self._seq_len is not None
        token_type = doc.token_type
        if token_type == "word_type":
            pad_token = pvocab.PAD
        elif token_type == "id_type":
            pad_token = pvocab.PAD_ID
        else:
            raise NotImplementedError("not implemented type")
        DocTransformer.assert_docs_token_type(token_type, *docs)
        return pad_token


    def transform_seq_docs(self, seq_doc, *seq_docs):
        pass

    def estimate_docs_transformed_size(self, *docs):
        num_docs = {}
        for doc in docs:
            label = doc.get_label("label")
            if label not in num_docs:
                num_docs[label] = 1
            else:
                num_docs[label] += 1
        return num_docs

    def estimate_seq_docs_transformed_size(self, *docs):
        pass


