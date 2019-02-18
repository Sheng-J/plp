from plp.transformers.interface import DocTransformer, ListStat
import plp.token as ptoken
import plp.seq as pseq
import plp.vocab as pvocab
import numpy as np


class QueAnsTransformer(DocTransformer):
    def __init__(self, q_max_len, cs_max_size, c_max_len, a_max_len):
        self._q_max_len = q_max_len
        self._cs_max_size = cs_max_size
        self._c_max_len = c_max_len
        self._a_max_len = a_max_len

    def get_lists_stats(self, doc, *docs):
        token_type = doc.token_type
        DocTransformer.assert_docs_token_type(token_type, *docs)
        if token_type == ptoken.EMBED_TYPE:
            embed_stat = ListStat(
                "embed", ptoken.VALUE_FLOAT_TYPE, doc.embed_size
                )
            context_stat = ListStat(
                "context", ptoken.LIST_TYPE, self._c_max_len,
                is_seq=True, sub_list_stat=embed_stat
                )
        else:
            context_stat = ListStat(
                "context", token_type, self._c_max_len,
                is_seq=True)
        return (
            ListStat("question", token_type, self._q_max_len, is_seq=True),
            ListStat("contexts", ptoken.LIST_TYPE, self._cs_max_size,
                     is_seq=True, sub_list_stat=context_stat),
            ListStat("answer", token_type, self._a_max_len, is_seq=True),
            ListStat("q_len", ptoken.VALUE_INT_TYPE, 1),
            ListStat("cs_size", ptoken.VALUE_INT_TYPE, 1),
            ListStat("a_len", ptoken.VALUE_INT_TYPE, 1),
            ListStat("c_lens", ptoken.VALUE_INT_TYPE,
                     self._cs_max_size, is_seq=True)
        )

    def transform_docs(self, doc, *docs):
        raise NotImplementedError("Not supported")

    def transform_seq_docs(self, qa_seq_doc, *qa_seq_docs):
        token_type = qa_seq_doc.token_type
        DocTransformer.assert_docs_token_type(token_type, *qa_seq_docs)
        if token_type == ptoken.ID_TYPE:
            create_contexts_f = lambda: np.full(
                (self._cs_max_size, self._c_max_len),
                pvocab.PAD_ID, dtype=np.int64
            )
            create_q_f = lambda: np.full(
                self._q_max_len,
                pvocab.PAD_ID, dtype=np.int64
            )
            create_a_f = lambda: np.full(
                self._a_max_len,
                pvocab.PAD_ID, dtype=np.int64
            )
        elif token_type == ptoken.WORD_TYPE:
            create_contexts_f = lambda: np.full(
                (self._cs_max_size, self._c_max_len),
                pvocab.PAD, dtype=object
            )
            create_q_f = lambda: np.full(
                self._q_max_len,
                pvocab.PAD, dtype=object
            )
            create_a_f = lambda: np.full(
                self._a_max_len,
                pvocab.PAD, dtype=object
            )
        else:
            embed_size = qa_seq_doc.embed_size
            pad_embedding = qa_seq_doc.pad_embedding
            create_contexts_f = lambda: np.full(
                (self._cs_max_size, self._c_max_len, embed_size),
                pad_embedding, dtype=np.float64
            )
            create_q_f = lambda: np.full(
                (self._q_max_len, embed_size),
                pad_embedding, dtype=np.float64
            )
            create_a_f = lambda: np.full(
                (self._a_max_len, embed_size),
                pad_embedding, dtype=np.float64
            )
        qa_seq_docs = [qa_seq_doc] + list(qa_seq_docs)
        for q_a_doc in qa_seq_docs:
            que, ans = None, None
            context_seqs = create_contexts_f()
            q_seq, a_seq = create_q_f(), create_a_f()
            q_len, a_len, c_lens = None, None, []
            c_index = 0
            for seq, flag in q_a_doc:
                if flag == "context":
                    if token_type == ptoken.EMBED_TYPE:
                        context_seqs[c_index, :len(seq), :] = seq
                    else:
                        context_seqs[c_index, :len(seq)] = seq
                    c_index += 1
                    c_lens.append(len(seq))
                elif flag == "question":
                    if token_type == ptoken.EMBED_TYPE:
                        q_seq[:len(seq), :] = seq
                    else:
                        q_seq[:len(seq)] = seq
                    q_len = len(seq)
                elif flag == "answer":
                    if token_type == ptoken.EMBED_TYPE:
                        a_seq[:len(seq), :] = seq
                    else:
                        a_seq[:len(seq)] = seq
                    a_len = len(seq)
                    yield q_seq, context_seqs, a_seq, \
                        [q_len], [c_index], [a_len], c_lens
                    context_seqs = create_contexts_f()
                    q_seq, a_seq = create_q_f(), create_a_f()
                    q_len, a_len, c_lens = None, None, []
                    c_index = 0
                else:
                    pass

    def estimate_max_size(self, *docs):
        raise NotImplementedError("Not implemented yet")

