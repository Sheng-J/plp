import tensorflow as tf
import json
import plp.doc as pdoc
import plp.seq as pseq
import plp.token as ptoken
import plp.vocab as pvocab
import plp.serializers.txt as ptxt
import plp.serializers.tfrecords as ptfrecords
import plp.deserializers.tfrecords.qa as pdtfrecords_qa
import plp.deserializers.tfrecords.iterator as pdtfrecords_iter
from plp.transformers.qa import QueAnsTransformer
from plp.transformers.interface import DocumentTransformState
import unittest
import pdb
import json


class TestbAbIDocument(unittest.TestCase):

    def setUp(self):
        self._doc1 = pdoc.Document.create_from_txt(
            "babi_sample/qa1_single-supporting-fact_test.txt",
            token_type=ptoken.WORD_TYPE,
            gen_eol_type=ptxt.GEN_KEEP_EOL_NL,
            isolating_tokens=["\t"]
        )
        self._doc1.record_new_flag_tokens("\t")

    def test_basic(self):
        #
        # 1. Remove number 1, 2.... in the front
        #
        babi_num_skip_transformer = ptoken.TokenTransformer(
            lambda left, center, right: True if (right[0] != "\n" and ptoken.is_num(center)) else False,
            num_left_tokens=0, num_right_tokens=1
        )
        # pdb.set_trace()
        self._doc1.skip_tokens([babi_num_skip_transformer]) 
        self._doc1.save_as_txt("babi_sample/qa1-token-skipped.txt")

        #
        # 2. tramsform "hello." into "hello"
        #

        comma_transformer = ptoken.TokenTransformer(
            lambda left, center, right: [center[:-1]] if center[-1] == "." else None,
            num_left_tokens=0, num_right_tokens=0
        )
        ques_transformer = ptoken.TokenTransformer(
            lambda left, center, right: [center[:-1]] if center[-1] == "?" else None,
            num_left_tokens=0, num_right_tokens=0
        )

        self._doc1.transform_tokens([comma_transformer, ques_transformer])
        self._doc1.save_as_txt("babi_sample/qa1-punctuations.txt")

        #
        # 3. Create SeqDocument split by flag tokens
        #
        seq_doc1 = pseq.SeqDocument.create_flag_separated_seq_doc(self._doc1)
        seq_doc1.save_as_txt("babi_sample/qa1-seq.txt", "babi_sample/qa1-flag.txt")

        #
        # 4. iter with updated flag tokens
        #

        def babi_qa_token_transform_f(left, center, right):
            if left[0] == "\n" and center == "\t":
                return "question"
            elif left[0] == "\t" and center == "\t":
                return "answer"
            elif left[0] == "\t" and center == "\n":
                return "support_id"
            else:
                return None
        qa_flag_token_transformer = ptoken.TokenTransformer(
            babi_qa_token_transform_f, num_left_tokens=1, num_right_tokens=0
        )
        qa_flag_context_token_transformer = ptoken.TokenTransformer(
                lambda left, center, right: "context" if center == "\n" else None,
                num_left_tokens=0, num_right_tokens=0
        )
        seq_doc1.transform_flags([qa_flag_token_transformer,
                                  qa_flag_context_token_transformer])
        # context, question, answer
        seq_doc1.save_as_txt("babi_sample/qa1-seq-updated.txt", "babi_sample/qa1-flag-updated.txt")

        #
        # 5. save with transformer for qa
        #
        len_dict = seq_doc1.get_max_len_dict() 
        with open('babi_sample/len_dict.json', 'w') as outfile:
            json.dump(len_dict, outfile)
        qa_transformer = QueAnsTransformer(
            q_max_len=len_dict["max_length"]["question"],
            cs_max_size=len_dict["max_cont"]["context"],
            c_max_len=len_dict["max_length"]["context"],
            a_max_len=len_dict["max_length"]["answer"]
        )
        doc_transform_state = DocumentTransformState(
                docs=[seq_doc1], transformer=qa_transformer, size=1000
            )

        # txt serialization
        ptxt.seq_docs_transformed_save(
            doc_transform_state,
            (
                "babi_sample/question.txt",
                "babi_sample/contexts.txt",
                "babi_sample/answer.txt",
                "babi_sample/q_len.txt",
                "babi_sample/cs_size.txt",
                "babi_sample/a_len.txt",
                "babi_sample/c_lens.txt"
            )
            )

        # tfrecords serialization
        ptfrecords.seq_docs_transformed_save(
            doc_transform_state,
            "babi_sample/qa.tfrecords"
        )

        with open("babi_sample/len_dict.json") as json_data:
            context_len = json.load(json_data)["max_length"]["context"]

        initializer, qa_iters = pdtfrecords_iter.simple_iterator(
                                    "babi_sample/qa.tfrecords",
                                    pdtfrecords_qa.create_qa_parse_f(context_len))
        iter2 = qa_transformer.transform_seq_docs(seq_doc1)
        with tf.Session() as sess:
            sess.run(initializer)
            i = 0
            while True:
                try:
                    q, cs, a, q_len, cs_size, a_len, c_lens = sess.run(qa_iters)
                    q_, cs_, a_, q_len_, cs_size_, a_len_, c_lens_ = next(iter2)
                    for token, token_ in zip(q, q_):
                        self.assertEqual(token.decode(), token_)
                    for c, c_ in zip(cs, cs_):
                        for token, token_ in zip(c, c_):
                            self.assertEqual(token.decode(), token_)
                    for token, token_ in zip(a, a_):
                        self.assertEqual(token.decode(), token_)
                    self.assertEqual(q_len, q_len_)
                    self.assertEqual(cs_size, cs_size_)
                    self.assertEqual(a_len, a_len_)
                    for c_len, c_len_ in zip(c_lens, c_lens_):
                        self.assertEqual(c_len, c_len_)
                    i += 1
                except tf.errors.OutOfRangeError:
                    self.assertEqual(i, 1000)
                    break


if __name__ == '__main__':
    unittest.main()
