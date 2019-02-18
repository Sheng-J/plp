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



class TestDocLabel(unittest.TestCase):
    pass



