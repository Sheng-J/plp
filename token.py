

def assert_type_valid(token_type):
    assert token_type == "word_type" or token_type == "id_type" \
        or token_type == "value_int_type" or token_type == "embed_type" \
        or token_type == "value_float_type" or token_type == "list_type"


def assert_type_doc_valid(token_type):
    assert token_type == "word_type" or token_type == "id_type" \
        or token_type == "embed_type"


def is_num(token):
    return token.lstrip('-').replace('.', '', 1).isdigit()


def create_word2id_gen_f(vocab_reader, flag_tokens):
    def word2id_gen(token_iter):
        for word_token in token_iter:
            if word_token in flag_tokens:
                yield word_token
            else:
                yield vocab_reader.word2id(word_token)
    return word2id_gen


def create_id2word_gen_f(vocab_reader, flag_tokens):
    def id2word_gen(token_iter):
        for id_token in token_iter:
            if id_token in flag_tokens:
                yield id_token
            else:
                yield vocab_reader.id2word(id_token)
    return id2word_gen


def create_word2embed_gen_f(embed_reader, flag_tokens):
    def word2embed_gen(token_iter):
        for word_token in token_iter:
            if word_token in flag_tokens:
                yield word_token
            else:
                yield embed_reader.word2embed_lookup_f(word_token)
    return word2embed_gen


def create_id2embed_gen_f(embed_reader, flag_tokens):
    def id2embed_gen(token_iter):
        for id_token in token_iter:
            if id_token in flag_tokens:
                yield id_token
            else:
                yield embed_reader.id2embed_lookup_f(id_token)
    return id2embed_gen


###################
# Token Transform #
###################
class TokenTransformer:
    @property
    def num_left_tokens(self):
        return self._num_left_tokens

    @property
    def num_right_tokens(self):
        return self._num_right_tokens

    def __init__(self, token_transformer_f, num_left_tokens, num_right_tokens):
        self._num_left_tokens = num_left_tokens
        self._num_right_tokens = num_right_tokens
        self._token_transformer_f = token_transformer_f

    def __getitem__(self, tokens_tuple):
        left, center, right = tokens_tuple
        assert len(left) >= self._num_left_tokens
        assert len(right) >= self._num_right_tokens
        left = left[len(left)-self._num_left_tokens-1:]
        right = right[:self._num_right_tokens]
        return self._token_transformer_f(left, center, right)

    def is_applicable(self, left_len, right_len):
        left_valid = (left_len == self._num_left_tokens)
        right_valid = (right_len == self._num_right_tokens)
        return left_valid and right_valid


def get_transformers_max_num_tokens(token_transformers):
    max_num_left = 0
    max_num_right = 0
    for transformer in token_transformers:
        if transformer.num_left_tokens > max_num_left:
            max_num_left = transformer.num_left_tokens
        if transformer.num_right_tokens > max_num_right:
            max_num_right = transformer.num_right_tokens
    return max_num_left, max_num_right


def shift_context_center_list_tokens(tokens_lists, token_gen, max_num_left):
    """
    changes left right in-place
    right context must be initialized with correct dimension
    if empty, center will be updated with iter, and right is not updated
    """
    left, center, right = tokens_lists
    num_elem = len(left)
    assert num_elem == len(center) and num_elem == len(right)
    left_len, right_len = len(left[0]), len(right[0])
    if max_num_left > 0:
        for left_list, center_elem in zip(left, center):
            left_list.append(center_elem)
    if left_len > max_num_left:
        for left_list in left:
            left_list.pop(0)

    if right_len == 0:
        try:
            center = next(token_gen)
        except StopIteration:
            center = [None for _ in range(num_elem)]
    else:
        center = [right[i].pop(0) for i in range(num_elem)]
        try:
            right_elems = next(token_gen)
            for right_list, right_elem in zip(right, right_elems):
                right_list.append(right_elem)
        except StopIteration:
            pass
    return center


def shift_context_center_tokens(tokens_list, token_gen, max_num_left):
    """
    changes left right in-place
    right context must be initialized with correct dimension
    if empty, center will be updated with iter, and right is not updated
    """
    left, center, right = tokens_list
    left_len, right_len = len(left), len(right)
    if max_num_left > 0:
        left.append(center)
    if left_len > max_num_left:
        left.pop(0)

    if right_len == 0:
        try:
            center = next(token_gen)
        except StopIteration:
            center = None
    else:
        center = right.pop(0)
        try:
            right.append(next(token_gen))
        except StopIteration:
            pass
    return center
