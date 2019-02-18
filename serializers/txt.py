import plp.token as ptoken
import plp.vocab as pvocab
from plp.utils.file import MultiWriteOpen
from plp.utils.iterator import limit_iter
import pdb

# GEN_YIELD_EOL = "gen_yield_eol"     # "\n" will be replaced with EOS token
# GEN_IGNORE_EOL = "gen_ignore_eol"   # "\n" will be skipeed
# GEN_KEEP_EOL_NL = "gen_keep_eol_nl" # "\n" will be preserved & iterated
                                    # However, it is "flag_token"

##############
# Gen module #
##############


def doc_gen_f_yield_eol(doc_path, token_type, isolating_tokens=None):
    if token_type == "word_type":
        eol = pvocab.EOS
    elif token_type == "id_type":
        eol = pvocab.EOS_ID
    else:
        raise NotImplementedError("Not supported token type")
    return _doc_gen_f(doc_path, token_type, eol, isolating_tokens)


def doc_gen_f_ignore_eol(doc_path, token_type, isolating_tokens=None):
    return _doc_gen_f(doc_path, token_type, isolating_tokens=isolating_tokens)


def doc_gen_f_keep_eol_nl(doc_path, token_type, isolating_tokens=None):
    return _doc_gen_f(doc_path, token_type, "\n", isolating_tokens)

def doc_gen_f_split_eol(doc_path, token_type, isolating_token=None):
    def gen_f(line):
        if isolating_tokens:
            for isolating_token in isolating_tokens:
                # slow, not efficient here for now :(
                line = line.replace(isolating_token, " " + isolating_token + " ")
        tokens = line.strip().split(" ")
        for token in tokens:
            if token == "":
                continue
            yield convert_f(token)
    with open(doc_path) as f:
        gen_fs = [gen_f(line) for line in f]
    return gen_fs


def _doc_gen_f(doc_path, token_type, eol=None, isolating_tokens=None):
    def doc_gen():
        # token type check?
        convert_f = get_convert_f(token_type)
        with open(doc_path) as f:
            for line in f:
                if isolating_tokens:
                    for isolating_token in isolating_tokens:
                        # slow, not efficient here for now :(
                        line = line.replace(isolating_token, " " + isolating_token + " ")
                tokens = line.strip().split(" ")
                for token in tokens:
                    if token == "":
                        continue
                    yield convert_f(token)
                # Handle the end of line if the doc is language based
                if (eol is not None) and len(tokens) > 0:
                    yield eol
    return doc_gen


def get_convert_f(token_type):
    if token_type == "value_int_type" or \
       token_type == "value_id_type":
        return lambda x: int(float(x))
    elif token_type == "value_float_type":
        return lambda x: float(x)
    else:
        return lambda x: x


###############
# Save module #
###############


def doc_save(doc_path, doc_iter):
    with open(doc_path, "w") as f:
        for token in doc_iter:
            f.write(token)
            if token != "\n":
                f.write(" ")


def doc_save_by_line(doc_path, doc_iter, num_tokens_per_line, token_type):
    with open(doc_path, "w") as f:
        for i, token in enumerate(doc_iter):
            if token == "\n":
                if token_type == ptoken.WORD_TYPE:
                    f.write(pvocab.EOS)
                elif token_type == ptoken.ID_TYPE:
                    f.write(pvocab.EOS_ID)
                else:
                    raise ValueError("No such token type")
            else:
                f.write(token)
            f.write(" ")
            if i % num_tokens_per_line == 0:
                f.write("\n")


def seq_doc_save(seq_path, flag_path, seq_flag_iter):
    with open(seq_path, "w") as seq_f, open(flag_path, "w") as flag_f:
        for seq, flag in seq_flag_iter:
            seq_f.write(" ".join([str(token) for token in seq]))
            seq_f.write("\n")
            if type(flag) == tuple:
                print(seq, flag)
            if flag == "\n":
                flag_f.write("\\n")
            elif flag == "\t":
                flag_f.write("\\t")
            else:
                flag_f.write(flag)
            flag_f.write("\n")


def docs_transformed_save(doc_transform_state, save_paths):
    iterator = doc_transform_state.transformer.transform_docs(
        *doc_transform_state.docs
    )
    lists_stats = doc_transform_state.transformer.get_lists_stats(
        *doc_transform_state.docs
    )
    _transformed_save(iterator, lists_stats, save_paths, doc_transform_state.size)


def seq_docs_transformed_save(doc_transform_state, save_paths):
    iterator = doc_transform_state.transformer.transform_seq_docs(
        *doc_transform_state.docs
    )
    lists_stats = doc_transform_state.transformer.get_lists_stats(
        *doc_transform_state.docs
    )
    _transformed_save(iterator, lists_stats, save_paths, doc_transform_state.size)


def _transformed_save(lists_iter, lists_stats, save_paths, size):
    if len(lists_stats) != len(save_paths):
        raise ValueError("seq gen num should match num of save paths")
    with MultiWriteOpen(*save_paths) as fs:
        for lists in limit_iter(lists_iter, size):
            for f, (t_list, list_stat) in zip(fs, zip(lists, lists_stats)):
                token_type = list_stat.token_type
                if token_type == ptoken.LIST_TYPE:
                    for sub_list in t_list:
                        if list_stat.sub_list_stat.token_type == ptoken.LIST_TYPE:
                            raise ValueError("Not supported")
                        else:
                            if list_stat.sub_list_stat.token_type == ptoken.WORD_TYPE:
                                f.write(" ".join(sub_list))
                            else:
                                f.write(" ".join([str(token) for token in sub_list]))

                        f.write("\t")
                else:
                    if token_type == ptoken.WORD_TYPE:
                        f.write(" ".join(t_list))
                    else:
                        f.write(" ".join([str(token) for token in t_list]))
                f.write("\n")
