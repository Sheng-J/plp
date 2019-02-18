def batched_items_iter(batch_size, *items):
    batched_items = []
    for item in items:
        batched_items.append(item)
        if len(batched_items) == batch_size:
            yield tuple(batched_items)
            batched_items = []


def merged_round_iter(*iters):
    while len(iters) != 0:
        next_iters = []
        for iterator in iters:
            try:
                res_tuple = next(iterator)
                next_iters.append(iterator)
                yield res_tuple
            except StopIteration:
                pass
        iters = next_iters


def merged_round_iter_with_label(iters, labels):
    while len(iters) != 0:
        next_iters = []
        next_labels = []
        for iterator, label in zip(iters, labels):
            try:
                res_tuple = next(iterator)
                next_iters.append(iterator)
                next_labels.append(label)
                yield res_tuple, label
            except StopIteration:
                pass
        iters = next_iters
        labels = next_labels



def limit_iter(iterator, max_num_examples):
    i = 0
    for tuple_ in iterator:
        yield tuple_
        i += 1
        if i >= max_num_examples:
            break
    if i < max_num_examples:
        print("Only found " + str(i) + " examples when expected "
              + str(max_num_examples))
