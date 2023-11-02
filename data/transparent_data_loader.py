"""
This dataloader inherently represents also a training session that can be saved and loaded
"""


class TransparentDataLoader:
    def __init__(self):
        super(TransparentDataLoader, self).__init__()

    # initialize the training on a specific epoch
    def init_epoch(self, epoch, batch_size):
        raise NotImplementedError

    def get_next_batch(self):
        raise NotImplementedError

    def set_epoch_it(self, epoch, verbose=False):
        assert epoch < len(
            self.array_of_init_seeds
        ), "requested epoch higher than the maximum: " + str(
            len(self.array_of_init_seeds)
        )
        self.epoch_it = epoch
        self.init_epoch(epoch_it=self.epoch_it, verbose=verbose)

    def get_epoch_it(self):
        return self.epoch_it

    def get_num_epoch(self):
        return self.max_num_epoch

    def get_num_batches(self):
        return self.num_batches

    def set_batch_it(self, batch_it):
        self.batch_it[self.rank] = batch_it

    def get_batch_it(self):
        return self.batch_it[self.rank]

    def change_batch_size(self, batch_size, verbose):
        self.batch_size = batch_size
        self.set_epoch_it(epoch=0, verbose=verbose)
        self.set_batch_it(batch_it=0)

    def get_batch_size(self):
        return self.batch_size

    def save_state(self):
        return {
            "batch_it": self.batch_it[self.rank],
            "epoch_it": self.epoch_it,
            "batch_size": self.batch_size,
            "array_of_init_seed": self.array_of_init_seeds,
        }

    def load_state(self, state):
        self.array_of_init_seeds = state["array_of_init_seed"]
        self.batch_size = state["batch_size"]
        self.set_epoch_it(state["epoch_it"])
        self.batch_it[self.rank] = state["batch_it"]

    def add_pad_according_to_batch(self, batch_sentences, pad_symbol):
        batch_size = len(batch_sentences)
        list_of_lengthes = [
            len(batch_sentences[batch_idx]) for batch_idx in range(batch_size)
        ]
        in_batch_max_seq_len = max(list_of_lengthes)
        batch_num_pads = []
        new_batch_sentences = []
        for batch_idx in range(batch_size):
            num_pads = in_batch_max_seq_len - len(batch_sentences[batch_idx])
            new_batch_sentences.append(
                batch_sentences[batch_idx] + [pad_symbol] * (num_pads)
            )
            batch_num_pads.append(num_pads)
        return new_batch_sentences, batch_num_pads
