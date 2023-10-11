import re
import numpy as np
import keras
from transformers import BertTokenizer, GPT2Tokenizer
import tokenizer_vars as tv


class Sequences:
    def __init__(self, tokens_ind, max_len, step):
        self.tokens_ind = tokens_ind
        self.max_len = max_len
        self.step = step
        self.sequences, self.next_words = self.create_sequences()

    def __repr__(self):
        return "Sequence object of max_len: %d and step: %d" % (self.max_len, self.step)

    def __len__(self):
        return len(self.sequences)

    def create_sequences(self):
        sequences = []
        next_words = []

        for i in range(0, len(self.tokens_ind) - self.max_len, self.step):
            sequences.append(self.tokens_ind[i : i + self.max_len])
            next_words.append(self.tokens_ind[i + self.max_len])
        return sequences, next_words

    def sequences_info(self):
        print(
            "number of sequences of length %d: %d" % (self.max_len, len(self.sequences))
        )


class ModelPredict:
    def __init__(
        self, model, tokens_ind_prefix, input_prefix, max_len, embedding=False
    ):
        self.tokenizer = tv.getTokenizer()
        self.model = model
        self.max_len = max_len
        self.input_prefix = input_prefix
        self.original_tokens_ind = tokens_ind_prefix.copy()
        self.tokens_ind = tokens_ind_prefix.copy()
        self.embedding = embedding

    def __repr__(self):
        return self.input_prefix

    def single_data_generation(self):
        single_sequence = np.zeros(
            (1, self.max_len, len(tokenizer.get_vocab_size())), dtype=np.bool
        )
        prefix = self.tokens_ind[-self.max_len :]

        for i, s in enumerate(prefix):
            single_sequence[0, i, s] = 1
        return single_sequence

    def model_predict(self):
        if self.embedding:
            model_input = np.array(self.tokens_ind).reshape(1, -1)
        else:
            model_input = self.single_data_generation()
        return self.model.predict(model_input)[0]

    @staticmethod
    def add_prob_temperature(prob, temperature=1):
        prob = prob.astype(float)
        prob_with_temperature = np.exp(
            np.where(prob == 0, 0, np.log(prob + 1e-10)) / temperature
        )
        prob_with_temperature /= np.sum(prob_with_temperature)
        return prob_with_temperature

    @staticmethod
    def reverse_preprocess(text):
        text_reverse = re.sub(r'\s+([!?"\'().,;-])', r"\1", text)
        text_reverse = re.sub(" +", " ", text_reverse)
        return text_reverse

    def return_next_word(self, temperature=1):
        prob = self.model_predict()

        prob_with_temperature = self.add_prob_temperature(prob, temperature)
        next_word = np.random.choice(
            len(prob_with_temperature), p=prob_with_temperature
        )

        return next_word

    def generate_sequence(self, temperature=1):
        counter = 0
        while True:
            next_word = self.return_next_word(temperature=temperature)
            self.tokens_ind.append(next_word)
            if "<RECIPE_END>" in self.tokenizer.decode([next_word]):
                break
            counter += 1
            if counter > 500:
                break

        # if not append:
        # self.tokens_ind = self.original_tokens_ind.copy()

        return self.tokenizer.decode(self.tokens_ind)

    def bulk_generate_sequence(self, k, n, temperature=1):
        for i in range(n):
            print(self.generate_sequence(k, temperature=temperature))
            print("\n")


class TextDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        sequences,
        next_words,
        sequence_length,
        vocab_size,
        batch_size=32,
        shuffle=True,
        embedding=False,
    ):
        self.batch_size = batch_size
        self.sequences = sequences
        self.next_words = next_words
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.embedding = embedding
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        sequences_batch = [self.sequences[k] for k in indexes]
        next_words_batch = [self.next_words[k] for k in indexes]

        if self.embedding:
            X = np.array(sequences_batch)
            y = keras.utils.to_categorical(
                next_words_batch, num_classes=self.vocab_size
            )
        else:
            X, y = self.__data_generation(sequences_batch, next_words_batch)

        return X, y

    def getItem(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # print(indexes)
        sequences_batch = [self.sequences[k] for k in indexes]
        next_words_batch = [self.next_words[k] for k in indexes]
        print(next_words_batch)

        if self.embedding:
            X = np.array(sequences_batch)
            y = keras.utils.to_categorical(
                next_words_batch, num_classes=self.vocab_size
            )
        else:
            X, y = self.__data_generation(sequences_batch, next_words_batch)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sequences_batch, next_words_batch):
        X = np.zeros(
            (self.batch_size, self.sequence_length, self.vocab_size), dtype=np.bool
        )
        y = np.zeros((self.batch_size, self.vocab_size), dtype=np.bool)

        for i, seq in enumerate(sequences_batch):
            for j, word in enumerate(seq):
                X[i, j, word] = 1
                y[i, next_words_batch[i]] = 1
        return X, y
