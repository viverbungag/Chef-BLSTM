import re
from transformers import BertTokenizer, GPT2Tokenizer
import constants as c


class Text2:
    def __init__(self, input_text, token2ind=None, ind2token=None):
        self.content = input_text
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokens, self.tokens_distinct = self.tokenize()

        if token2ind != None and ind2token != None:
            self.token2ind, self.ind2token = token2ind, ind2token

        self.tokens_ind = [
            self.token2ind[token]
            if token in self.token2ind.keys()
            else self.token2ind["<unknown>"]
            for token in self.tokens
        ]

    def __repr__(self):
        return self.content

    def __len__(self):
        return len(self.tokens_distinct)

    # @staticmethod
    # def create_word_mapping(values_list):
    #     values_list.append("<unknown>")
    #     value2ind = {value: ind for ind, value in enumerate(values_list)}
    #     ind2value = dict(enumerate(values_list))
    #     return value2ind, ind2value

    def preprocess(self):
        punctuation_pad = "!?.,:-;"
        punctuation_remove = ""

        self.content_preprocess = re.sub(r"(\S)(\n)(\S)", r"\1 \2 \3", self.content)
        self.content_preprocess = self.content_preprocess.translate(
            str.maketrans("", "", punctuation_remove)
        )
        self.content_preprocess = self.content_preprocess.translate(
            str.maketrans({key: " {0} ".format(key) for key in punctuation_pad})
        )
        self.content_preprocess = re.sub(" +", " ", self.content_preprocess)
        self.content = self.content_preprocess.strip()

    def tokenize(self):
        self.preprocess()
        tokenized_text = []
        for token in self.content.split(" "):
            if token in c.NEW_TOKENS_NO_SPACES:
                tokenized_text.append(token)
                continue
            tokenized_text.extend(self.tokenizer.tokenize(token))
        return tokenized_text, list(set(tokenized_text))

    # def tokenizeGPT2(self):
    #     self.preprocess()
    #     tokens = self.tokenizer.tokenize(self.content)
    #     return tokens, list(set(tokens))

    def tokens_info(self):
        print(
            "total tokens: %d, distinct tokens: %d"
            % (len(self.tokens), len(self.tokens_distinct))
        )

    # def print_needs(self):
    #     print(self.tokens[0] + "hello2")
