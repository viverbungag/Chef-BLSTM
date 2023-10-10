import re
import constants as c
import tokenizer_vars as tv


class Text:
    def __init__(self, input_text):
        self.content = input_text
        tokenizer = tv.getTokenizer()
        self.tokens_ind = tokenizer.encode(self.content).ids

    def __repr__(self):
        return self.content

    def __len__(self):
        return len(self.tokens_distinct)

    def tokens_info(self):
        print("total indices: %d" % (len(self.tokens_ind)))
