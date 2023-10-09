from transformers import AutoTokenizer, AutoModel
import constants as c
from tokenizers import ByteLevelBPETokenizer


def getTokenizer():
    # model_name = "gpt2"
    # model_name = "bert-base-uncased"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tempNewTokens = c.NEW_TOKENS
    # tempNewTokens.append("<unknown>")
    # new_tokens = list(set(tempNewTokens) - set(tokenizer.get_vocab().keys()))
    # tokenizer.add_tokens(new_tokens)

    tokenizer = ByteLevelBPETokenizer(
        "tokenizers/chef_tokenizer-vocab.json",
        "tokenizers/chef_tokenizer-merges.txt",
    )

    tokenizer.add_tokens(c.NEW_TOKENS_NO_SPACES)

    return tokenizer
