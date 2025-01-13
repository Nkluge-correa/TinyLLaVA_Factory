from transformers import LlamaForCausalLM, AutoTokenizer

from . import register_llm

@register_llm('tucano')
def return_tucanoclass():
    def tokenizer_and_post_load(tokenizer):
        return tokenizer
    return LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load)
