from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
from packaging import version

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template
from ...utils.constants import *

from transformers import PreTrainedTokenizer
import torch
import tokenizers

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

system = "Um bate-papo entre um usuário curioso e um assistente de inteligência artificial. O assistente dá respostas úteis, detalhadas e educadas às perguntas do usuário."

@register_template('llama')
@dataclass
class LlamaTemplate(Template):
    format_image_token: Formatter = field(default_factory=lambda: StringFormatter(slot="<image>\n{{content}}"))
    format_user: Formatter = field(default_factory=lambda: StringFormatter(slot="Usuário: {{content}} "))
    format_assistant: Formatter = field(default_factory=lambda: StringFormatter(slot="Assistente: {{content}}</s>"))
    system: Formatter = field(default_factory=lambda: EmptyFormatter(slot=system + " "))
    separator: Formatter = field(default_factory=lambda: EmptyFormatter(slot=[' Assistente: ', '</s>']))
    
    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):

        """
        #### Masks only the default image token ####
        cur_len = 0 # no bos_token
        eos_token_length = 1
        labels[:] = input_ids[:]
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + eos_token_length
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1
            cur_len += round_len
        
        labels[labels == -200] = -100
        """
        #### Maskes the input_ids to ignore the instruction part of the prompt ####
        cur_len = 0 # no bos_token
        eos_token_length = 1
        labels[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        

        labels[cur_len:] = IGNORE_INDEX
        

        return labels, cur_len

