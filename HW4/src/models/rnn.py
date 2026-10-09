import torch.nn as nn

from .base import SequenceClassifier


class RNNClassifier(SequenceClassifier):
    """最基本的 Elman RNN（tanh）文字分類器，作為 LSTM 的比較基準。"""

    def __init__(self, vocab_size, **kwargs):
        super().__init__(nn.RNN, vocab_size, **kwargs)
