from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .tsrnn import NETWORKS, TSRNN
from ..module import PositionEmbedding, RelativePositionalEncodingSelfAttention, ExpSelfAttention


def generate_square_subsequent_mask(sz, prev):
    mask = (torch.triu(torch.ones(sz, sz), -prev) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


class BatchFirstTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, batch_first=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(self, x, *args, **kwargs):
        if self.batch_first:
            x = x.transpose(0, 1)
        return super().forward(x, *args, **kwargs).transpose(0, 1)


@NETWORKS.register_module()
class TSTransformer(TSRNN):
    def __init__(
        self,
        emb_dim: int,
        emb_type: str,
        hidden_size: int,
        dropout: float,
        num_layers: int = 1,
        num_heads: int = 4,
        is_bidir: bool = False,
        use_last: bool = False,
        max_length: Optional[int] = 100,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        """
        The Transformer network for time-series prediction.

        Args:
            emb_dim: embedding dimension.
            emb_type: "static" or "learn", static or learnable embedding.
            hidden_size: hidden size of the RNN cell.
            dropout: dropout rate.
            num_layers: number of self-attention layers.
            num_heads: number of self-attention heads.
            is_bidir: whether to use bidirectional transformer (without mask).
            max_length: maximum length of the input sequence.
            input_size: input dimension of the time-series data.
            weight_file: path to the pretrained model.

        Raises:
            ValueError: If `emb_type` is not supported.
        """
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        encoder_layers = BatchFirstTransformerEncoderLayer(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.temporal_encoder = TransformerEncoder(encoder_layers, num_layers)

        if emb_dim != 0:
            self.emb = PositionEmbedding(emb_type, input_size, max_length, dropout=dropout)

        self.is_bidir = is_bidir
        self.use_last = use_last
        self.emb_dim = emb_dim
        self.__output_size = hidden_size
        self.__hidden_size = hidden_size

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    def forward(self, inputs):
        # positional encoding
        if self.emb_dim > 0:
            inputs = self.emb(inputs)

        # non-regressive encoder
        z = self.encoder(inputs)

        # mask generation
        mask = generate_square_subsequent_mask(z.size()[1], 0)
        if torch.cuda.is_available():
            mask = mask.cuda()
        if self.is_bidir:
            mask = torch.zeros_like(mask)

        # regressive encoder
        attn_outs = self.temporal_encoder(z, mask)

        if self.use_last:
            out = attn_outs[:, -1, :]
        else:
            out = attn_outs.mean(dim=1)

        return attn_outs, out

    def get_cpc_repre(self, inputs):
        """
        Get the representation of the input sequence for cpc pre-training.
        """
        assert (
            self.is_bidir is False
        ), "Conduct CPC pre-training on bidirectional transformer would cause information leakage."

        # positional encoding
        if self.emb_dim > 0:
            inputs = self.emb(inputs)

        # non-regressive encoder
        z = self.encoder(inputs)

        # mask generation
        mask = generate_square_subsequent_mask(z.size()[1], 0)
        if torch.cuda.is_available():
            mask = mask.cuda()

        # regressive encoder
        attn_outs = self.temporal_encoder(z, mask)

        return z, attn_outs

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


@NETWORKS.register_module()
class RelativeTSTransformer(TSTransformer):
    def __init__(
        self,
        emb_dim: int,
        emb_type: str,
        hidden_size: int,
        dropout: float,
        num_layers: int = 1,
        num_heads: int = 4,
        is_bidir: bool = False,
        use_last: bool = False,
        max_length: Optional[int] = 100,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        encoder_layers = RelativePositionalEncodingSelfAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True, max_len=max_length
        )
        self.temporal_encoder = TransformerEncoder(encoder_layers, num_layers)

        if emb_dim != 0:
            self.emb = PositionEmbedding(emb_type, input_size, max_length)

        self.is_bidir = is_bidir
        self.use_last = use_last
        self.emb_dim = emb_dim
        self.__output_size = hidden_size
        self.__hidden_size = hidden_size

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


@NETWORKS.register_module()
class ExpTSTransformer(TSTransformer):
    def __init__(
        self,
        emb_dim: int,
        emb_type: str,
        hidden_size: int,
        dropout: float,
        alpha: float = 0.5,
        num_layers: int = 1,
        num_heads: int = 4,
        is_bidir: bool = False,
        use_last: bool = False,
        max_length: Optional[int] = 100,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        encoder_layers = ExpSelfAttention(
            hidden_size, num_heads, alpha=alpha, dropout=dropout, batch_first=True, max_len=max_length
        )
        self.temporal_encoder = TransformerEncoder(encoder_layers, num_layers)

        if emb_dim != 0:
            self.emb = PositionEmbedding(emb_type, input_size, max_length)

        self.is_bidir = is_bidir
        self.use_last = use_last
        self.emb_dim = emb_dim
        self.__output_size = hidden_size
        self.__hidden_size = hidden_size

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size
