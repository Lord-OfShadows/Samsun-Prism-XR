"""The classifier: (64 frames, 227 features) -> probabilities over the vocabulary.

Shape of the network (arch="conv_attn", the default):
  stem                 a Dense layer lifts each frame's 227 numbers to `dim` channels
  3 x conv block       depthwise-separable 1D convolutions: read local motion (a few frames)
  1 x attention block  self-attention: relate any moment of the sign to any other
  3 x conv block
  1 x attention block
  average over time -> dropout -> Dense -> softmax

Why depthwise-separable: a normal 1D conv mixes time and channels in one big weight
tensor. Splitting it into "filter each channel along time" + "mix channels at one instant"
gives nearly the same result for a fraction of the parameters and multiply-adds.

Choices made for clean INT8 conversion:
  - BatchNorm instead of LayerNorm everywhere (BatchNorm folds into the neighbouring layer
    at conversion; LayerNorm becomes a chain of ops that quantises badly).
  - ReLU activations, and attention written with plain Dense / matmul / softmax ops, all of
    which have integer TFLite kernels.
arch="conv" is the PRD's fallback (same network without the attention blocks).
"""
import keras
from keras import layers, ops

from src.config import SEQ_LEN
from src.normalize import N_FEATURES


@keras.saving.register_keras_serializable(package="echogesture")
class SelfAttention(layers.Layer):
    """Multi-head self-attention over the time axis, built from INT8-friendly ops."""

    def __init__(self, dim, heads=4, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.heads = dim, heads
        self.q = layers.Dense(dim, use_bias=False)
        self.k = layers.Dense(dim, use_bias=False)
        self.v = layers.Dense(dim, use_bias=False)
        self.out = layers.Dense(dim)

    def _split(self, x, t):   # (B, T, dim) -> (B, heads, T, dim/heads)
        return ops.transpose(ops.reshape(x, (-1, t, self.heads, self.dim // self.heads)), (0, 2, 1, 3))

    def call(self, x):
        t = x.shape[1]
        q, k, v = self._split(self.q(x), t), self._split(self.k(x), t), self._split(self.v(x), t)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * ((self.dim // self.heads) ** -0.5)
        y = ops.matmul(ops.softmax(scores, axis=-1), v)                 # (B, heads, T, d)
        y = ops.reshape(ops.transpose(y, (0, 2, 1, 3)), (-1, t, self.dim))
        return self.out(y)

    def get_config(self):
        return {**super().get_config(), "dim": self.dim, "heads": self.heads}


def conv_block(x, dim, kernel, drop):
    """Inverted-bottleneck separable conv: widen, filter along time per channel, narrow, add."""
    y = layers.Dense(dim * 2, use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.DepthwiseConv1D(kernel, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Dense(dim, use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(drop)(y)
    return layers.Add()([x, y])


def attention_block(x, dim, heads, drop):
    y = layers.BatchNormalization()(x)
    y = SelfAttention(dim, heads)(y)
    x = layers.Add()([x, layers.Dropout(drop)(y)])
    y = layers.BatchNormalization()(x)
    y = layers.Dense(dim * 2, activation="relu")(y)
    y = layers.Dense(dim)(y)
    return layers.Add()([x, layers.Dropout(drop)(y)])


def build_model(n_classes, arch="conv_attn", dim=192, kernel=9, heads=4, drop=0.2,
                head_drop=0.4, seq_len=SEQ_LEN, n_features=N_FEATURES):
    if arch not in ("conv_attn", "conv"):
        raise ValueError(f"unknown arch {arch!r}")
    inp = keras.Input((seq_len, n_features), name="landmarks")
    x = layers.Dense(dim, use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    for _ in range(2):                       # two stages
        for _ in range(3):
            x = conv_block(x, dim, kernel, drop)
        if arch == "conv_attn":
            x = attention_block(x, dim, heads, drop)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(head_drop)(x)
    out = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name=f"echogesture_{arch}")
