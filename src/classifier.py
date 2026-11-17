"""Runs the INT8 .tflite classifier with the LiteRT interpreter. Used by predict(), the
benchmark, the robustness evaluation and the live demo, so the INT8 input scaling lives
in exactly one place.

A full-integer model does not accept floats. Its input tensor carries two numbers, a
scale and a zero point, that define the mapping
    real_value = scale * (int8_value - zero_point)
so before every call we convert features with  q = round(x / scale) + zero_point  and
clamp to [-128, 127]; after the call we map the int8 output back the same way. Getting
this wrong does not crash, it just silently gives garbage predictions, hence the tests.
"""
import numpy as np
from ai_edge_litert.interpreter import Interpreter

from src import config


def quantize(x, scale, zero_point):
    """float -> int8 using a tensor's (scale, zero_point). Saturates instead of wrapping."""
    q = np.round(np.asarray(x, np.float32) / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize(q, scale, zero_point):
    """int8 -> float, the inverse mapping."""
    return (np.asarray(q, np.float32) - zero_point) * scale


class Int8Classifier:
    def __init__(self, model_path=None, num_threads=1):
        self.path = str(model_path or config.INT8_MODEL)
        self.num_threads = num_threads
        self.interp = Interpreter(model_path=self.path, num_threads=num_threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        if self.inp["dtype"] != np.int8 or self.out["dtype"] != np.int8:
            raise ValueError(f"{self.path} is not a full-integer INT8 model "
                             f"(input {self.inp['dtype']}, output {self.out['dtype']})")
        self.in_scale, self.in_zero = self.inp["quantization"]
        self.out_scale, self.out_zero = self.out["quantization"]
        self.n_classes = int(self.out["shape"][-1])

    def probs_from_int8(self, q):
        """int8 (1, 64, 227) -> float probabilities (n_classes,). The timed part."""
        self.interp.set_tensor(self.inp["index"], q)
        self.interp.invoke()
        return dequantize(self.interp.get_tensor(self.out["index"])[0], self.out_scale, self.out_zero)

    def probs(self, features):
        """float features (64, 227) -> float probabilities (n_classes,)."""
        return self.probs_from_int8(quantize(features[None], self.in_scale, self.in_zero))

    def probs_batch(self, features):
        """(N, 64, 227) -> (N, n_classes), one clip at a time (the model has batch size 1)."""
        return np.stack([self.probs(f) for f in features])
