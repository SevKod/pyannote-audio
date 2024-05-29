# MIT License
#
# Copyright (c) 2023- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from functools import lru_cache
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from functools import cached_property
from pyannote.core.utils.generators import pairwise
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel,Wav2Vec2BertConfig
from pyannote.core import SlidingWindow
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)



class w2vbert(Model):
    """Self-Supervised Representation for Speaker Segmentation

    wav2vec > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    wav2vec: dict or str, optional
        Defaults to "WAVLM_BASE".
    wav2vec_layer: int, optional
        Index of layer to use as input to the LSTM.
        Defaults (-1) to use average of all layers (with learnable weights).
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 4, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    WAV2VEC_DEFAULTS = "WAVLM_BASE"

    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 4,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        wav2vec: Union[dict, str] = None,
        wav2vec_layer: int = -1,
        lstm: Optional[dict] = None,
        linear: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        
        configuration = Wav2Vec2BertConfig()
        configuration.output_hidden_states=True
        wav2vec_num_layers = configuration.num_hidden_layers
        wav2vec_dim = configuration.output_hidden_size
        cache = '/home/severin.baroudi/work/ssl/DPRNN/WSJ0-2MIX/speechbrain/separation'
        configuration.cache_dir= cache
        self.processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0",cache_dir = cache)
        self.wav2vec = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0",config=configuration,cache_dir=cache)
        
        self.wav2vec.eval()
        if wav2vec_layer < 0:
            self.wav2vec_weights = nn.Parameter(
                data=torch.ones(wav2vec_num_layers), requires_grad=True
            )

        #Little workaround to trick the model
        wav2vec_convlay_kernel = [10,3,3,3,3,2,2]
        wav2vec_convlay_stride = [5,2,2,2,2,2,2]
        self.wav2vec_convlay = [[kernel, stride] for kernel, stride in zip(wav2vec_convlay_kernel, wav2vec_convlay_stride)]


        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)

        self.save_hyperparameters("wav2vec", "wav2vec_layer", "lstm", "linear")

        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(wav2vec_dim, **multi_layer_lstm)

        else:
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0
            del one_layer_lstm["monolithic"]

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        wav2vec_dim
                        if i == 0
                        else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
                        **one_layer_lstm,
                    )
                    for i in range(num_layers)
                ]
            )

        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("SSeRiouSS does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        self.classifier = nn.Linear(in_features, self.dimension)
        self.activation = self.default_activation()

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """

        num_frames = num_samples
        for kernel,stride in self.wav2vec_convlay:
            num_frames = conv1d_num_frames(
                num_frames,
                kernel_size=kernel,
                stride=stride,
                padding=0,
                dilation=1,
            )

        return num_frames

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """

        receptive_field_size = num_frames
        for kernel,stride in reversed(self.wav2vec_convlay):
            receptive_field_size = conv1d_receptive_field_size(
                num_frames=receptive_field_size,
                kernel_size=kernel,
                stride=stride,
                dilation=1,
            )
        return receptive_field_size

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """
        receptive_field_center = frame
        for kernel,stride in reversed(self.wav2vec_convlay):
            receptive_field_center = conv1d_receptive_field_center(
                receptive_field_center,
                kernel_size=kernel,
                stride=stride,
                padding=0,
                dilation=1,
            )
        return receptive_field_center
    
    @cached_property
    def receptive_field(self) -> SlidingWindow:
        """(Internal) frames"""

        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        receptive_field_start = (
            self.receptive_field_center(frame=0) - (receptive_field_size - 1) / 2
        )
        return SlidingWindow(
            start=receptive_field_start / self.hparams.sample_rate,
            duration=receptive_field_size / self.hparams.sample_rate,
            step=receptive_field_step / self.hparams.sample_rate,
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        waveforms_to_process = waveforms.squeeze(1).cpu()

        processed_waveforms = []
        for waveform in waveforms :
            processed_waveforms.append(self.processor(waveform.cpu(), sampling_rate=16000, return_tensors="pt")['input_features'])
        
        mel_feats = torch.cat(processed_waveforms, dim=0).cuda()
        
        with torch.no_grad():
            outputs = self.wav2vec(mel_feats)
        outputs = outputs.hidden_states[1:25]

        outputs = torch.stack(outputs, dim=-1) @ F.softmax(
            self.wav2vec_weights, dim=0
        )

        if self.hparams.lstm["monolithic"]:
            outputs, _ = self.lstm(outputs)
        else:
            for i, lstm in enumerate(self.lstm):
                outputs, _ = lstm(outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    outputs = self.dropout(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))

