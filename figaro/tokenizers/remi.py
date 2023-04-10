import os
import json
import logging
from tempfile import TemporaryDirectory
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import WordPiece

import pretty_midi
from transformers import PreTrainedTokenizerFast

from figaro.preprocessing.remi import RemiConfig
from figaro.chord_recognition.utils import CHORD_ROOTS, CHORD_QUALITIES


logger = logging.getLogger(__name__)


def _generate_vocab_from_config(
    config: RemiConfig,
    pad_token: str = "<pad>",
    bos_token: str = "<bos>",
    eos_token: str = "<eos>",
    unk_token: str = "<unk>",
):
    vocab = [pad_token, bos_token, eos_token, unk_token]
    vocab += [f"<Bar:{i}>" for i in range(config.max_bars)]
    vocab += [f"<Position:{i}>" for i in range(config.positions_per_beat * config.max_beats_per_bar)]
    vocab += [f"<Pitch:{i}>" for i in range(128)]
    vocab += [f"<Velocity:{x}>" for x in config.velocity_bins]
    vocab += [f"<Duration:{x}>" for x in config.duration_bins]
    vocab += [f"<Tempo:{x}>" for x in config.tempo_bins]
    vocab += [f"<TimeSig:{x}/{y}>" for x, y in config.valid_time_signatures]

    vocab.append("<Instrument:Drum>")
    for i in range(128):
        instrument = pretty_midi.program_to_instrument_name(i)
        instrument = instrument.replace(" ", "_")
        vocab.append(f"<Instrument:{instrument}>")

    if config.extract_chords:
        vocab.append("<Chord:N>")
        for root in CHORD_ROOTS:
            for quality in CHORD_QUALITIES:
                vocab.append(f"<Chord:{root}:{quality}>")
    
    return vocab


class RemiTokenizer(PreTrainedTokenizerFast):
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        **kwargs,
    ):  
        kwargs.pop("model_input_names", None)
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            model_input_names=["input_ids", "attention_mask"],
            **kwargs
        )

        self._token_ids_by_type = {
            "bar": [],
            "position": [],
            "pitch": [],
            "velocity": [],
            "duration": [],
            "tempo": [],
            "time_sig": [],
            "instrument": [],
            "chord": [],
        }

        for token, i in self.vocab.items():
            prefix = token.split(":")[0][1:]  # e.g. "<Position:0>" -> "Position"
            key = {
                "Bar": "bar",
                "Position": "position",
                "Pitch": "pitch",
                "Velocity": "velocity",
                "Duration": "duration",
                "Tempo": "tempo",
                "TimeSig": "time_sig",
                "Instrument": "instrument",
                "Chord": "chord",
            }.get(prefix, None)
            if key is not None:
                self._token_ids_by_type[key].append(i)
        
        for key, token_ids in self._token_ids_by_type.items():
            self._token_ids_by_type[key] = sorted(token_ids)

    @property
    def bar_token_ids(self):
        return self._token_ids_by_type["bar"]
    
    @property
    def position_token_ids(self):
        return self._token_ids_by_type["position"]

    @staticmethod
    def from_config(
        config: RemiConfig,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
    ):
        vocab_list = _generate_vocab_from_config(config, pad_token, bos_token, eos_token, unk_token)
        vocab = {token: i for i, token in enumerate(vocab_list)}

        new_tokenizer = Tokenizer(WordLevel(vocab, unk_token=unk_token))
        new_tokenizer.add_special_tokens([pad_token, bos_token, eos_token, unk_token])
        new_tokenizer.pre_tokenizer = WhitespaceSplit()
        new_tokenizer.post_processor = TemplateProcessing(
            single=f"{bos_token} $0 {eos_token}",
            special_tokens=[(bos_token, 1), (eos_token, 2)],
        )
        new_tokenizer.decoder = WordPiece(cleanup=False)

        with TemporaryDirectory() as tmp_dir:
            tokenizer_file = os.path.join(tmp_dir, "tokenizer.json")
            new_tokenizer.save(tokenizer_file)
            tokenizer_config_file = os.path.join(tmp_dir, "tokenizer_config.json")
            with open(tokenizer_config_file, "w") as f:
                json.dump({
                    "tokenizer_class": "RemiTokenizer",
                    "bos_token": bos_token,
                    "eos_token": eos_token,
                    "unk_token": unk_token,
                }, f)

            return RemiTokenizer.from_pretrained(tmp_dir)
