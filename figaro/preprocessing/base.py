import copy
import os
import json
from typing import Union, Tuple, Dict, Any

import numpy as np
from transformers.utils import (
    PushToHubMixin,
    cached_file,
    download_url,
    is_offline_mode,
    is_remote_url,
    logging
)

MIDI_PROCESSOR_NAME = "preprocessor_config.json"

logger = logging.get_logger(__name__)


class MidiProcessingMixin(PushToHubMixin):

    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""
        # Pop "processor_class" as it should be saved as private attribute
        self._processor_class = kwargs.pop("processor_class", None)
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def _set_processor_class(self, processor_class: str):
        """Sets processor class as an attribute."""
        self._processor_class = processor_class

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        midi_processor_dict, kwargs = cls.get_midi_processor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(midi_processor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_midi_processor_file = os.path.join(save_directory, MIDI_PROCESSOR_NAME)

        self.to_json_file(output_midi_processor_file)
        logger.info(f"Midi processor saved in {output_midi_processor_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("use_auth_token"),
            )

        return [output_midi_processor_file]

    @classmethod
    def get_midi_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "midi processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            midi_processor_file = os.path.join(pretrained_model_name_or_path, MIDI_PROCESSOR_NAME)
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_midi_processor_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            midi_processor_file = pretrained_model_name_or_path
            resolved_midi_processor_file = download_url(pretrained_model_name_or_path)
        else:
            midi_processor_file = MIDI_PROCESSOR_NAME
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_midi_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    midi_processor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load midi processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {MIDI_PROCESSOR_NAME} file"
                )

        try:
            # Load midi_processor dict
            with open(resolved_midi_processor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            midi_processor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_midi_processor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_midi_processor_file}")
        else:
            logger.info(
                f"loading configuration file {midi_processor_file} from cache at {resolved_midi_processor_file}"
            )

        return midi_processor_dict, kwargs

    @classmethod
    def from_dict(cls, midi_processor_dict: Dict[str, Any], **kwargs):
        midi_processor_dict = midi_processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # The `size` parameter is a dict and was previously an int or tuple in feature extractors.
        # We set `size` here directly to the `midi_processor_dict` so that it is converted to the appropriate
        # dict within the midi processor and isn't overwritten if `size` is passed in as a kwarg.
        if "size" in kwargs and "size" in midi_processor_dict:
            midi_processor_dict["size"] = kwargs.pop("size")
        if "crop_size" in kwargs and "crop_size" in midi_processor_dict:
            midi_processor_dict["crop_size"] = kwargs.pop("crop_size")

        midi_processor = cls(**midi_processor_dict)

        # Update midi_processor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(midi_processor, key):
                setattr(midi_processor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Midi processor {midi_processor}")
        if return_unused_kwargs:
            return midi_processor, kwargs
        else:
            return midi_processor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.
        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this MIDI processor instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["midi_processor_type"] = self.__class__.__name__

        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        Instantiates a MIDI processor of type [`~MidiProcessingMixin`] from the path to a JSON
        file of parameters.
        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.
        Returns:
            A MIDI processor of type [`~MidiProcessingMixin`]: The midi_processor object
            instantiated from that JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        midi_processor_dict = json.loads(text)
        return cls(**midi_processor_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.
        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.
        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this midi_processor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

class BaseMidiProcessor(MidiProcessingMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, midi_files, **kwargs) -> Any:
        """Preprocess a MIDI file or a batch of MIDI files."""
        return self.preprocess(midi_files, **kwargs)

    def preprocess(self, midi_files, **kwargs) -> Any:
        raise NotImplementedError("Each MIDI processor must implement its own preprocess method")
