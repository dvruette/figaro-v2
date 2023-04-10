import os
import glob
import logging
from typing import Union, List, Generator

from torch.utils.data import Dataset, IterableDataset

from figaro.preprocessing.remi import RemiProcessor


logger = logging.getLogger(__name__)

class RemiDataset(Dataset):
    def __init__(
        self,
        files_or_folder: Union[str, List[str]],
        processor: RemiProcessor,
        file_pattern: str = "**/*.mid",
    ):
        if isinstance(files_or_folder, str):
            self.files = glob.glob(os.path.join(files_or_folder, file_pattern), recursive=True)
        else:
            self.files = files_or_folder
        self.processor = processor

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx) -> str:
        file = self.files[idx]
        try:
            return self.processor(file)
        except Exception as e:
            logger.error(f"Error ocurred while processing '{file}', returning empty sample: {e}")
            return ""


class IterableRemiDataset(IterableDataset):
    def __init__(
        self,
        files_or_folder: Union[str, List[str]],
        processor: RemiProcessor,
        file_pattern: str = "**/*.mid",
        raise_on_parsing_error: bool = False,
    ):
        if isinstance(files_or_folder, str):
            self.files = glob.glob(os.path.join(files_or_folder, file_pattern), recursive=True)
        else:
            self.files = files_or_folder
        self.processor = processor
        self.raise_on_parsing_error = raise_on_parsing_error

    def __iter__(self) -> Generator[str, None, None]:
        for file in self.files:
            try:
                yield self.processor(file)
            except Exception as e:
                if self.raise_on_parsing_error:
                    raise ValueError(f"Error ocurred while processing '{file}'.") from e
