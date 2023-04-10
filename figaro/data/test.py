from torch.utils.data import DataLoader

from figaro.data.dataset import IterableRemiDataset
from figaro.preprocessing.remi import RemiProcessor
from figaro.data.collators import RemiCollatorForCausalLM
from figaro.tokenizers.remi import RemiTokenizer

def main():
    input_path = "data_cache/isophonics_annotated"

    tokenizer = RemiTokenizer.from_pretrained("dvruette/remi-plus")
    processor = RemiProcessor.from_pretrained("dvruette/remi-plus")

    dataset = IterableRemiDataset(input_path, processor)
    collator = RemiCollatorForCausalLM(
        tokenizer,
        max_length=510,
        pad_to_multiple_of=8,
        drop_empty=True,
    )

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)

    batch = next(iter(dataloader))
    print(batch)

if __name__ == "__main__":
    main()
