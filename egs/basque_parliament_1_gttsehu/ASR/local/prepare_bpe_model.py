from lhotse import CutSet
from lhotse.utils import Pathlike
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

def prepare_bpe_model(manifest_path: Pathlike, output_path: Pathlike, vocab_size: int = 1000):
    cuts = CutSet.from_file(manifest_path)
    texts = [cut.supervisions[0].text for cut in cuts]
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
    
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(output_path))

if __name__ == '__main__':
    prepare_bpe_model('data/fbank/train/cuts.jsonl.gz', 'data/lang/bpe.model')
