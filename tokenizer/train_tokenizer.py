import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input='data/train.txt',
    model_prefix='tokenizer/mytok',
    model_type='unigram',
    vocab_size=50,   # <= 64 for small corpus
    character_coverage=1.0
)

print("Tokenizer trained: tokenizer/mytok.model")
