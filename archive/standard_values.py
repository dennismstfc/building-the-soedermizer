from pathlib import Path

DATA_PATH = Path('data')

RAW_DATA_PATH = DATA_PATH / 'gendered_words_raw.json'
PROCESSED_DATA_PATH = DATA_PATH / 'gendered_words_processed.json'
FINAL_DICTIONARY_PATH = DATA_PATH / 'gender_dictionary.json'
WIKIPEDIA_SENTENCES_PATH = DATA_PATH / 'sentences.txt'

TRAIN_DATA_PATH = DATA_PATH / 'train.csv'
EVAL_DATA_PATH = DATA_PATH / 'eval.csv'
TEST_DATA_PATH = DATA_PATH / 'test.csv'