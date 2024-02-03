import pytest

from whisper.tokenizer import get_tokenizer


@pytest.mark.parametrize("multilingual", [True, False])
def test_tokenizer(multilingual):
    """Tokenizes text for a specific language model.
    Args:
        multilingual: Whether the tokenizer supports multiple languages
    Returns:
        tokenizer: Tokenizer object initialized for the given language model
    - Initializes tokenizer for monolingual model if multilingual is False
    - Checks if start of text token is in start of text sequence
    - Checks if number of language codes equals number of language tokens
    - Checks if all language tokens have timestamp before tokenizer timestamp begin"""
    
    tokenizer = get_tokenizer(multilingual=False)
    assert tokenizer.sot in tokenizer.sot_sequence
    assert len(tokenizer.all_language_codes) == len(tokenizer.all_language_tokens)
    assert all(c < tokenizer.timestamp_begin for c in tokenizer.all_language_tokens)


def test_multilingual_tokenizer():
    """
    Tests multilingual tokenizer against GPT-2 tokenizer
    Args:
        None: None
    Returns:
        None: None
    - Get GPT-2 tokenizer and multilingual tokenizer
    - Encode Korean text with both tokenizers
    - Decode tokens and check if original text is recovered
    - Check if multilingual tokenizer results in fewer tokens than GPT-2 tokenizer
    """
    
    gpt2_tokenizer = get_tokenizer(multilingual=False)
    multilingual_tokenizer = get_tokenizer(multilingual=True)

    text = "다람쥐 헌 쳇바퀴에 타고파"
    gpt2_tokens = gpt2_tokenizer.encode(text)
    multilingual_tokens = multilingual_tokenizer.encode(text)

    assert gpt2_tokenizer.decode(gpt2_tokens) == text
    assert multilingual_tokenizer.decode(multilingual_tokens) == text
    assert len(gpt2_tokens) > len(multilingual_tokens)


def test_split_on_unicode():
    """
    Splits tokens on unicode characters
    Args:
        tokens: List of token ids
    Returns:
        words: List of words
        word_tokens: List of token ids for each word
    Splits tokens on unicode characters and returns words and corresponding token ids:
    - Splits input tokens on unicode characters
    - Returns words list with split tokens joined as strings
    - Returns word_tokens list with token ids for each word
    """
    
    multilingual_tokenizer = get_tokenizer(multilingual=True)

    tokens = [8404, 871, 287, 6, 246, 526, 3210, 20378]
    words, word_tokens = multilingual_tokenizer.split_tokens_on_unicode(tokens)

    assert words == [" elle", " est", " l", "'", "\ufffd", "é", "rit", "oire"]
    assert word_tokens == [[8404], [871], [287], [6], [246], [526], [3210], [20378]]
