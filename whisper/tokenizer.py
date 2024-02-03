import base64
import os
import string
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Dict, List, Optional, Tuple

import tiktoken

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}


@dataclass
class Tokenizer:
    """A thin wrapper around `tiktoken` providing quick access to special tokens"""

    encoding: tiktoken.Encoding
    num_languages: int
    language: Optional[str] = None
    task: Optional[str] = None
    sot_sequence: Tuple[int] = ()
    special_tokens: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Initializes special tokens and start-of-transcript sequence
        Args:
            self: The instance of the class being initialized
        Returns:
            None: This method does not return anything
        - Loops through special tokens and encodes them
        - Gets integer IDs for special start-of-transcript, translate, and transcribe tokens
        - Gets languages and builds start-of-transcript sequence with language and task tokens if set
        - Assigns start-of-transcript sequence to self.sot_sequence"""
        
        for special in self.encoding.special_tokens_set:
            special_token = self.encoding.encode_single_token(special)
            self.special_tokens[special] = special_token

        sot: int = self.special_tokens["<|startoftranscript|>"]
        translate: int = self.special_tokens["<|translate|>"]
        transcribe: int = self.special_tokens["<|transcribe|>"]

        langs = tuple(LANGUAGES.keys())[: self.num_languages]
        sot_sequence = [sot]
        if self.language is not None:
            sot_sequence.append(sot + 1 + langs.index(self.language))
        if self.task is not None:
            task_token: int = transcribe if self.task == "transcribe" else translate
            sot_sequence.append(task_token)

        self.sot_sequence = tuple(sot_sequence)

    def encode(self, text, **kwargs):
        """
        Encodes text using the encoding object
        Args:
            text: The text to encode
            kwargs: Additional keyword arguments to pass to the encoding object
        Returns:
            bytes: The encoded bytes
        - Encodes the provided text using the encoding object's encode method
        - Passes any additional keyword arguments to the encoding object
        - Returns the encoded bytes from the encoding object"""
        
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token ids into a string
        Args:
            token_ids: List of token ids to decode
        Returns:
            str: Decoded string from token ids
        Filter out token ids past timestamp begin:
        - Iterate through token_ids list
        - Only keep tokens whose id is less than self.timestamp_begin
        Decode filtered token ids:
        - Pass filtered token ids to encoding object's decode method
        - Return decoded string"""
        
        token_ids = [t for t in token_ids if t < self.timestamp_begin]
        return self.encoding.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, token_ids: List[int], **kwargs) -> str:
        """
        Timestamp tokens are above other special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        return self.encoding.decode(token_ids, **kwargs)

    @cached_property
    def eot(self) -> int:
        """
        Returns the end-of-text token
        Args:
            self: The object whose encoding is accessed
        Returns:
            int: The integer value of the end-of-text token
        - Access the encoding attribute of the object
        - Return the end-of-text token value from the encoding"""
        
        return self.encoding.eot_token

    @cached_property
    def transcribe(self) -> int:
        """
        Transcribe a special token to an integer
        Args:
            self: The object being operated on
        Returns:
            int: The integer value of the special token
        - Lookup the special token "<|transcribe|>" in the object's special_tokens dictionary
        - Return the value associated with that key
        """
        
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        """
        Translates the special token to its integer value
        Args:
            self: The object containing the special tokens
        Returns:
            int: The integer value of the special token
        - Looks up the special token "<|translate|>" in the object's special_tokens dictionary
        - Returns the integer value associated with that token"""
        
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:
        """
        Returns the index of the start of transcript token
        Args:
            self: The object whose special tokens are being accessed
        Returns:
            int: The index of the start of transcript token
        - Access the special_tokens attribute of the object
        - Return the value corresponding to the "<|startoftranscript|>" key
        """
        
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self) -> int:
        """
        Returns the start of list marker token
        Args:
            self: The object whose special tokens are being accessed
        Returns:
            int: The integer representation of the start of list marker token
        - Access the special_tokens dictionary on the object
        - Return the value corresponding to the "<|startoflm|>" key
        """
        
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:
        """
        Returns the index of the previous token
        Args:
            self: The object whose previous token index is returned
        Returns:
            int: The index of the previous token
        - The function returns the index stored in the special_tokens dictionary under the key "<|startofprev|>"
        - This index refers to the token immediately preceding the current token in the sequence
        """
        
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self) -> int:
        """
        Returns the special token for no speech
        Args:
            self: The class instance
        Returns:
            int: The integer id of the no speech token
        - Searches the special_tokens dictionary for the "<|nospeech|>" key
        - Returns the value associated with that key, which is the integer id of the no speech token"""
        
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self) -> int:
        """
        Returns the special token for no timestamps
        Args:
            self: The object containing special tokens
        Returns:
            int: The integer value of the no timestamps special token
        - Searches the self.special_tokens dictionary for the "<|notimestamps|>" key
        - Returns the value associated with that key, which is the integer code for no timestamps
        """
        
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        """
        Returns the timestamp token for the beginning of the video
        Args:
            self: The class instance
        Returns:
            int: The timestamp token as an integer
        - Retrieves the special token "<|0.00|>" from the class instance
        - "<|0.00|>" represents the beginning of the video
        """
        
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self) -> int:
        """Returns the token id corresponding to the value of the `language` field"""
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")

        return self.to_language_token(self.language)

    def to_language_token(self, language):
        """Converts language to special token
        Args:
            language: The language to convert
        Returns:
            token: The special token for the language
        - Checks if a special token exists for the given language
        - Returns the token if found
        - Raises KeyError if language is not found"""
        
        if token := self.special_tokens.get(f"<|{language}|>", None):
            return token

        raise KeyError(f"Language {language} not found in tokenizer.")

    @cached_property
    def all_language_tokens(self) -> Tuple[int]:
        result = []
        for token, token_id in self.special_tokens.items():
        """
        Returns language tokens ids from the special tokens.
        Args:
            self: The object instance.
        Returns:
            Tuple[int]: Tuple of language token ids.
        - Loops through special tokens and checks if token name is in LANGUAGES list
        - Appends the token id to result list if matches
        - Returns first self.num_languages items from result as a tuple
        """
        
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)[: self.num_languages]

    @cached_property
    def all_language_codes(self) -> Tuple[str]:
        """
        Returns a tuple of all language codes
        Args:
            self: The object whose method is called
        Returns:
            Tuple[str]: A tuple of language code strings
        - Loops through all language tokens in the object
        - Decodes each token using the decode method
        - Strips XML tags from decoded string
        - Adds decoded string to a tuple
        - Returns the final tuple of language codes"""
        
        return tuple(self.decode([_l]).strip("<|>") for _l in self.all_language_tokens)

    @cached_property
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        """Generate sequence of start of transmission including no timestamps flag
        Args:
            self: The class instance
        Returns:
            Tuple[int]: Tuple containing the SOT sequence and no timestamps flag
        - Get the SOT sequence from the class instance
        - Append the no timestamps flag to the end of the sequence
        - Return a tuple containing the combined sequence"""
        
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @cached_property
    def non_speech_tokens(self) -> Tuple[int]:
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += (
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        )

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [
                self.encoding.encode(symbol),
                self.encoding.encode(" " + symbol),
            ]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))

    def split_to_word_tokens(self, tokens: List[int]):
        """
        Splits tokens into word tokens based on language.
        Args:
            tokens: List of int tokens to split
        Returns:
            List[List[int]]: List of word token lists
        Processing Logic:
        - Checks if language is Chinese, Japanese, Thai, Lao, Burmese or Cantonese
        - If so, splits tokens where unicode points are valid without spaces
        - Otherwise, splits tokens on spaces
        """
        
        if self.language in {"zh", "ja", "th", "lo", "my", "yue"}:
            # These languages don't typically use spaces, so it is difficult to split words
            # without morpheme analysis. Here, we instead split words at any
            # position where the tokens are decoded as valid unicode points
            return self.split_tokens_on_unicode(tokens)

        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(self, tokens: List[int]):
        """
        Splits tokens on unicode replacement characters.
        Args:
            tokens: List[int]: The list of tokens to split.
        Returns:
            words: List[str]: The list of decoded strings split on unicode characters.
            word_tokens: List[List[int]]: The list of token lists for each decoded string.
        Processing Logic:
        - Decodes tokens and stores full decoded string
        - Iterates through tokens, decoding current token subset
        - If unicode character not found, add to words and word_tokens
        - If found, reset current tokens and increment unicode offset
        - Returns words and word_tokens lists
        """
        
        decoded_full = self.decode_with_timestamps(tokens)
        replacement_char = "\ufffd"

        words = []
        word_tokens = []
        current_tokens = []
        unicode_offset = 0

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)

            if (
                replacement_char not in decoded
                or decoded_full[unicode_offset + decoded.index(replacement_char)]
                == replacement_char
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)

        return words, word_tokens

    def split_tokens_on_spaces(self, tokens: List[int]):
        """
        Splits tokens on spaces into words.
        Args:
            tokens (List[int]): List of token IDs
        Returns:
            words: List of words
            word_tokens: List of lists of token IDs for each word
        Processing Logic:
        - Splits tokens on unicode boundaries into subwords
        - Iterates over subwords and subword token lists
        - Checks if subword starts a new word based on punctuation, spaces etc
        - Appends to words and word_tokens lists or extends existing items
        """
        
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens


@lru_cache(maxsize=None)
def get_encoding(name: str = "gpt2", num_languages: int = 99):
    """
    Gets the encoding for a TikTok model.
    
    Args:
        name (str): Name of the model. Defaults to "gpt2".
        num_languages (int): Number of language tokens to include. Defaults to 99.
    
    Returns:
        Encoding: The encoding object.
    
    Processes the vocabulary file to get ranks and special tokens, and returns a tiktoken Encoding object with the encoded information.
    """
    
    vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{name}.tiktoken")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,
    *,
    num_languages: int = 99,
    language: Optional[str] = None,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
) -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        encoding_name = "multilingual"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    encoding = get_encoding(name=encoding_name, num_languages=num_languages)

    return Tokenizer(
        encoding=encoding, num_languages=num_languages, language=language, task=task
    )
