import json
import os
import re
import sys
import zlib
from typing import Callable, Optional, TextIO

system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":

    def make_safe(string):
        # replaces any character not representable using the system default encoding with an '?',
        # avoiding UnicodeEncodeError (https://github.com/openai/whisper/discussions/729).
        return string.encode(system_encoding, errors="replace").decode(system_encoding)

else:

    def make_safe(string):
        # utf-8 can encode any Unicode code point, so no need to do the round-trip encoding
        return string


def exact_div(x, y):
    """
    Divides two numbers and returns the exact quotient
    Args:
        x: The dividend number
        y: The divisor number
    Returns:
        The exact quotient of x/y
    - Checks if x is exactly divisible by y using assert
    - Performs integer division of x/y
    - Returns the exact quotient"""
    
    assert x % y == 0
    return x // y


def str2bool(string):
    """Convert a string to a boolean
    Args:
        string: String to convert to boolean
    Returns:
        bool: Converted boolean value from string
    - Check if the input string is a key in the mapping dictionary
    - If present, return the mapped boolean value
    - If not present, raise a ValueError with the expected and received values"""
    
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string):
    """
    Converts a string to an integer or None
    Args:
        string: String to convert to int or None
    Returns:
        int or None: Integer if string can be converted, else None
    - Check if input string is equal to "None"
    - If equal, return None
    - Else try converting string to integer and return the integer
    - If conversion fails, return None"""
    
    return None if string == "None" else int(string)


def optional_float(string):
    """
    Converts a string to a float or None
    Args:
        string: String to convert to float or None
    Returns:
        float or None: Converted string as float or None
    - Check if input string is equal to "None"
    - If equal, return None
    - Else try to convert string to float
    - Return converted float"""
    
    return None if string == "None" else float(string)


def compression_ratio(text) -> float:
    """
    Calculate compression ratio of text
    Args:
        text: Input text string
    Returns:
        ratio: Compression ratio of text as float
    Calculate compression ratio by:
    - Encode text to bytes using UTF-8 encoding
    - Compress encoded bytes using zlib
    - Return ratio of original length to compressed length
    """
    
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    """
    Formats a timestamp in hours:minutes:seconds.milliseconds format.
    
    Args:
        seconds: float - The number of seconds to format
        always_include_hours: bool - Whether to always include hours even if 0
        decimal_marker: str - The marker between seconds and milliseconds
    Returns:
        str: The formatted timestamp as a string
    
    Processing Logic:
        - Convert seconds to milliseconds
        - Extract hours, minutes, seconds from milliseconds
        - Format components with leading zeros
        - Join components with ':' delimiter and decimal_marker
        - Optionally include hours component even if 0
    """
    
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        """
        Initialize the class with output directory
        Args:
            output_dir: The directory to write outputs
        Returns:
            None
        - Set the output directory attribute from the input parameter
        - No other processing is done during initialization"""
        
        self.output_dir = output_dir

    def __call__(
        self, result: dict, audio_path: str, options: Optional[dict] = None, **kwargs
    ):
        """
        Writes the result to an output file.
        Args:
            result: {The result dictionary to write}
            audio_path: {The path to the input audio file}
            options: {Optional configuration dictionary}
        Returns:
            None: {Does not return anything}
        - Gets the base filename of the audio file
        - Generates the output path by joining the output directory and filename
        - Opens the output file and writes the result
        - Closes the output file
        """
        
        audio_basename = os.path.basename(audio_path)
        audio_basename = os.path.splitext(audio_basename)[0]
        output_path = os.path.join(
            self.output_dir, audio_basename + "." + self.extension
        )

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f, options=options, **kwargs)

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        """Writes result to file
        Args:
            result: The result to write
            file: The file to write to
            options: Additional options for writing
            kwargs: Additional keyword arguments
        Returns:
            None: Writes the result to the given file
        - Opens the given file for writing
        - Serializes the result to a string
        - Writes the serialized result to the file
        - Closes the file"""
        
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        """Writes the text from speech recognition result segments to a file
        Args:
            result: dict: The speech recognition result
            file: TextIO: The file to write results to
            options: Optional[dict]: Additional options
            kwargs: Additional keyword arguments
        Returns:
            None: The function does not return anything
        - Loops through each segment in the result
        - Prints the stripped text from each segment to the given file
        - Flushes the output to ensure it is written immediately"""
        
        for segment in result["segments"]:
            print(segment["text"].strip(), file=file, flush=True)


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def iterate_result(
        self,
        result: dict,
        options: Optional[dict] = None,
        """Iterates over the result of a transcription to yield formatted subtitle lines
        Args:
            result: dict - The transcription result
            options: dict - Configuration options for formatting
            max_line_width: int - Maximum characters per line
            max_line_count: int - Maximum lines per subtitle
            highlight_words: bool - Whether to highlight the word being spoken
            max_words_per_line: int - Maximum words per line
        Returns:
            generator - Yields (start, end, text) tuples for each formatted subtitle line
        Processing Logic:
            - Splits result into segments and words
            - Groups words into lines while respecting timing and length limits
            - Yields formatted subtitles with start/end times and text
            - Optionally highlights the word being spoken in each line
        """
        
        *,
        max_line_width: Optional[int] = None,
        max_line_count: Optional[int] = None,
        highlight_words: bool = False,
        max_words_per_line: Optional[int] = None,
    ):
        options = options or {}
        max_line_width = max_line_width or options.get("max_line_width")
        max_line_count = max_line_count or options.get("max_line_count")
        highlight_words = highlight_words or options.get("highlight_words", False)
        max_words_per_line = max_words_per_line or options.get("max_words_per_line")
        preserve_segments = max_line_count is None or max_line_width is None
        max_line_width = max_line_width or 1000
        max_words_per_line = max_words_per_line or 1000

        def iterate_subtitles():
            line_len = 0
            line_count = 1
            # the next subtitle to yield (a list of word timings with whitespace)
            subtitle: list[dict] = []
            last = result["segments"][0]["words"][0]["start"]
            for segment in result["segments"]:
                chunk_index = 0
                words_count = max_words_per_line
                while chunk_index < len(segment["words"]):
                    remaining_words = len(segment["words"]) - chunk_index
                    if max_words_per_line > len(segment["words"]) - chunk_index:
                        words_count = remaining_words
                    for i, original_timing in enumerate(
                        segment["words"][chunk_index : chunk_index + words_count]
                    ):
                        timing = original_timing.copy()
                        long_pause = (
                            not preserve_segments and timing["start"] - last > 3.0
                        )
                        has_room = line_len + len(timing["word"]) <= max_line_width
                        seg_break = i == 0 and len(subtitle) > 0 and preserve_segments
                        if (
                            line_len > 0
                            and has_room
                            and not long_pause
                            and not seg_break
                        ):
                            # line continuation
                            line_len += len(timing["word"])
                        else:
                            # new line
                            timing["word"] = timing["word"].strip()
                            if (
                                len(subtitle) > 0
                                and max_line_count is not None
                                and (long_pause or line_count >= max_line_count)
                                or seg_break
                            ):
                                # subtitle break
                                yield subtitle
                                subtitle = []
                                line_count = 1
                            elif line_len > 0:
                                # line break
                                line_count += 1
                                timing["word"] = "\n" + timing["word"]
                            line_len = len(timing["word"].strip())
                        subtitle.append(timing)
                        last = timing["start"]
                    chunk_index += max_words_per_line
            if len(subtitle) > 0:
                yield subtitle

        if len(result["segments"]) > 0 and "words" in result["segments"][0]:
            for subtitle in iterate_subtitles():
                subtitle_start = self.format_timestamp(subtitle[0]["start"])
                subtitle_end = self.format_timestamp(subtitle[-1]["end"])
                subtitle_text = "".join([word["word"] for word in subtitle])
                if highlight_words:
                    last = subtitle_start
                    all_words = [timing["word"] for timing in subtitle]
                    for i, this_word in enumerate(subtitle):
                        start = self.format_timestamp(this_word["start"])
                        end = self.format_timestamp(this_word["end"])
                        if last != start:
                            yield last, start, subtitle_text

                        yield start, end, "".join(
                            [
                                re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", word)
                                if j == i
                                else word
                                for j, word in enumerate(all_words)
                            ]
                        )
                        last = end
                else:
                    yield subtitle_start, subtitle_end, subtitle_text
        else:
            for segment in result["segments"]:
                segment_start = self.format_timestamp(segment["start"])
                segment_end = self.format_timestamp(segment["end"])
                segment_text = segment["text"].strip().replace("-->", "->")
                yield segment_start, segment_end, segment_text

    def format_timestamp(self, seconds: float):
        """Formats a timestamp in seconds to a human-readable string.
        Args:
            seconds: Float representing timestamp in seconds.
        Returns:
            str: Formatted timestamp string.
        Processes the timestamp:
            - Converts seconds to datetime object
            - Formats datetime as string based on configuration
            - Returns formatted string
        """
        
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )


class WriteVTT(SubtitlesWriter):
    extension: str = "vtt"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        """Writes the result to a WebVTT file
        Args:
            self: The object
            result: The result dictionary
            file: The file object to write to
            options: Options for iterating over the result
            kwargs: Additional keyword arguments
        Returns:
            None: Does not return anything
        - Iterates over the result dictionary using iterate_result()
        - Prints the WebVTT header
        - Prints each line with start, end times and text
        - Flushes the output after each line to write to file immediately"""
        
        print("WEBVTT\n", file=file)
        for start, end, text in self.iterate_result(result, options, **kwargs):
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteSRT(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        """Writes result to file with line numbers, offsets and text.
        Args:
            result: Dict containing result to write
            file: File object to write to
            options: Dict of additional options
            kwargs: Additional keyword arguments
        Returns:
            None: No return value
        - Iterates through result using iterate_result()
        - Prints line number, offsets and text to the given file
        - Flushes output after each item to write immediately"""
        
        for i, (start, end, text) in enumerate(
            self.iterate_result(result, options, **kwargs), start=1
        ):
            print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    extension: str = "tsv"

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        """
        Writes result segments to a file with start, end and text separated by tabs
        Args:
            result: {The result dictionary containing segments}
            file: {The file object to write output to}
            options: {Optional dictionary of additional options}
        Returns:
            None: {Does not return anything, just writes to the file}
        - Loops through each segment in the result
        - Prints the start time, end time and text to the file separated by tabs
        - Rounds start and end times to milliseconds and strips/replaces tabs in text
        - Flushes output after each line to write immediately to file
        """
        
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment["start"]), file=file, end="\t")
            print(round(1000 * segment["end"]), file=file, end="\t")
            print(segment["text"].strip().replace("\t", " "), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        """Writes result to file as JSON
        Args:
            result: dict - Result data to write
            file: TextIO - File object to write to
            options: Optional[dict] - Additional options
            kwargs: Additional keyword arguments
        Returns:
            None: Writes result as JSON to file
        Processes result as JSON:
            - Dumps result dictionary to file object as JSON string
            - Handles any additional options or keyword arguments"""
        
        json.dump(result, file)


def get_writer(
    output_format: str, output_dir: str
    """
    Get writer function for output format
    Args:
        output_format: {Output format string}
        output_dir: {Output directory string}
    Returns:
        Callable: {Function to write output}
    Processing Logic:
        - Define mapping of writers for each format
        - If format is 'all', return function that calls all writers
        - Otherwise return single writer function for given format
    """
    
) -> Callable[[dict, TextIO, dict], None]:
    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(
            result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
        ):
            for writer in all_writers:
                writer(result, file, options, **kwargs)

        return write_all

    return writers[output_format](output_dir)
