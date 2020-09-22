"""
Import necessary packages:
`re` for regular expressions
`pandas` for providing a data structure to store the CSV data in within Python
`StringIO` to provide an object for pandas functions to parse
"""

import re
import pandas as pd
from io import StringIO
from pathlib import Path


def get_annotation_data(annotation_file) -> pd.DataFrame:
    """
    This function reads files containing annotations and processes the data, storing it in a pandas.DataFrame object.
    The type of annotation (e.g. word-level or accent-level) is deduced from the file extension.
    Unnecessary information from the files is not returned, only timestamps and labels are returned.
    """

    annotation_file = Path(annotation_file).resolve()

    if annotation_file.is_file():
        pass
    else:
        raise FileNotFoundError("Input file could not be found.")

    annotation_type = annotation_file.suffix[
        1:
    ]  # Get annotation type from file extension, but without the dot

    if annotation_type in ["accents", "words", "tones"]:
        pass
    else:
        raise ValueError("Input file does not contain supported annotation type")

    raw_content = read_file(annotation_file)

    content = StringIO(
        clean_text(raw_content, annotation_type)
    )  # Store string in a StringIO object to enable pandas to parse it like a CSV file.

    if annotation_type == "words":
        data = pd.read_csv(
            content,
            sep=" ",
            engine="python",
            quoting=3,
            names=["end", "xwaves", "label"],
        ).drop(
            ["xwaves"], axis=1
        )  # Drop unnecessary xwaves column straight away

        data["start_est"] = None

        num_rows = len(data.index)

        # Add data for estimated start timestamps of each word, which are 10ms after the end timestamp of the previous label (to avoid overlap for now). Set the starting time of the first label to zero.
        for i in data.index:
            if i <= num_rows and i > 0:
                data.at[i, "start_est"] = data.loc[i - 1, "end"] + 0.001
            elif i == 0:
                data.at[i, "start_est"] = 0

    else:
        data = pd.read_csv(
            content,
            sep=" ",
            engine="python",
            quoting=3,
            names=["time", "xwaves", "label"],
        ).drop(
            ["xwaves"], axis=1  # Drop unnecessary xwaves column straight away)
        )

    return data


# Ancillary functions


def clean_text(raw_text, annotation_type) -> str:
    # Reduce all instances two or more consecutive whitespaces to one whitespace, remove leading and trailing whitespaces
    text = re.sub("[ ]{2,}", " ", raw_text).lstrip()
    text = re.sub("[ ]*\\n[ ]*", "\n", text)

    if annotation_type == "words":
        # Replace escaped Umlauts with proper ones
        replacements = {
            '"u': "ü",
            '"U': "Ü",
            '"a': "ä",
            '"A': "Ä",
            '"o': "ö",
            '"O': "Ö",
            '"s': "ß",
            '"S': "ß",
        }

        for string, replacement in replacements.items():
            text = text.replace(string, replacement)

    elif annotation_type in ["accents", "tones"]:
        pass

    else:
        raise ValueError

    return text


def read_file(input_file) -> str:

    # Use ISO 8859-1 (Western European) encoding to open TSV file containing the annotations
    with open(input_file, "r", encoding="iso-8859-1") as f:
        # Don't save any lines until a lone hash sign is found in the line
        # This signifies the end of the metadata and start of the data block
        while f.readline() != "#\n":
            pass
        rawContent = f.read()

    return rawContent
