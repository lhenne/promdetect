"""
Import necessary packages:
`re` for regular expressions
`pandas` for providing a data structure to store the CSV data in within Python
`StringIO` to provide an object for pandas functions to parse
"""

import re
import pandas as pd
from io import StringIO


def get_accents_data(accentsFile):

    # Read data from the input file starting from specific marker
    rawContent = read_file(accentsFile)

    # Clean data and convert to a StringIO object that can be parsed by the `pandas.read_csv` function
    content = clean_text(rawContent, "accents")

    # Read the data into a pandas DataFrame
    # Drop empty column 'idx' directly (caused by a leading whitespace in each line)
    accentsData = pd.read_csv(content, sep=" ", engine="python", quoting=3, names=[
                              "idx", "time", "xwaves", "label"]).drop(["idx"], axis=1)

    return accentsData


def get_tones_data(tonesFile):

    # Read data from the input file starting from specific marker
    rawContent = read_file(tonesFile)

    # Clean data and convert to a StringIO object that can be parsed by the `pandas.read_csv` function
    content = clean_text(rawContent, "tones")

    # Read the data into a pandas DataFrame
    # Drop empty column 'idx' directly (caused by a leading whitespace in each line)
    tonesData = pd.read_csv(content, sep=" ", engine="python", quoting=3, names=[
                            "idx", "time", "xwaves", "label"]).drop(["idx"], axis=1)

    return tonesData


def get_transcript_data(transcriptFile):

    # Read data from the input file starting from specific marker
    rawContent = read_file(transcriptFile)

    # Clean data and replace escaped Umlaute with proper ones by running `clean_text` function
    # Convert data to a StringIO object that can be parsed by the `pandas.read_csv` function
    content = clean_text(rawContent, "words")

    # Read the data into a pandas DataFrame
    # Drop empty column 'idx' directly (caused by a leading whitespace in each line)
    transcriptData = pd.read_csv(content,
                                 sep=" ",
                                 engine="python",
                                 quoting=3,
                                 names=["idx",
                                        "end",
                                        "xwaves",
                                        "label"]).drop(["idx"], axis=1)

    # Create an empty column to which the start timestamps for each word will be assigned
    transcriptData["start"] = None

    # Iterate over the rows of transcriptData, and use the 'end' timestamp from the preceding row
    # to derive the 'start' timestamp for the current one.
    # The 'start' timestamp is estimated by adding 10 milliseconds to the 'end' timestamp of the previous word
    for i in transcriptData.index:
        if i <= len(transcriptData.index) and i > 0:
            transcriptData.at[i, "start"] = transcriptData.loc[i - 1,
                                                               "end"] + 0.001
        elif i == 0:
            transcriptData.at[i, "start"] = 0

    return transcriptData


# Ancillary functions block


def clean_text(rawText, annotationType):
    # Reduce all instances two or more consecutive whitespaces to one whitespace
    text = re.sub(r" {2,}", " ", rawText)

    if annotationType == "words":
        #  Replace escaped Umlaut with proper ones
        replacements = {
            "\"u": "ü",
            "\"U": "Ü",
            "\"a": "ä",
            "\"A": "Ä",
            "\"o": "ö",
            "\"O": "Ö",
            "\"s": "ß",
            "\"S": "ß"
        }

        for string, replacement in replacements.items():
            text = text.replace(string, replacement)

    elif annotationType in ["accents", "tones"]:
        pass

    else:
        raise ValueError

    return StringIO(text)


def read_file(inputFile):

    # Use ISO 8859-1 (Western European) encoding to open TSV file containing the annotations
    with open(inputFile, "r", encoding="iso-8859-1") as f:
        # Don't save any lines until a lone hash sign is found in the line
        # This signifies the end of the metadata and start of the data block
        while f.readline() != "#\n":
            pass
        rawContent = f.read()

    return rawContent
