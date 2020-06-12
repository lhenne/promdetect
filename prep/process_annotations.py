"""
Import necessary packages:
`re` for regular expressions
`pandas` for providing a data structure to store the CSV data in within Python
`StringIO` to provide an object for pandas functions to parse
"""

import re
import pandas as pd
from io import StringIO


def get_annotation_data(annotationFile, annotationType):

    # Read data from the input file starting from specific marker
    rawContent = read_file(annotationFile)

    # Clean data and convert to a StringIO object that can be parsed by the `pandas.read_csv` function
    content = clean_text(rawContent, annotationType)

    # Read the data into a pandas DataFrame
    # Drop empty column 'idx' directly (caused by a leading whitespace in each line)
    if annotationType == "words":
        annotationData = pd.read_csv(content,
                                     sep=" ",
                                     engine="python",
                                     quoting=3,
                                     names=["idx",
                                            "end",
                                            "xwaves",
                                            "label"]).drop(["idx"], axis=1)

        # Create an empty column to which the start timestamps for each word will be assigned
        annotationData["start"] = None

        # Iterate over the rows of annotationData, and use the 'end' timestamp from the preceding row
        # to derive the 'start' timestamp for the current one.
        # The 'start' timestamp is estimated by adding 10 milliseconds to the 'end' timestamp of the previous word
        for i in annotationData.index:
            if i <= len(annotationData.index) and i > 0:
                annotationData.at[i, "start"] = annotationData.loc[i - 1,
                                                                   "end"] + 0.001
            elif i == 0:
                annotationData.at[i, "start"] = 0

    elif annotationType in ["tones", "accents"]:
        # Read the data into a pandas DataFrame
        # Drop empty column 'idx' directly (caused by a leading whitespace in each line)
        annotationData = pd.read_csv(content, sep=" ", engine="python", quoting=3, names=[
            "idx", "time", "xwaves", "label"]).drop(["idx"], axis=1)

    return annotationData


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
