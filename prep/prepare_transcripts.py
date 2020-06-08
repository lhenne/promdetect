"""
Import necessary packages:
`re` for regular expressions
`pandas` for providing a data structure to store the CSV data in within Python
`StringIO` to provide an object for pandas functions to parse
"""

import re
import pandas as pd
from io import StringIO


def get_transcript_data(transcriptFile):

    with open(transcriptFile, "r", encoding="iso-8859-1") as f:
        while f.readline() != "#\n":
            pass
        rawContent = f.read()

    content = StringIO(clean_text(rawContent))

    transcriptData = pd.read_csv(content,
                                 sep=" ",
                                 engine="python",
                                 quoting=3,
                                 names=["idx",
                                        "end",
                                        "xwaves",
                                        "label"]).drop(["idx"], axis=1)

    transcriptData["start"] = None

    for i in transcriptData.index:
        if i <= len(transcriptData.index) and i > 0:
            transcriptData.at[i, "start"] = transcriptData.loc[i-1,
                                                               "end"] + 0.001

    return transcriptData


# Ancillary functions block


def clean_text(rawText):
    text = re.sub(r" {2,}", " ", rawText)

    #  Fix Umlaute and scharfes S
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

    return text
