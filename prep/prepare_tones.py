"""
Import necessary packages:
`re` for regular expressions
`pandas` for providing a data structure to store the CSV data in within Python
`StringIO` to provide an object for pandas functions to parse
"""

import re
import pandas as pd
from io import StringIO


def get_tones_data(tonesFile):

    # Use ISO 8859-1 (Western European) encoding to open TSV file containing tone annotations
    with open(tonesFile, "r", encoding="iso-8859-1") as f:
        # Don't save any lines until a lone hash sign is found in the line
        # This signifies the end of the metadata and start of the data block
        while f.readline() != "#\n":
            pass
        rawContent = f.read()

    # Clean data and convert to a StringIO object that can be parsed by the `pandas.read_csv` function
    content = StringIO(re.sub(r" {2,}", " ", rawContent))

    # Read the data into a pandas DataFrame
    # Drop empty column 'idx' directly (caused by a leading whitespace in each line)
    tonesData = pd.read_csv(content, sep=" ", engine="python", quoting=3, names=[
                            "idx", "time", "xwaves", "label"]).drop(["idx"], axis=1)

    return tonesData
