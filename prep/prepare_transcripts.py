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
        while f.readline() != "#\n": # A lone hash sign in a line signifies the start of the data block
            pass
        rawContent = f.read()
    
    content = StringIO(clean_text(rawContent)) # Clean data and convert to a StringIO object that can be parsed by the `pandas.read_csv` function

    transcriptData = pd.read_csv(content, sep=" ", engine="python" , quoting=3, names=["start","xwaves","label"])

    return transcriptData


#### Ancillary functions block


def clean_text(rawText):
    text = re.sub(r" {2,}", " ", rawText) # Eliminate the indent before each line, and regularize whitespaces
    
    #### Fix Umlaute and scharfes S
    replacements = { # Create catalogue of strings that have to be replaced, with their replacements
        "\"u": "ü",
        "\"U": "Ü",
        "\"a": "ä",
        "\"A": "Ä",
        "\"o": "ö",
        "\"O": "Ö",
        "\"s": "ß",
        "\"S": "ß"
        }
    
    
    for string, replacement in replacements.items(): # do the replacement for each of the dictionary's entries
        text = text.replace(string, replacement)

    return text
