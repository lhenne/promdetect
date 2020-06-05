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
    
    rawContent = re.sub("[ ]{3,}", "", rawContent).lstrip() # Eliminate the indent before each line

    content = StringIO(rawContent) # Convert to a StringIO object that can be parsed by the `pandas.read_csv` function
    transcriptData = pd.read_csv(content, sep=" ", engine="python" , quoting=3, names=["start","xwaves","label"])

    return transcriptData
