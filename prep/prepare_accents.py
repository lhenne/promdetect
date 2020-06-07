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

    with open(accentsFile, "r", encoding="iso-8859-1") as f:
        while f.readline() != "#\n": # A lone hash sign in a line signifies the start of the data block
            pass
        rawContent = f.read()

     
    # Clean data and convert to a StringIO object that can be parsed by the `pandas.read_csv` function
    content = StringIO(re.sub(r" {2,}", " ", rawContent))
    
    accentsData = pd.read_csv(content, sep=" ", engine="python" , quoting=3, names=["idx","time","xwaves","label"]).drop(["idx"], axis=1)

    return accentsData
