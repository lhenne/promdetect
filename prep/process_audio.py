"""
Import necessary packages:
`parselmouth` to send commands to the Praat phonetics software
`numpy` to iterate with high performance
`pandas` to manage data
"""

import parselmouth as pm
import numpy as np
import pandas as pd


def get_accent_intensity(wavFile, accentsData):

    soundObj = pm.Sound(wavFile)
    intensityObj = soundObj.to_intensity()

    intensityValues = list()

    for row in accentsData.itertuples(index=False):
        intensityValues.append(intensityObj.get_value(row.time))

    return intensityValues
