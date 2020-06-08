"""
Import necessary packages:
`parselmouth` to send commands to the Praat phonetics software
`numpy` to iterate with high performance
`pandas` to manage data
"""

import parselmouth as pm
import numpy as np
import pandas as pd


def get_accent_intensity(soundObj, accentsData):

    intensityObj = soundObj.to_intensity()

    intensityValues = list()

    for row in accentsData.itertuples(index=False):
        intensityValues.append(intensityObj.get_value(row.time))

    return intensityValues


def get_accent_f0(soundObj, accentsData):

    pitchObj = soundObj.to_pitch()

    f0Values = list()

    for row in accentsData.itertuples(index=False):
        f0Values.append(pitchObj.get_value_at_time(
            time=row.time,
            unit=pm.PitchUnit.HERTZ
            )
        )

    return f0Values
