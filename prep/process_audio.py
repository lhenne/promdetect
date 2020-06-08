"""
Import necessary packages:
`parselmouth` to send commands to the Praat phonetics software
`numpy` to iterate with high performance
`pandas` to manage data
"""

import parselmouth as pm
import numpy as np
import pandas as pd


# Accent functions

def get_accent_intensity(soundObj, accentsData):

    intensityObj = soundObj.to_intensity()

    intensityValues = list()

    for row in accentsData.itertuples(index=False):
        intensityValues.append(intensityObj.get_value(row.time))

    return intensityValues


def get_accent_f0(soundObj, accentsData, unit):

    pitchObj = soundObj.to_pitch()

    if unit == "Hertz":
        unit = pm.PitchUnit.HERTZ
    elif unit == "ERB":
        unit = pm.PitchUnit.ERB
    else:
        print("Function get_tone_f0: \
           Please provide appropriate unit of measurement")
        raise ValueError

    f0Values = list()

    for row in accentsData.itertuples(index=False):
        f0Values.append(pitchObj.get_value_at_time(
            time=row.time,
            unit=unit
            )
        )

    return f0Values


# Tonal phrase boundary functions

def get_tone_intensity(soundObj, tonesData):

    intensityObj = soundObj.to_intensity()

    intensityValues = list()

    for row in tonesData.itertuples(index=False):
        intensityValues.append(intensityObj.get_value(row.time))

    return intensityValues


def get_tone_f0(soundObj, tonesData, unit):

    pitchObj = soundObj.to_pitch()

    if unit == "Hertz":
        unit = pm.PitchUnit.HERTZ
    elif unit == "ERB":
        unit = pm.PitchUnit.ERB
    else:
        print("Function get_tone_f0: \
            Please provide appropriate unit of measurement")
        raise ValueError

    f0Values = list()

    for row in tonesData.itertuples(index=False):
        f0Values.append(pitchObj.get_value_at_time(
            time=row.time,
            unit=unit
            )
        )

    return f0Values


# Word-based functions

def get_word_intensity(soundObj, transcriptData):

    intensityValues = list()

    for row in transcriptData.itertuples(index=False):

        soundObjWord = soundObj.extract_part(
            from_time=row.start,
            to_time=row.end
            )

        if soundObjWord.total_duration < 0.064:
            min_intensity = np.nan
            max_intensity = np.nan
            mean_intensity = np.nan

        else:
            intensityObjWord = soundObjWord.to_intensity()

            min_intensity = intensityObjWord.get_minimum()
            max_intensity = intensityObjWord.get_maximum()
            mean_intensity = intensityObjWord.get_average()

        word_values = tuple([min_intensity, max_intensity, mean_intensity])

        intensityValues.append(word_values)

    return intensityValues
