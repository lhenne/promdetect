"""
Import necessary packages:
`parselmouth` to send commands to the Praat phonetics software
`numpy` to iterate with high performance
`math` to calculate pitch excursion
`pandas` to manage data
"""

import parselmouth as pm
from parselmouth.praat import call
import numpy as np
import math
import pandas as pd


# Accent functions

def get_accent_intensity(soundObj, accentsData):

    intensityObj = soundObj.to_intensity(minimum_pitch=50)

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
        print("Function get_accent_f0: \
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

    intensityObj = soundObj.to_intensity(minimum_pitch=50)

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


# Word-level functions

def get_word_intensity(soundObj, transcriptData):

    intensityValues = list()

    for row in transcriptData.itertuples(index=False):

        soundObjWord = soundObj.extract_part(
            from_time=row.start,
            to_time=row.end
            )

        if soundObjWord.total_duration < 0.064 or row.label in ["[@]",
                                                                "[t]",
                                                                "[n]",
                                                                "[f]",
                                                                "[h]",
                                                                "<P>"]:
            min_intensity = np.nan
            max_intensity = np.nan
            mean_intensity = np.nan

        else:
            intensityObjWord = soundObjWord.to_intensity(minimum_pitch=50)

            min_intensity = intensityObjWord.get_minimum()
            max_intensity = intensityObjWord.get_maximum()
            mean_intensity = intensityObjWord.get_average()

        word_values = tuple([min_intensity, max_intensity, mean_intensity])

        intensityValues.append(word_values)

    return intensityValues


def get_word_f0(soundObj, transcriptData, unit):

    f0Values = list()

    if unit not in ["Hertz", "ERB"]:
        print("Function get_word_f0: \
            Please provide appropriate unit of measurement")
        raise ValueError

    for row in transcriptData.itertuples(index=False):

        soundObjWord = soundObj.extract_part(
            from_time=row.start,
            to_time=row.end
            )

        if row.label in ["[@]", "[t]", "[n]", "[f]", "[h]", "<P>"]:
            min_f0 = np.nan
            max_f0 = np.nan
            mean_f0 = np.nan
        else:
            pitchObjWord = soundObjWord.to_pitch()

            min_f0 = call(pitchObjWord,
                          "Get minimum",
                          0,
                          0,
                          unit,
                          "Parabolic")
            max_f0 = call(pitchObjWord,
                          "Get maximum",
                          0,
                          0,
                          unit,
                          "Parabolic")
            mean_f0 = call(pitchObjWord,
                           "Get mean",
                           0,
                           0,
                           unit)

        word_values = tuple([min_f0, max_f0, mean_f0])

        f0Values.append(word_values)

    return f0Values


def get_accent_f0_excursion(soundObj, accentData, transcriptData):

    excursionValues = list()

    for row in accentData.itertuples(index=False):
        accentTimestamp = row.time
        accentF0 = row.f0_hz

        wordAtTimestamp = transcriptData.loc[
            (transcriptData["start"] <= accentTimestamp) &
            (transcriptData["end"] >= accentTimestamp)]
        lowEndF0 = wordAtTimestamp.min_f0_hz

        f0Excursion = math.log2(accentF0/lowEndF0)

        excursionValues.append(f0Excursion)

    return excursionValues
