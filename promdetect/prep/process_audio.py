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


# Intensity functions

def get_intensity(intensityObj, annotationData):

    # Create an empty list and iterate over rows of the annotation DataFrame to add intensity point values
    # for each of the accent timings one by one
    intensityValues = list()

    for row in annotationData.itertuples(index=False):
        intensityValues.append(intensityObj.get_value(row.time))

    return intensityValues


# F0 functions

def get_f0(pitchObj, annotationData, unit):

    # Depending on the unit of measurement given as a function parameter, convert to a suitable parameter
    # for usage with parselmouth
    if unit == "Hertz":
        unit = pm.PitchUnit.HERTZ
    elif unit == "ERB":
        unit = pm.PitchUnit.ERB
    else:
        print("Function get_f0: \
           Please provide appropriate unit of measurement")
        raise ValueError

    # Create an empty list and iterate over rows of the annotation DataFrame to add F0 point values
    # for each of the accent timings one by one, using the previously determined unit of measurement
    f0Values = list()

    for row in annotationData.itertuples(index=False):
        f0Values.append(pitchObj.get_value_at_time(
            time=row.time,
            unit=unit
        )
        )

    return f0Values


def get_accent_f0_excursion(soundObj, accentData, transcriptData):

    excursionValues = list()

    for row in accentData.itertuples(index=False):
        accentTimestamp = row.time
        accentF0 = row.f0_hz

        wordAtTimestamp = transcriptData.loc[
            (transcriptData["start"] <= accentTimestamp) & (transcriptData["end"] >= accentTimestamp)]
        lowEndF0 = wordAtTimestamp.min_f0_hz

        f0Excursion = math.log2(accentF0 / lowEndF0)

        excursionValues.append(f0Excursion)

    return excursionValues


# Word-level functions

# Get intensity values for all accent annotations in a recording through Praat.
# As word annotations extend across a timespan instead of being individual points,
# the minimum, maximum and mean values for intensity are calculated
def get_word_intensity(soundObj, transcriptData):

    # Create an empty list to store all the intensity measurements in
    intensityValues = list()

    # Iterating over the rows of the word-level annotation DataFrame, extract the part of
    # the audio recording corresponding to the labelled word
    for row in transcriptData.itertuples(index=False):

        soundObjWord = soundObj.extract_part(
            from_time=row.start,
            to_time=row.end
        )

        # Don't measure intensity in two cases:
        # (a) The word duration is too short to provide enough material for the calculation
        #     Praat requires the duration to be at least 6.4 divided by the minimum pitch (~0.106s)
        # (b) The label for the word indicates that it is a phrase boundary or breathing sound
        if soundObjWord.total_duration < 0.106 or row.label in ["[@]",
                                                                "[t]",
                                                                "[n]",
                                                                "[f]",
                                                                "[h]",
                                                                "<P>"]:
            # Instead, return NA values
            min_intensity = np.nan
            max_intensity = np.nan
            mean_intensity = np.nan

        # Otherwise, create an intensity contour (parselmouth intensity object) just for the word,
        # with minimum speaker pitch of 50Hz
        else:
            intensityObjWord = soundObjWord.to_intensity(minimum_pitch=60)

            # Get the minimum, maximum and mean intensity under the duration of the word production
            min_intensity = intensityObjWord.get_minimum()
            max_intensity = intensityObjWord.get_maximum()
            mean_intensity = intensityObjWord.get_average()

        # Create a 3-tuple out of these measurements and append it to the output list
        word_values = tuple([min_intensity, max_intensity, mean_intensity])
        intensityValues.append(word_values)

    return intensityValues


# Get F0 values for all accent annotations in a recording through Praat.
# As word annotations extend across a timespan instead of being individual points,
# the minimum, maximum and mean values for F0 are calculated
def get_word_f0(soundObj, transcriptData, unit):

    # Create an empty list to store all the intensity measurements in
    f0Values = list()

    # If the unit of measurement specified as a function parameter does not match one of the
    # intended units, throw an error.
    # The 'unit' parameter of type string does not have to be converted in this case, as the 'call'
    # function later on will just accept the strings
    if unit not in ["Hertz", "ERB"]:
        print("Function get_word_f0: \
            Please provide appropriate unit of measurement")
        raise ValueError

    # Iterating over the rows of the word-level annotation DataFrame, extract the part of
    # the audio recording corresponding to the labelled word
    for row in transcriptData.itertuples(index=False):

        soundObjWord = soundObj.extract_part(
            from_time=row.start,
            to_time=row.end
        )

        # Don't measure F0 if the label for the word indicates a phrase boundary or breathing sound
        if row.label in ["[@]", "[t]", "[n]", "[f]", "[h]", "<P>"]:
            # Instead, return NA values
            min_f0 = np.nan
            max_f0 = np.nan
            mean_f0 = np.nan

        # Otherwise, create a pitch contour (parselmouth pitch object) just for the word
        else:
            pitchObjWord = soundObjWord.to_pitch()

            # Get the minimum, maximum and mean F0 under the duration of the word production
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

        # Create a 3-tuple out of these measurements and append it to the output list
        word_values = tuple([min_f0, max_f0, mean_f0])
        f0Values.append(word_values)

    return f0Values
