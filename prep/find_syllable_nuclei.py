"""
This script detect syllable nuclei in a sound file
and returns their timestamps as annotations.

It is a loose translation of a Praat script by Nivja de Jong and Ton Wempe,
and takes aspects from David Feinberg's Python translation of that script.
"""

import parselmouth as pm
from parselmouth.praat import call
import re
from glob import glob

silenceThreshold = -25
minDipBetweenPeaks = 2
minPauseDuration = 0.3

recordings = glob(
    "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/*.wav")


def determine_nucleus_points(soundFile):
    soundObj = pm.Sound(soundFile)
    intensityObj = soundObj.to_intensity(minimum_pitch=50)

    totalDuration = soundObj.get_total_duration()
    minIntensity = intensityObj.get_minimum()
    maxIntensity = intensityObj.get_maximum()
    max99Intensity = call(intensityObj, "Get quantile", 0, 0, 0.99)

    threshold1 = max99Intensity + silenceThreshold
    threshold2 = maxIntensity - max99Intensity
    threshold3 = silenceThreshold - threshold2
    if threshold1 < minIntensity:
        threshold1 = minIntensity

    textGridObj = call(intensityObj, "To TextGrid (silences)",
                       threshold3, minPauseDuration, 0.1, "silent", "sounding")

    intensityMatrix = call(intensityObj, "Down to Matrix")
    soundFromIntensityMatrix = call(intensityMatrix, "To Sound (slice)", 1)
    maxIntensity = soundFromIntensityMatrix.get_maximum()

    pointProcess = call(soundFromIntensityMatrix,
                        "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")

    numPeaksPrelim = call(pointProcess, "Get number of points")
    peakTimingsPrelim = [call(pointProcess, "Get time from index", i + 1)
                         for i in range(numPeaksPrelim)]

    peakTimings = []
    peakCount = 0
    peakIntensities = []
    for i in range(numPeaksPrelim):
        peakValue = call(soundFromIntensityMatrix,
                         "Get value at time", peakTimingsPrelim[i], "Cubic")
        if peakValue > threshold1:
            peakCount += 1
            peakIntensities.append(peakValue)
            peakTimings.append(peakTimingsPrelim[i])

    validPeakCount = 0
    validPeakTimings = []
    currentTiming = peakTimings[0]
    currentIntensity = peakIntensities[0]

    for peakIndex in range(peakCount - 1):
        followingPeak = peakIndex + 1
        followingTiming = peakTimings[peakIndex + 1]

        dipIntensity = call(intensityObj, "Get minimum",
                            currentTiming, followingTiming, "None")
        intensityDifference = abs(currentIntensity - dipIntensity)
        if intensityDifference > minDipBetweenPeaks:
            validPeakCount += 1
            validPeakTimings.append(peakTimings[peakIndex])
        currentTiming = peakTimings[followingPeak]
        currentIntensity = call(intensityObj, "Get value at time",
                                peakTimings[followingPeak], "Cubic")

    pitchObj = soundObj.to_pitch_ac(
        0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedPeakCount = 0
    voicedPeakTimings = []

    for validPeakIndex in range(validPeakCount):
        queryTiming = validPeakTimings[validPeakIndex]
        queryInterval = call(
            textGridObj, "Get interval at time", 1, queryTiming)
        queryLabel = call(
            textGridObj, "Get label of interval", 1, queryInterval)

        pitchValue = pitchObj.get_value_at_time(queryTiming)
        if pitchValue == pitchValue and queryLabel == "sounding":
            voicedPeakCount += 1
            voicedPeakTimings.append(validPeakTimings[validPeakIndex])

    return voicedPeakTimings


for wavFile in recordings:

    syllableNuclei = determine_nucleus_points(wavFile)

    outputPath = "data/" + \
        re.search("[ \w-]+?(?=\.)", wavFile)[0] + ".nuclei"
    with open(outputPath, "w+") as outputFile:
        for nucleus in syllableNuclei:
            outputFile.write(str(nucleus) + "\n")


# To save the nucleus data to a Praat TextGrid, uncomment and include the following lines:

# call(textGridObj, "Insert point tier", 1, "nuclei")
# for i in range(len(voicedPeakTimings)):
#     call(textGridObj, "Insert point", 1, voicedPeakTimings[i], "")

# textGridObj.save_as_text_file(".TextGrid")
