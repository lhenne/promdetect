"""
This script detects syllable nuclei in a sound file
and returns their timestamps.

It is a loose translation of a Praat script by Nivja de Jong and Ton Wempe[1],
and takes aspects from David Feinberg's Python translation[2] of that script.

[1]: https://sites.google.com/site/speechrate/Home/praat-script-syllable-nuclei-v2
[2]: https://github.com/drfeinberg/PraatScripts/blob/master/syllable_nuclei.py

The analysis is run on filtered versions of the input files, filtering is done
by means of an internal Praat function with custom parameters.
This script adds the condition that a syllable nucleus not only has to be followed
by a 2 dB dip in intensity, but that it also has to be preceded by one.

Import necessary packages:
`parselmouth` to send commands to the Praat phonetics software
`re` for regular expressions
`glob` to collect all recordings that are to be processed
"""

import parselmouth as pm
from parselmouth.praat import call
import pandas as pd
import re
from io import StringIO
from glob import glob

silenceThreshold = -25
minDipBetweenPeaks = 2
minPauseDuration = 0.3


def determine_nucleus_points(soundFile):
    rawSoundObj = pm.Sound(soundFile)
    soundObj = call(rawSoundObj, "Remove noise", 0, 0, 0.025,
                    50, 10000, 40, "Spectral subtraction")
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

        if peakIndex > 0:
            precedingPeak = peakIndex - 1
            precedingTiming = peakTimings[peakIndex - 1]

            followingDipIntensity = call(intensityObj, "Get minimum",
                                         currentTiming, followingTiming, "None")
            followingIntensityDifference = abs(
                currentIntensity - followingDipIntensity)
            if followingIntensityDifference > minDipBetweenPeaks:
                precedingDipIntensity = call(
                    intensityObj, "Get minimum", currentTiming, precedingTiming, "None")
                precedingIntensityDifference = abs(
                    currentIntensity - precedingDipIntensity)
                if precedingIntensityDifference > minDipBetweenPeaks:
                    validPeakCount += 1
                    validPeakTimings.append(peakTimings[peakIndex])
            currentTiming = peakTimings[followingPeak]
            currentIntensity = call(intensityObj, "Get value at time",
                                    peakTimings[followingPeak], "Cubic")
        else:
            followingDipIntensity = call(intensityObj, "Get minimum",
                                         currentTiming, followingTiming, "None")
            followingIntensityDifference = abs(
                currentIntensity - followingDipIntensity)
            if followingIntensityDifference > minDipBetweenPeaks:
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


def get_nucleus_durations_labels(recordingFile, syllableNuclei):

    # Create list of characters that can signify vowel sounds in SAMPA
    sampaVowels = ["a:", "e:", "i:", "o:", "u:", "E:", "2:",
                   "y:", "a", "E", "I", "O", "U", "Y", "9",
                   "@", "6", "OY", "aI", "aU", "_6", "m",
                   "n", "l", "N", "_9", "_2:"]

    phoneData = prepare_df("phones", recordingFile)
    wordData = prepare_df("words", recordingFile)

    # Create an empty list to store labels and timestamps in
    nucleiWithLabels = list()

    # Locate the label corresponding to each syllable nucleus timestamp
    # If the label is a vowel sound label, add the combination to the list of confirmed syllable nuclei
    for nucleusTiming in syllableNuclei:
        nucleusRow = phoneData.loc[(phoneData["start"] <= nucleusTiming) & (
            phoneData["end"] >= nucleusTiming)]

        if len(nucleusRow) == 1:
            nucleusStart = nucleusRow.iloc[0]["start"]
            nucleusEnd = nucleusRow.iloc[0]["end"]
            nucleusLabel = nucleusRow.iloc[0]["label"]

            if nucleusLabel in sampaVowels:

                wordRow = wordData.loc[(wordData["start"] <= nucleusTiming) & (
                    wordData["end"] >= nucleusTiming)]

                if (len(wordRow) == 1) & (wordRow.iloc[0]["label"] not in ["[@]", "[t]", "[n]", "[f]", "[h]", "<P>"]):
                    wordLabel = wordRow.iloc[0]["label"]
                    nucleiWithLabels.append(
                        (nucleusStart, nucleusEnd, nucleusTiming, nucleusLabel, wordLabel))
                else:
                    print("Suggested nucleus at time",
                          nucleusTiming, "with label", nucleusLabel, "is not part of a word")
            else:
                print("Nucleus at time", nucleusTiming, "with label",
                      nucleusLabel, "is not a vowel or sonorant nucleus")

    nucleiWithLabels = pd.DataFrame.from_records(
        nucleiWithLabels, columns=["start", "end", "timestamp_auto", "phone_label", "word_label"])

    return nucleiWithLabels


def prepare_df(annotationType, recordingFile):

    # Get annotation labels for the current recording
    if annotationType == "phones":
        annotationFile = re.search(
            ".*[ \w-]+?(?=\.)", recordingFile)[0] + ".phones"
    elif annotationType == "words":
        annotationFile = re.search(
            ".*[ \w-]+?(?=\.)", recordingFile)[0] + ".words"

    # Use ISO 8859-1 (Western European) encoding to open TSV file containing the phone labels
    with open(annotationFile, "r", encoding="iso-8859-1") as f:
        # Don't save any lines until a lone hash sign is found in the line
        # This signifies the end of the metadata and start of the data block
        while f.readline() != "#\n":
            pass
        rawContent = f.read()

    # Clean and convert to a StringIO object to be able to parse it with pandas
    content = StringIO(re.sub(r" {2,}", " ", rawContent))

    # Create a pandas DataFrame from the data
    annotationData = pd.read_csv(content,
                                 sep=" ",
                                 engine="python",
                                 quoting=3,
                                 names=["idx",
                                        "end",
                                        "xwaves",
                                        "label"]).drop(["idx"], axis=1)

    # Create an empty column to which the start timestamps for each word will be assigned
    annotationData["start"] = None

    # Iterate over the rows of phoneData, and use the 'end' timestamp from the preceding row
    # to derive the 'start' timestamp for the current one.
    # The 'start' timestamp is estimated by adding 10 milliseconds to the 'end' timestamp of the previous word
    for i in annotationData.index:
        if i <= len(annotationData.index) and i > 0:
            annotationData.at[i, "start"] = annotationData.loc[i - 1,
                                                               "end"] + 0.001
        elif i == 0:
            annotationData.at[i, "start"] = 0

    # Convert the values in columns 'start' and 'end' to float64 so they can be directly compared with nucleus timestamps
    annotationData.astype({"start": "float64", "end": "float64"})

    return annotationData


def run_for_all_files():

    recordings = glob(
        "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/*.wav")

    for wavFile in recordings:

        syllableNuclei = determine_nucleus_points(wavFile)

        nucleiWithDurations = get_nucleus_durations_labels(
            wavFile, syllableNuclei)

        outputPath = "data/dirndl/nuclei/" + \
            re.search("[ \w-]+?(?=\.)", wavFile)[0] + ".nuclei"
        with open(outputPath, "w+") as outputFile:
            nucleiWithDurations.to_csv(outputFile, sep=",")


run_for_all_files()
# To save the nucleus data to a Praat TextGrid, uncomment and include the following lines:
# call(textGridObj, "Insert point tier", 1, "nuclei")
# for i in range(len(voicedPeakTimings)):
#     call(textGridObj, "Insert point", 1, voicedPeakTimings[i], "")

# textGridObj.save_as_text_file(str(wavFile + ".TextGrid"))
