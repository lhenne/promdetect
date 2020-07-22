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
`pandas` to manage and output data in a nice format
`re` for regular expressions
`io` for a parseable StringIO object to forward to pandas functions
`glob` to collect all recordings that are to be processed
"""

import parselmouth as pm
from parselmouth import praat
import pandas as pd
import re
from io import StringIO
from glob import glob

# Set global values relating to 'determine_nucleus_points()' so that they
# need not be set every time the function is run.

# - silenceThreshold sets a base value to add to intensity maxima (dB), estimating silence
# - minDipBetweenPeaks sets the minimum difference in intensity (dB) between a syllable
#   nucleus and its context
# - minPauseDuration sets the minimum duration in seconds for a pause

silenceThreshold = -25
minDipBetweenPeaks = 2
minPauseDuration = 0.3


# Main function detecting syllable nuclei point-timestamps


def determine_nucleus_points(soundFile):

    # Convert input argument to a parselmouth/Praat Sound object
    rawSoundObj = pm.Sound(soundFile)

    # Clean the sound object by removing noise using a praat function.
    # ["Remove noise"] from start [0] to end [0] of the Sound, work with overlapping windows
    # with length [0.025] seconds, filter everything between frequencies [50] Hz to [10_000] Hz
    # with [40] Hz smoothing factor using the ["Spectral subtraction"] noise reduction method.
    soundObj = praat.call(
        rawSoundObj, "Remove noise", 0, 0, 0.025, 50, 10_000, 40, "Spectral subtraction"
    )

    # Convert the de-noised sound to a parselmouth/Praat intensity object
    intensityObj = soundObj.to_intensity(minimum_pitch=50)

    # Get basic measurements from the sound and intensity objects
    minIntensity = intensityObj.get_minimum()
    maxIntensity = intensityObj.get_maximum()

    # Get the intensity value in the 99th quantile, eliminating extreme outliers
    max99Intensity = praat.call(intensityObj, "Get quantile", 0, 0, 0.99)

    # Define additional thresholds to distinguish nuclei from their surroundings:
    # - threshold1 newly defines the silence threshold, if it is lower than the minimum
    #   intensity (base noise level), set it to the minimum intensity
    # - threshold3 another silence threshold for conversion of the intensity object to
    #   silence annotations
    threshold1 = max99Intensity + silenceThreshold
    threshold2 = maxIntensity - max99Intensity
    threshold3 = silenceThreshold - threshold2
    if threshold1 < minIntensity:
        threshold1 = minIntensity

    # Derive silent and sounding periods from the intensity object: use [threshold3] as
    # the silence threshold, set the minimum duration of silent periods to
    # [minPauseDuration], set the minimum duration of sounding periods to [0.1] seconds,
    # label the two period types with ["silent"]/["sounding"]
    textGridObj = praat.call(
        intensityObj,
        "To TextGrid (silences)",
        threshold3,
        minPauseDuration,
        0.1,
        "silent",
        "sounding",
    )

    # Convert the intensity object to a Matrix, then to a Sound object
    intensityMatrix = praat.call(intensityObj, "Down to Matrix")
    soundFromIntensityMatrix = praat.call(intensityMatrix, "To Sound (slice)", 1)

    # Convert the resulting Sound object to a PointProcess object to estimate peaks.
    # Parameters unclear.
    pointProcess = praat.call(
        soundFromIntensityMatrix,
        "To PointProcess (extrema)",
        "Left",
        "yes",
        "no",
        "Sinc70",
    )

    # Make a list of the peaks from the PointProcess object to establish the number and
    # timestamps of potential syllable nuclei
    numPeaksPrelim = praat.call(pointProcess, "Get number of points")
    peakTimingsPrelim = [
        praat.call(pointProcess, "Get time from index", i + 1)
        for i in range(numPeaksPrelim)
    ]

    # Create empty lists to store timestamps and intensity values, and an index counter
    peakTimings = []
    peakCount = 0
    peakIntensities = []

    # For each of the potential peaks, check if its intensity is above the silence threshold.
    # If yes, append to list
    for i in range(numPeaksPrelim):
        peakValue = praat.call(
            soundFromIntensityMatrix, "Get value at time", peakTimingsPrelim[i], "Cubic"
        )
        if peakValue > threshold1:
            peakCount += 1
            peakIntensities.append(peakValue)
            peakTimings.append(peakTimingsPrelim[i])

    # Create empty list and index counter for validated peaks, keep track of currentTiming
    # and currentIntensity
    validPeakCount = 0
    validPeakTimings = []
    currentTiming = peakTimings[0]
    currentIntensity = peakIntensities[0]

    # For each of the potential nuclei, make the following and preceding peaks accessible
    for peakIndex in range(peakCount - 1):
        followingPeak = peakIndex + 1
        followingTiming = peakTimings[peakIndex + 1]

        if peakIndex > 0:
            precedingPeak = peakIndex - 1
            precedingTiming = peakTimings[precedingPeak]

            # Get the intensity value at the local minimum following the peak
            followingDipIntensity = praat.call(
                intensityObj, "Get minimum", currentTiming, followingTiming, "None"
            )

            # Get the absolute intensity difference between peak and dip
            followingIntensityDifference = abs(currentIntensity - followingDipIntensity)

            # If that difference is greater than the minimum difference, also check for
            # the intensity dip preceding the peak
            if followingIntensityDifference > minDipBetweenPeaks:
                precedingDipIntensity = praat.call(
                    intensityObj, "Get minimum", currentTiming, precedingTiming, "None"
                )
                precedingIntensityDifference = abs(
                    currentIntensity - precedingDipIntensity
                )

                # If that difference is greater than the minimum difference, add the peak
                # and its timestamp to the list of valid peaks
                if precedingIntensityDifference > minDipBetweenPeaks:
                    validPeakCount += 1
                    validPeakTimings.append(peakTimings[peakIndex])

            # Progress to the next peak
            currentTiming = peakTimings[followingPeak]
            currentIntensity = praat.call(
                intensityObj, "Get value at time", peakTimings[followingPeak], "Cubic"
            )

        # If the current peak is at index 0, don't consider the preceding peak
        else:
            followingDipIntensity = praat.call(
                intensityObj, "Get minimum", currentTiming, followingTiming, "None"
            )
            followingIntensityDifference = abs(currentIntensity - followingDipIntensity)
            if followingIntensityDifference > minDipBetweenPeaks:
                validPeakCount += 1
                validPeakTimings.append(peakTimings[peakIndex])
            currentTiming = peakTimings[followingPeak]
            currentIntensity = praat.call(
                intensityObj, "Get value at time", peakTimings[followingPeak], "Cubic"
            )

    # Create a parselmouth/Praat Pitch object from the Sound object, calculate
    # with [0.02] second steps, a minimum pitch of [30] Hz, [4] pitch values,
    # [False]=not a 'very accurate' algorithm, at least [0.03] times the maximum
    # intensity, [0.25] voicing threshold factor, [0.01] octave cost, [0.35] octave-
    # jump cost, [0.25] voiced/unvoiced transition cost and a pitch ceiling of 450 Hz.
    pitchObj = soundObj.to_pitch_ac(
        0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450
    )

    # Create list and index counter to store voiced peaks
    voicedPeakCount = 0
    voicedPeakTimings = []

    # For each of the potential nuclei, find out if they occur during a period where
    # pitch can be measured (i.e. during a voiced sound).
    # First, get the timestamp of the peak and locate it within the Pitch object
    for validPeakIndex in range(validPeakCount):
        queryTiming = validPeakTimings[validPeakIndex]
        queryInterval = praat.call(textGridObj, "Get interval at time", 1, queryTiming)
        queryLabel = praat.call(textGridObj, "Get label of interval", 1, queryInterval)

        # Get the pitch value at the timestamp
        pitchValue = pitchObj.get_value_at_time(queryTiming)

        # If the pitch value is available and if it is during a period not labeled as silent,
        # add to the final list of voiced peaks and potential syllable nuclei
        if pitchValue == pitchValue and queryLabel == "sounding":
            voicedPeakCount += 1
            voicedPeakTimings.append(validPeakTimings[validPeakIndex])

    return voicedPeakTimings


# Additional function estimating durations of syllable nucleus and validating them further


def get_nucleus_durations_labels(recordingFile, syllableNuclei):

    # Create list of characters that can signify vowel or sonorant sounds in SAMPA,
    # the only sounds that can be syllable nuclei
    sampaVowels = [
        "a:",
        "e:",
        "i:",
        "o:",
        "u:",
        "E:",
        "2:",
        "y:",
        "a",
        "E",
        "I",
        "O",
        "U",
        "Y",
        "9",
        "@",
        "6",
        "OY",
        "aI",
        "aU",
        "_6",
        "m",
        "n",
        "l",
        "N",
        "_9",
        "_2:",
    ]

    # Import the annotation data for phones and words
    phoneData = prepare_df("phones", recordingFile)
    wordData = prepare_df("words", recordingFile)

    # Create an empty list to store labels and timestamps in
    nucleiWithLabels = list()

    # Locate the label corresponding to each syllable nucleus timestamp between the 'start' and 'end'
    # timestamps of the phone label.
    for nucleusTiming in syllableNuclei:
        nucleusRow = phoneData.loc[
            (phoneData["start"] <= nucleusTiming) & (phoneData["end"] >= nucleusTiming)
        ]

        # If an annotation for the timestamp has been found, get the additional timestamps and label
        if len(nucleusRow) == 1:
            nucleusStart = nucleusRow.iloc[0]["start"]
            nucleusEnd = nucleusRow.iloc[0]["end"]
            nucleusLabel = nucleusRow.iloc[0]["label"]

            # If the label is a vowel sound label, keep going
            if nucleusLabel in sampaVowels:

                # Find the word annotation that the nucleus belongs to (in the corresponding DataFrame)
                wordRow = wordData.loc[
                    (wordData["start"] <= nucleusTiming)
                    & (wordData["end"] >= nucleusTiming)
                ]

                # If a word annotation for the timestamp has been found, check if it points to one of
                # the 'breathing sounds' in the annotation scheme.
                # Only keep going if the nucleus is not during a breathing sound period
                if (len(wordRow) == 1) & (
                    wordRow.iloc[0]["label"]
                    not in ["[@]", "[t]", "[n]", "[f]", "[h]", "<P>"]
                ):

                    # Append a tuple containing all the data to the prepared list of confirmed nuclei
                    wordLabel = wordRow.iloc[0]["label"]
                    nucleiWithLabels.append(
                        (
                            nucleusStart,
                            nucleusEnd,
                            nucleusTiming,
                            nucleusLabel,
                            wordLabel,
                        )
                    )

                # If the word annotation is a breathing sound, inform the user about it
                else:
                    print(
                        "Suggested nucleus at time",
                        nucleusTiming,
                        "with label",
                        nucleusLabel,
                        "is not part of a word",
                    )

            # If the phone annotation is not a vowel or sonorant, inform the user about it
            else:
                print(
                    "Nucleus at time",
                    nucleusTiming,
                    "with label",
                    nucleusLabel,
                    "is not a vowel or sonorant nucleus",
                )

    # Construct a pandas DataFrame from the information stored in the list of tuples 'nucleiWithLabels'.
    # Each tuple corresponds to a row/entry
    nucleiWithLabels = pd.DataFrame.from_records(
        nucleiWithLabels,
        columns=["start", "end", "timestamp_auto", "phone_label", "word_label"],
    )

    return nucleiWithLabels


def prepare_df(annotationType, recordingFile):

    # Get annotation labels for the current recording
    if annotationType == "phones":
        annotationFile = re.search(r".*[ \w-]+?(?=\.)", recordingFile)[0] + ".phones"
    elif annotationType == "words":
        annotationFile = re.search(r".*[ \w-]+?(?=\.)", recordingFile)[0] + ".words"

    # Use ISO 8859-1 (Western European) encoding to open TSV file containing the phone labels
    with open(annotationFile, "r", encoding="iso-8859-1") as f:
        # Don't save any lines until a lone hash sign is found in the line
        # This signifies the end of the metadata and start of the data block
        while f.readline() != "#\n":
            pass
        rawContent = f.read()

    # Clean and convert to a StringIO object to be able to parse it with pandas
    content = clean_text(rawContent, annotationType)

    # Create a pandas DataFrame from the data
    annotationData = pd.read_csv(
        content,
        sep=" ",
        engine="python",
        quoting=3,
        names=["idx", "end", "xwaves", "label"],
    ).drop(["idx"], axis=1)

    # Create an empty column to which the start timestamps for each word will be assigned
    annotationData["start"] = None

    # Iterate over the rows of phoneData, and use the 'end' timestamp from the preceding row
    # to derive the 'start' timestamp for the current one.
    # The 'start' timestamp is estimated by adding 10 milliseconds to the 'end' timestamp of the previous word
    for i in annotationData.index:
        if i <= len(annotationData.index) and i > 0:
            annotationData.at[i, "start"] = annotationData.loc[i - 1, "end"] + 0.001
        elif i == 0:
            annotationData.at[i, "start"] = 0

    # Convert the values in columns 'start' and 'end' to float64 so they can be directly compared with nucleus timestamps
    annotationData.astype({"start": "float64", "end": "float64"})

    return annotationData


def clean_text(rawText, annotationType):
    # Reduce all instances two or more consecutive whitespaces to one whitespace
    text = re.sub(r" {2,}", " ", rawText)

    # If the current annotations are at word level, replace escaped Umlaute with proper ones
    # If the current annotations are at phone level, nothing needs to be done
    if annotationType == "words":
        replacements = {
            '"u': "ü",
            '"U': "Ü",
            '"a': "ä",
            '"A': "Ä",
            '"o': "ö",
            '"O': "Ö",
            '"s': "ß",
            '"S': "ß",
        }

        for string, replacement in replacements.items():
            text = text.replace(string, replacement)

    elif annotationType == "phones":
        pass

    else:
        raise ValueError

    # Return a StringIO object of the cleaned string that can be parsed into a pandas DataFrame
    return StringIO(text)


def run_for_all_files():

    # Collect all recording .wav-files in the source directory
    recordings = glob(
        "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/*.wav"
    )

    # For each of the recordings, get timestamps where syllable nuclei are estimated to be,
    # then derive the durations of these syllable nuclei by collecting the 'start' and 'end'
    # timestamps from the phone annotations
    for wavFile in recordings:

        syllableNuclei = determine_nucleus_points(wavFile)

        nucleiWithDurations = get_nucleus_durations_labels(wavFile, syllableNuclei)

        # Output the pandas DataFrame to one separate file per recording
        outputPath = (
            "data/dirndl/nuclei/" + re.search(r"[ \w-]+?(?=\.)", wavFile)[0] + ".nuclei"
        )

        with open(outputPath, "w+") as outputFile:
            nucleiWithDurations.to_csv(outputFile, sep=",", index=False)
