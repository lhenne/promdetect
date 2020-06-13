"""
Import necessary packages:
`pandas` to manage data
`parselmouth` to send commands to the Praat phonetics software
"""

import pandas as pd
import parselmouth as pm

import promdetect.prep.process_annotations as annotations
import promdetect.prep.get_speaker_info as speaker
import promdetect.prep.process_audio as audio

"""
The functions in this module reformat the data from the DIRNDL corpus in order
to later be able to make use of them for the acoustic analysis.
This file coordinates and collects data by running functions defined in some of
the other files in this directory.
"""


class DataPreparation:

    def __init__(self, corpusPath, recordingID):

        # Assign all relevant file paths to their corresponding variables
        self.transcriptFile = "{0}/dlf-nachrichten-{1}.words".format(
            corpusPath,
            recordingID
        )
        self.tonesFile = "{0}/dlf-nachrichten-{1}.tones".format(
            corpusPath,
            recordingID
        )
        self.accentsFile = "{0}/dlf-nachrichten-{1}.accents".format(
            corpusPath,
            recordingID
        )
        self.wavFile = "{0}/dlf-nachrichten-{1}.wav".format(
            corpusPath,
            recordingID
        )

        # Get gender and ID of speaker
        self.speakerGender = speaker.get_gender(recordingID)
        self.speakerID = speaker.get_id(recordingID)

    def transform_annotations(self):

        # Process and transform the annotation data from CSV files to pandas DataFrames
        self.transcript = annotations.get_annotation_data(
            self.transcriptFile,
            "words")
        self.tones = annotations.get_annotation_data(
            self.tonesFile,
            "tones")
        self.accents = annotations.get_annotation_data(
            self.accentsFile,
            "accents")

    def compute_audio_values(self):

        # Create Praat/parselmouth sound object to use across all following audio processing functions
        soundObj = pm.Sound(self.wavFile)

        # Create Praat/parselmouth intensity object to use across functions for tones and accents
        # The minimum possible pitch for speaker voices is assumed to be 60Hz, the minimum for male speakers
        # (cf. Cruttenden 1996)
        intensityObj = soundObj.to_intensity(minimum_pitch=60)

        # Calculate a pitch contour from the parselmouth sound object to use across functions for tones and
        # accents
        pitchObj = soundObj.to_pitch()

        # Get intensity and F0 (pitch) point values for the timestamps given for accents
        self.accents["intensity"] = audio.get_intensity(
            intensityObj,
            self.accents
        )
        self.accents["f0_hz"] = audio.get_f0(
            pitchObj,
            self.accents,
            "Hertz"
        )
        self.accents["f0_erb"] = audio.get_f0(
            pitchObj,
            self.accents,
            "ERB"
        )

        # Get intensity and F0 (pitch) point values for the timestamps given for tonal phrase boundaries
        self.tones["intensity"] = audio.get_intensity(
            intensityObj,
            self.tones
        )
        self.tones["f0_hz"] = audio.get_f0(
            pitchObj,
            self.tones,
            "Hertz"
        )
        self.tones["f0_erb"] = audio.get_f0(
            pitchObj,
            self.tones,
            "ERB"
        )

        # Create temporary DataFrames for intensity and F0 (pitch) measurements:
        # Three columns are needed for each, which would make their direct addition to the word-level
        # DataFrame more complicated than necessary
        tmpWordIntensities = pd.DataFrame(audio.get_word_intensity(
            soundObj,
            self.transcript
        ), columns=["min_intensity",
                    "max_intensity",
                    "mean_intensity"]
        )
        tmpWordPitchHz = pd.DataFrame(audio.get_word_f0(
            soundObj,
            self.transcript,
            "Hertz"
        ), columns=["min_f0_hz",
                    "max_f0_hz",
                    "mean_f0_hz"]
        )
        tmpWordPitchERB = pd.DataFrame(audio.get_word_f0(
            soundObj,
            self.transcript,
            "ERB"
        ), columns=["min_f0_erb",
                    "max_f0_erb",
                    "mean_f0_erb"]
        )

        # Concatenate all four DataFrames containing word-level data into one
        self.transcript = pd.concat([self.transcript,
                                     tmpWordIntensities,
                                     tmpWordPitchHz,
                                     tmpWordPitchERB],
                                    axis=1)

        # TODO: check if excursion measurements are proper
        # Get F0 (pitch) excursion measurements for accent annotations in semitones using a standard formula
        self.accents["f0_excursion"] = audio.get_accent_f0_excursion(
            soundObj,
            self.accents,
            self.transcript
        )


# Initialize an instance of the DataPreparation class for an example recording and run
# the annotation and audio processing functions on it to prepare the data for further use
a = DataPreparation(
    "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody",
    "200703271500")
a.transform_annotations()
a.compute_audio_values()
