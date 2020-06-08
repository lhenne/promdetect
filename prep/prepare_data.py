"""
Import necessary packages:
`pandas` to manage data
"""

import pandas as pd

import prepare_transcripts as transcripts
import prepare_tones as tones
import prepare_accents as accents
import get_speaker_info as speaker
import process_audio as audio

"""
The functions in this module reformat the data from the DIRNDL corpus in order
to later be able to make use of them for the acoustic analysis.
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

        self.transcript = transcripts.get_transcript_data(
            self.transcriptFile)
        self.tones = tones.get_tones_data(self.tonesFile)
        self.accents = accents.get_accents_data(self.accentsFile)

    def compute_audio_values(self):

        self.accents["intensity"] = audio.get_accent_intensity(
            self.wavFile,
            self.accents)


a = DataPreparation(
    "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody",
    "200703271500")
a.transform_annotations()
a.compute_audio_values()
