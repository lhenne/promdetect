""" 
The functions in this module reformat the data from the DIRNDL corpus in order to later be able to make use of them for the acoustic analysis.
"""

import prepare_transcriptions
import prepare_tones
import prepare_accents
import get_speaker_info

class DataPreparation:
    

    def __init__(self, corpusPath, recordingID):
        # assign all relevant file paths to their corresponding variables
        self.transcriptFile = "{0}/dlf-nachrichten-{1}.words".format(corpusPath, recordingID)
        self.tonesFile = "{0}/dlf-nachrichten-{1}.tones".format(corpusPath, recordingID)
        self.accentsFile = "{0}/dlf-nachrichten-{1}.accents".format(corpusPath, recordingID)

        # get gender of speaker
        self.speakerGender = get_speaker_info.get_gender(recordingID)

a = DataPreparation("~/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody", "200703271500")
