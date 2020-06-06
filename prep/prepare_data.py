""" 
The functions in this module reformat the data from the DIRNDL corpus in order to later be able to make use of them for the acoustic analysis.
"""

import prepare_transcripts
import prepare_tones
import prepare_accents
import get_speaker_info

class DataPreparation:
    

    def __init__(self, corpusPath, recordingID):

        # Assign all relevant file paths to their corresponding variables
        self.transcriptFile = "{0}/dlf-nachrichten-{1}.words".format(corpusPath, recordingID)
        self.tonesFile = "{0}/dlf-nachrichten-{1}.tones".format(corpusPath, recordingID)
        self.accentsFile = "{0}/dlf-nachrichten-{1}.accents".format(corpusPath, recordingID)

        # Get gender and ID of speaker
        self.speakerGender = get_speaker_info.get_gender(recordingID)
        self.speakerID = get_speaker_info.get_id(recordingID)
    
    def transform_data(self):
        
        self.transcript = prepare_transcripts.get_transcript_data(self.transcriptFile)
        self.tones = prepare_tones.get_tones_data(self.tonesFile)
        self.accents = prepare_accents.get_accents_data(self.accentsFile)


a = DataPreparation("/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody", "200703271500")
a.transform_data()
