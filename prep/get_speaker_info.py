import re

def get_gender(recordingID):

    with open("/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/speakers-prosodically-annotated-part.txt", "r") as speakerFile:
        for line in speakerFile:
            if recordingID in line:
                speakerGender = re.sub(r"[^a-z]*", "", line)
                break
        if speakerGender == "m":
            return "male"
        elif speakerGender == "f":
            return "female"
        else:
            print(speakerGender)
            raise ValueError