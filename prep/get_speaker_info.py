import re

#### Get the gender of a speaker from their ID

def get_gender(recordingID):

    with open("/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/speakers-prosodically-annotated-part.txt", "r") as speakerFile: # Use speaker info text file
        for line in speakerFile:
            if recordingID in line:
                speakerGender = re.sub(r"[^a-z]*", "", line) # Clean all unnecessary characters from the line where the ID was found
                break
        if speakerGender == "m":
            return "male"
        elif speakerGender == "f":
            return "female"
        else:
            print(speakerGender)
            raise ValueError


def get_id(recordingID):

    with open("/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/speakers-prosodically-annotated-part.txt", "r") as speakerFile: # Use speaker info text file
        for line in speakerFile:
            if recordingID in line:
                speakerID = re.findall(r"[0-9 ]+SP([0-9]?)[fm]", line)[0]
    return speakerID    
