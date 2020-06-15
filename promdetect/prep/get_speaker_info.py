"""
Import necessary packages:
`re` for regular expressions
"""

import re


# Get the gender of a speaker from the recording ID

def get_gender(recordingID):

    speakerGender = "unknown"
    # Use speaker info text file to collect all data
    with open("/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/speakers-prosodically-annotated-part.txt", "r") as speakerFile:
        # Search by line if it includes the current recording's ID
        for line in speakerFile:
            if recordingID in line:
                # Clean all unnecessary characters from the line where the ID was found
                speakerGender = re.sub(r"[^a-z]*", "", line)
                break
        # After cleaning, only the last letter in the line remains, which indicates the speaker gender.
        # If the format is not as expected, throw an error
        if speakerGender == "m":
            return "male"
        elif speakerGender == "f":
            return "female"
        else:
            print(speakerGender)
            raise ValueError


# Get the ID of a speaker from the recording ID

def get_id(recordingID):

    speakerID = "unknown"
    # Use speaker info text file to collect all data
    with open("/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/speakers-prosodically-annotated-part.txt", "r") as speakerFile:
        # Search by line if it includes the current recording's ID
        for line in speakerFile:
            if recordingID in line:
                # Extract the ID (a number) of the speaker
                speakerID = re.findall(r"[0-9 ]+SP([0-9]?)[fm]", line)[0]

    if re.match(r"[0-9]+", speakerID):
        return speakerID
    else:
        print("Speaker ID is:", speakerID)
        raise ValueError
