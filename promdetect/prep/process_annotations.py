"""
Import necessary packages:
`re` for regular expressions
`pandas` for providing a data structure to store the CSV data in within Python
`io.StringIO` to provide an object for pandas functions to parse
`pathlib.Path` to conveniently extend supplied relative paths
`numpy.nan` to insert NaN values where needed
"""

import re
import pandas as pd
from io import StringIO
from pathlib import Path
from numpy import nan


def get_annotation_data(annotation_file) -> pd.DataFrame:
    """
    This function reads files containing annotations and processes the data, storing it in a pandas.DataFrame object.
    The type of annotation (e.g. word-level or accent-level) is deduced from the file extension.
    Unnecessary information from the files is not returned, only timestamps and labels are.
    """

    annotation_file = Path(annotation_file).resolve()

    if annotation_file.is_file():
        pass
    else:
        raise FileNotFoundError("Input file could not be found.")

    annotation_type = annotation_file.suffix[
        1:
    ]  # Get annotation type from file extension, but without the dot

    if annotation_type in ["accents", "phones", "tones", "words"]:
        pass
    else:
        raise ValueError("Input file does not contain supported annotation type")

    raw_content = read_file(annotation_file)

    content = StringIO(
        clean_text(raw_content, annotation_type)
    )  # Store string in a StringIO object to enable pandas to parse it like a CSV file.

    data = content_to_df(content, annotation_type)

    return data


def get_speaker_info(recording_id):
    """
    This functions reads the file "speakers-prosodically-annotated-part.txt" and finds the supplied recording ID in that file, returning a tuple with the corresponding speaker's ID and gender.
    The ID will later be used to investigate cross-speaker similarities and differences, the gender is relevant to the calculation of acoustical features.
    """

    with open(
        Path(
            "../quelldaten/DIRNDL-prosody/speakers-prosodically-annotated-part.txt"
        ).resolve(),
        "r",
    ) as speakerFile:
        found_id = False

        for line in speakerFile:
            if recording_id in line:
                found_id = True
                # Extract ID number of the speaker
                speaker_id = re.findall(r"[0-9 \t]+SP([0-9]?)[fm]", line)[0]
                speaker_gender = re.sub(r"[^a-z]*", "", line)

                if speaker_id != "" and speaker_gender in ["m", "f"]:
                    return (speaker_id, speaker_gender)
                else:
                    raise ValueError(
                        "Speaker ID and/or gender could not be determined."
                    )

        if not found_id:
            raise ValueError("Supplied recording ID could not be found.")


# ANCILLARY FUNCTIONS


def clean_text(raw_text, annotation_type) -> str:
    # Reduce all instances two or more consecutive whitespaces to one whitespace, remove leading and trailing whitespaces
    text = re.sub("[ ]{2,}", " ", raw_text).lstrip()
    text = re.sub("[ ]*\\n[ ]*", "\n", text)

    if annotation_type == "words":
        # Replace escaped Umlauts with proper ones
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

    elif annotation_type in ["accents", "phones", "tones"]:
        pass

    else:
        raise ValueError

    return text


def read_file(input_file) -> str:

    # Use ISO 8859-1 (Western European) encoding to open TSV file containing the annotations
    with open(input_file, "r", encoding="iso-8859-1") as f:
        # Don't save any lines until a lone hash sign is found in the line
        # This signifies the end of the metadata and start of the data block
        while f.readline() != "#\n":
            pass
        rawContent = f.read()

    return rawContent


def content_to_df(content, annotation_type):
    """
    This function processes the StringIO object containing annotation file content and returns a cleaned pandas.DataFrame object with added info about estimated start times for phone and word boundaries
    """

    if annotation_type in ["phones", "words"]:
        content_as_df = pd.read_csv(
            content,
            sep=" ",
            engine="python",
            quoting=3,
            names=["end", "xwaves", "label"],
        ).drop(
            ["xwaves"], axis=1
        )  # Drop unnecessary xwaves column straight away

        content_as_df["start_est"] = nan

        num_rows = len(content_as_df.index)

        # Add data for estimated start timestamps of each word, which are 1ms after the end timestamp of the previous label (to avoid overlap for now). Set the starting time of the first label to N/A.
        for i in content_as_df.index:
            if i <= num_rows and i > 0:
                content_as_df.at[i, "start_est"] = (
                    content_as_df.loc[i - 1, "end"] + 0.0001
                )
            else:
                pass

    else:
        content_as_df = pd.read_csv(
            content,
            sep=" ",
            engine="python",
            quoting=3,
            names=["time", "xwaves", "label"],
        ).drop(
            ["xwaves"], axis=1  # Drop unnecessary xwaves column straight away)
        )

    return content_as_df
