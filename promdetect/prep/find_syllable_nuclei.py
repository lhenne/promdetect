"""
This script detects syllable nuclei in a sound file
and returns their timestamps.

It is a loose translation of a Praat script by Nivja de Jong and Ton Wempe:
https://sites.google.com/site/speechrate/Home/praat-script-syllable-nuclei-v2

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
from pandas import DataFrame

SAMPA_VOWELS = [
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
    # "m",  # TODO: Decide on these glides and nasals
    # "n",
    # "l",
    # "N",
    "_9",
    "_2:",
]
EXTRALING_SOUNDS = ["[@]", "[t]", "[n]", "[f]", "[h]", "<P>"]


def assign_points_labels(nuclei, phones, words):
    """
    This function assigns phone and word labels, as well as phone boundaries and the corresponding durations.
    """

    phones_filtered = filter_labels(phones, "phones")
    words_filtered = filter_labels(words, "words")

    assigned_df = DataFrame(
        columns=["nucl_time", "phone", "word", "start_est", "end", "duration_est"]
    )

    assigned_df["nucl_time"] = nuclei

    for row in assigned_df.itertuples():
        match_phones = phones_filtered.loc[
            (phones_filtered["start_est"] <= row.nucl_time)
            & (phones_filtered["end"] >= row.nucl_time)
        ].to_numpy()
        match_words = words_filtered.loc[
            (words_filtered["start_est"] <= row.nucl_time)
            & (words_filtered["end"] >= row.nucl_time)
        ]["label"].array

        if match_phones.any():
            assigned_df.at[row.Index, ["end", "phone", "start_est"]] = match_phones[0]

        if match_words.any():
            assigned_df.at[row.Index, "word"] = match_words[0]

        assigned_df["duration_est"] = assigned_df["end"] - assigned_df["start_est"]
        assigned_df.round({"duration_est": 4})

    return assigned_df


def filter_labels(annotation_df, annotation_type):
    """
    This function filters dataframes containing word or phone labels, dropping rows containing non-vowel or extralinguistic labels, respectively.
    """

    if annotation_type == "phones":
        label_inv = SAMPA_VOWELS
        keep_if = True  # keep the rows where elements of SAMPA_VOWELS are found

    elif annotation_type == "words":
        label_inv = EXTRALING_SOUNDS
        keep_if = False  # drop the rows where elements of EXTRALING_SOUNDS are found

    else:
        raise ValueError("Annotation type '{}' not supported.".format(annotation_type))

    filtered_df = annotation_df.loc[(annotation_df["label"].isin(label_inv)) == keep_if]

    return filtered_df
