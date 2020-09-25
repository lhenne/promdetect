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


def filter_labels(annotation_df, annotation_type):
    """
    This function filters dataframes containing word or phone labels, dropping rows containing non-vowel or extralinguistic labels, respectively.
    """

    if annotation_type == "phones":
        label_inv = SAMPA_VOWELS
        keep_if = True

    elif annotation_type == "words":
        label_inv = EXTRALING_SOUNDS
        keep_if = False

    else:
        raise ValueError("Annotation type '{}' not supported.".format(annotation_type))

    filtered_df = annotation_df.loc[(annotation_df["label"].isin(label_inv)) == keep_if]

    return filtered_df
