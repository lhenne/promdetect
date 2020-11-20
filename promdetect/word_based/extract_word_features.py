"""
This script contains all preparatory extraction functions for the acoustic parameters that will later serve as the input for the neural network.
Import necessary packages:
`parselmouth` to send commands to the Praat phonetics software
`numpy` to iterate with high performance
`pandas` to manage data
"""

import numpy as np
import pandas as pd
import parselmouth as pm
from parselmouth import praat


class WordLevelExtractor:
    def __init__(self, wav_file, words, nuclei="", gender="f", ip=None):
        self.wav_file = wav_file
        self.words = words
        self.snd_obj = pm.Sound(self.wav_file)
        self.gender = gender

        if gender == "f":
            self.__pitch_range = (75, 500)
        else:
            self.__pitch_range = (50, 300)

    def get_duration_features(self):
        """
        Get duration features:
        - estimated duration from word start and end timestamps
        """

    def get_intensity_features(self):
        """
        Get intensity features:
        - minimum intensity
        - maximum intensity
        - mean intensity
        - intensity std
        - minimum intensity relative position
        - maximum intensity relative position
        """

    def get_pitch_features(self):
        """
        Get pitch features:
        - minimum pitch
        - maximum pitch
        - mean pitch
        - pitch slope
        - pitch excursion
        - minimum pitch relative position
        - maximum pitch relative position
        """

    def get_spectral_features(self):
        """
        Get spectral features:
        - mean spectral tilt (C1)
        - spectral tilt range (C1)
        - spectral centre of gravity
        """
