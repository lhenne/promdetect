"""
Code to test functions in the `prep` submodule
"""

import unittest
import promdetect.prep.find_syllable_nuclei as syllable_nuclei
import promdetect.prep.get_speaker_info as speaker_info
from parselmouth import PraatError


class TestFindSyllableNuclei(unittest.TestCase):

    def test_that_fun_outputs_correct_data_types(self):
        output = syllable_nuclei.determine_nucleus_points("/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/dlf-nachrichten-200703251100.wav")
        self.assertIsInstance(output, list)
        self.assertIsInstance(output[0], float)
        self.assertIsInstance(output[-1], float)

    def test_that_fun_fails_when_input_is_wrong_type(self):
        self.assertRaises(ValueError, syllable_nuclei.determine_nucleus_points, 42)

    def test_that_fun_fails_if_file_is_not_right_format(self):
        self.assertRaises(PraatError, syllable_nuclei.determine_nucleus_points, "lorem")
