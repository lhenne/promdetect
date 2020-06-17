"""
Code to test functions in the `prep` submodule
"""

import unittest
from promdetect.prep import (
    find_syllable_nuclei,
    get_speaker_info,
    prepare_data,
    process_annotations,
    process_audio,
)
from parselmouth import PraatError
import pandas as pd


class TestFindSyllableNuclei(unittest.TestCase):
    def test_that_fun_outputs_correct_data_types(self):
        output = find_syllable_nuclei.determine_nucleus_points(
            "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/dlf-nachrichten-200703251100.wav"
        )
        self.assertIsInstance(output, list)
        self.assertIsInstance(output[0], float)
        self.assertIsInstance(output[-1], float)

    def test_that_fun_fails_when_input_is_wrong_type(self):
        self.assertRaises(ValueError, find_syllable_nuclei.determine_nucleus_points, 42)

    def test_that_fun_fails_if_file_is_not_right_format(self):
        self.assertRaises(
            PraatError, find_syllable_nuclei.determine_nucleus_points, "lorem"
        )


class TestGetSpeakerInfo(unittest.TestCase):
    def test_that_funs_output_correct_data_type(self):
        self.assertIsInstance(get_speaker_info.get_gender("200703271500"), str)
        self.assertIsInstance(get_speaker_info.get_id("200703271300"), str)

    def test_that_funs_fail_when_input_is_wrong_type(self):
        self.assertRaises(TypeError, get_speaker_info.get_gender, [42, 63])
        self.assertRaises(TypeError, get_speaker_info.get_id, [52, 73])
        self.assertRaises(TypeError, get_speaker_info.get_gender, list)
        self.assertRaises(TypeError, get_speaker_info.get_id, str)

    def test_that_funs_fail_when_input_is_not_found(self):
        self.assertRaises(ValueError, get_speaker_info.get_gender, "20071234567")
        self.assertRaises(ValueError, get_speaker_info.get_id, "20079876543")


class TestPrepareData(unittest.TestCase):
    def test_that_metadata_is_generated_correctly(self):
        initializedInstance = prepare_data.DataPreparation(
            "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody",
            "200703271300",
        )
        self.assertEqual(
            initializedInstance.transcriptFile,
            "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/dlf-nachrichten-200703271300.words",
        )
        self.assertEqual(
            initializedInstance.tonesFile,
            "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/dlf-nachrichten-200703271300.tones",
        )
        self.assertEqual(
            initializedInstance.accentsFile,
            "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/dlf-nachrichten-200703271300.accents",
        )
        self.assertEqual(initializedInstance.speakerGender, "male")
        self.assertEqual(initializedInstance.speakerID, "7")

    def test_that_annotation_conversion_works(self):
        initializedInstance = prepare_data.DataPreparation(
            "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody",
            "200703271300",
        )
        initializedInstance.transform_annotations()

        self.assertIsInstance(initializedInstance.transcript, pd.DataFrame)
        self.assertIsInstance(initializedInstance.tones, pd.DataFrame)
        self.assertIsInstance(initializedInstance.accents, pd.DataFrame)

    def test_that_audio_processing_works(self):
        initializedInstance = prepare_data.DataPreparation(
            "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody",
            "200703271300",
        )
        initializedInstance.transform_annotations()
        initializedInstance.compute_audio_values()

        self.assertIsInstance(initializedInstance.transcript, pd.DataFrame)
        self.assertIsInstance(initializedInstance.tones, pd.DataFrame)
        self.assertIsInstance(initializedInstance.accents, pd.DataFrame)

    def test_that_funs_fail_when_input_is_not_found(self):
        with self.assertRaises(FileNotFoundError):
            prepare_data.DataPreparation("test", "string")
