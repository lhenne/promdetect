import unittest
import pandas as pd
import numpy as np
from glob import glob

from promdetect.word_based import segmentation, extract_word_features


class WordSegmentationTests(unittest.TestCase):
    def test_include_annotations(self):
        tester = segmentation.Segmenter(
            "words", "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody"
        )

        tester.read_annotations()

        self.assertTrue("dlf-nachrichten-200703250000" in tester.annotations.keys())

    def test_output_type(self):
        tester = segmentation.Segmenter(
            "words", "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody"
        )

        tester.read_annotations()

        self.assertTrue(isinstance(tester.annotations, dict))
        self.assertTrue(
            isinstance(tester.annotations["dlf-nachrichten-200703250000"], pd.DataFrame)
        )

    def test_frame_calculation_output_format(self):
        tester = segmentation.Segmenter(
            "words", "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody"
        )

        tester.read_annotations()
        tester.add_frame_info()

        ex_df = tester.annotations["dlf-nachrichten-200703250000"]

        self.assertTrue("start_frame" in ex_df.columns)
        self.assertTrue("end_frame" in ex_df.columns)
        self.assertTrue("duration_frames" in ex_df.columns)

    def test_frame_calculation_output_values(self):
        tester = segmentation.Segmenter(
            "words", "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody"
        )

        tester.read_annotations()
        tester.add_frame_info()

        ex_df = tester.annotations["dlf-nachrichten-200703250000"]

        ex_vals = ex_df.loc[145]

        self.assertTrue(ex_vals["start_frame"] == (ex_vals["start"] * 48_000))
        self.assertTrue(ex_vals["end_frame"] == (ex_vals["end"] * 48_000))
        self.assertTrue(ex_vals["duration_frames"] == (ex_vals["duration"] * 48_000))

    def test_file_output(self):
        tester = segmentation.Segmenter(
            "words", "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody"
        )

        tester.read_annotations()
        tester.add_frame_info()
        tester.save_output()  # to default directory

        self.assertTrue(
            "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/dlf-nachrichten-200703250000_words.csv"
            in glob(
                "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/*.csv"
            )
        )
        self.assertTrue(
            "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/dlf-nachrichten-200703250000_words.npy"
            in glob(
                "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/*.npy"
            )
        )


class IPSegmentationTests(unittest.TestCase):
    def test_include_annotations(self):
        tester = segmentation.Segmenter(
            "tones", "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody"
        )

        tester.read_annotations()

        self.assertTrue("dlf-nachrichten-200703250000" in tester.annotations.keys())

    def test_output_type(self):
        tester = segmentation.Segmenter(
            "tones", "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody"
        )

        tester.read_annotations()

        self.assertTrue(isinstance(tester.annotations, dict))
        self.assertTrue(
            isinstance(tester.annotations["dlf-nachrichten-200703250000"], pd.DataFrame)
        )

    def test_frame_calculation_output_format(self):
        tester = segmentation.Segmenter(
            "tones", "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody"
        )

        tester.read_annotations()
        tester.add_frame_info()

        ex_df = tester.annotations["dlf-nachrichten-200703250000"]

        self.assertTrue("start_frame" in ex_df.columns)
        self.assertTrue("end_frame" in ex_df.columns)
        self.assertTrue("duration_frames" in ex_df.columns)

    def test_frame_calculation_output_values(self):
        tester = segmentation.Segmenter(
            "tones", "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody"
        )

        tester.read_annotations()
        tester.add_frame_info()

        ex_df = tester.annotations["dlf-nachrichten-200703250000"]

        ex_vals = ex_df.loc[42]

        self.assertTrue(ex_vals["start_frame"] == (ex_vals["start"] * 48_000))
        self.assertTrue(ex_vals["end_frame"] == (ex_vals["end"] * 48_000))
        self.assertTrue(ex_vals["duration_frames"] == (ex_vals["duration"] * 48_000))

    def test_file_output(self):
        tester = segmentation.Segmenter(
            "tones", "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody"
        )

        tester.read_annotations()
        tester.add_frame_info()
        tester.save_output()  # to default directory

        self.assertTrue(
            "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/dlf-nachrichten-200703250000_tones.csv"
            in glob(
                "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/*.csv"
            )
        )
        self.assertTrue(
            "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/dlf-nachrichten-200703250000_tones.npy"
            in glob(
                "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/*.npy"
            )
        )


class FeatureExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wav_file = (
            cls.wav_file
        ) = "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/dlf-nachrichten-200703250000.wav"
        cls.words = "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/dlf-nachrichten-200703250000_words.csv"
        cls.tones = "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/dlf-nachrichten-200703250000_tones.csv"
        cls.tester = extract_word_features.WordLevelExtractor(
            cls.wav_file, cls.words, cls.tones, "m"
        )

        cls.tester.get_duration_features()

    def test_duration_extraction(cls):

        expected_dur_vals = np.array([0.3799, 0.3199, 0.8299, 0.2799, 0.2599, 0.5199])
        true_dur_vals = np.around(
            cls.tester.features.iloc[12:18]["dur"].to_numpy(dtype=float), decimals=4
        )

        cls.assertTrue(np.array_equal(expected_dur_vals, true_dur_vals))

    def test_relative_duration_extraction(cls):

        expected_dur_normed_vals = np.array(
            [0.880281, 0.741253, 1.922994, 0.648567, 0.602224, 1.204681]
        )
        true_dur_normed_vals = np.around(
            cls.tester.features.iloc[12:18]["dur_normed"].to_numpy(dtype=float),
            decimals=6,
        )

        cls.assertTrue(np.array_equal(expected_dur_normed_vals, true_dur_normed_vals))
