import unittest
import pandas as pd
from glob import glob

from promdetect.word_based import segmentation


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
