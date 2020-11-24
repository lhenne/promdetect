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


class DurationExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wav_file = "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/dlf-nachrichten-200703250000.wav"
        cls.words = "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/dlf-nachrichten-200703250000_words.csv"
        cls.tones = "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/dlf-nachrichten-200703250000_tones.csv"
        cls.tester = extract_word_features.WordLevelExtractor(
            cls.wav_file, cls.words, cls.tones, "m"
        )

        cls.tester.get_duration_features()

    def test_duration_extraction(cls):
        expected_dur_vals = np.array([0.3799, 0.3199, 0.8299, 0.2799, 0.2599, 0.5199])
        true_dur_vals = np.around(
            cls.tester.features.iloc[25:31]["dur"].to_numpy(dtype=float), decimals=4
        )

        cls.assertTrue(np.array_equal(expected_dur_vals, true_dur_vals))

    def test_relative_duration_extraction(cls):
        expected_dur_normed_vals = np.array(
            [0.880281, 0.741253, 1.922994, 0.648567, 0.602224, 1.204681]
        )
        true_dur_normed_vals = np.around(
            cls.tester.features.iloc[25:31]["dur_normed"].to_numpy(dtype=float),
            decimals=6,
        )

        cls.assertTrue(np.array_equal(expected_dur_normed_vals, true_dur_normed_vals))


class IntensityExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wav_file = "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/dlf-nachrichten-200703250000.wav"
        cls.words = "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/dlf-nachrichten-200703250000_words.csv"
        cls.tones = "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/dlf-nachrichten-200703250000_tones.csv"
        cls.tester = extract_word_features.WordLevelExtractor(
            cls.wav_file, cls.words, cls.tones, "m"
        )

        cls.tester.get_intensity_features()

    def test_rms_extraction(cls):
        expected_rms_vals = np.around(
            [
                0.10161370663028274,
                0.14051771492058313,
                0.09366884288212228,
                0.11640538869693093,
                0.09683181167771422,
                0.11143758362879955,
            ],
            decimals=6,
        )
        true_rms_vals = np.around(
            cls.tester.features.iloc[25:31]["int_rms"].to_numpy(dtype=float), decimals=6
        )

        cls.assertTrue(np.array_equal(expected_rms_vals, true_rms_vals))

    def test_min_intensity_extraction(cls):
        expected_min_int_vals = np.around(
            [
                48.015972386392306,
                49.904039160722874,
                55.55279568485452,
                65.56535790930751,
                54.14122780455792,
                54.41637949569892,
            ],
            decimals=4,
        )
        true_min_int_vals = np.around(
            cls.tester.features.iloc[25:31]["int_min"].to_numpy(dtype=float), decimals=4
        )

        cls.assertTrue(np.array_equal(expected_min_int_vals, true_min_int_vals))

    def test_max_intensity_extraction(cls):
        expected_max_int_vals = np.around(
            [
                81.16198677156079,
                80.36040661525334,
                77.9641258108993,
                77.95985630263552,
                78.52793346787278,
                79.25344766573109,
            ],
            decimals=4,
        )
        true_max_int_vals = np.around(
            cls.tester.features.iloc[25:31]["int_max"].to_numpy(dtype=float), decimals=4
        )

        cls.assertTrue(np.array_equal(expected_max_int_vals, true_max_int_vals))

    def test_mean_intensity_extraction(cls):
        expected_mean_int_vals = np.around(
            [
                74.11840410522626,
                76.94435348598887,
                73.37698372634644,
                75.36038967466746,
                73.61321071679221,
                74.94396199363445,
            ],
            decimals=4,
        )
        true_mean_int_vals = np.around(
            cls.tester.features.iloc[25:31]["int_mean"].to_numpy(dtype=float),
            decimals=4,
        )

        cls.assertTrue(np.array_equal(expected_mean_int_vals, true_mean_int_vals))

    def test_intensity_std_extraction(cls):
        expected_int_std_vals = np.around(
            [
                10.815229401716747,
                9.601092146508785,
                4.822591533323357,
                4.382206242422594,
                7.943270723266582,
                7.934130792564047,
            ],
            decimals=4,
        )
        true_int_std_vals = np.around(
            cls.tester.features.iloc[25:31]["int_std"].to_numpy(dtype=float),
            decimals=4,
        )

        cls.assertTrue(np.array_equal(expected_int_std_vals, true_int_std_vals))

    def test_min_intensity_position_extraction(cls):
        expected_int_min_pos_vals = np.around(
            [
                0.09951632008421196,
                0.9309354485776591,
                0.6094785516327182,
                0.09934351554124295,
                0.5687043093497132,
                0.6766806116560727,
            ],
            decimals=4,
        )
        true_int_min_pos_vals = np.around(
            cls.tester.features.iloc[25:31]["int_min_pos"].to_numpy(dtype=float),
            decimals=4,
        )

        cls.assertTrue(np.array_equal(expected_int_min_pos_vals, true_int_min_pos_vals))

    def test_max_intensity_position_extraction(cls):
        expected_int_max_pos_vals = np.around(
            [
                0.5627961305606498,
                0.28073226008124885,
                0.6865962766598303,
                0.3851598785280165,
                0.07620719507499511,
                0.830556356991711,
            ],
            decimals=4,
        )
        true_int_max_pos_vals = np.around(
            cls.tester.features.iloc[25:31]["int_max_pos"].to_numpy(dtype=float),
            decimals=4,
        )

        cls.assertTrue(np.array_equal(expected_int_max_pos_vals, true_int_max_pos_vals))


class PitchExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wav_file = "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/dlf-nachrichten-200703250000.wav"
        cls.words = "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/dlf-nachrichten-200703250000_words.csv"
        cls.tones = "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based/dlf-nachrichten-200703250000_tones.csv"
        cls.tester = extract_word_features.WordLevelExtractor(
            cls.wav_file, cls.words, cls.tones, "m"
        )

        cls.tester.get_pitch_features()

    def test_f0_min_extraction(cls):
        expected_min_f0_vals = np.around(
            [
                93.42051927019007,
                119.74223083992594,
                66.28219915326169,
                60.263752559173824,
                76.74685352476915,
                69.66641648219964,
            ],
            decimals=4,
        )
        true_min_f0_vals = np.around(
            cls.tester.features.iloc[25:31]["f0_min"].to_numpy(dtype=float), decimals=4,
        )

        cls.assertTrue(np.array_equal(expected_min_f0_vals, true_min_f0_vals))

    def test_f0_max_extraction(cls):
        expected_max_f0_vals = np.around(
            [
                175.8310095151027,
                148.73301023639186,
                110.13415499332966,
                79.85128984114829,
                83.28485978576516,
                122.86386702663233,
            ],
            decimals=4,
        )
        true_max_f0_vals = np.around(
            cls.tester.features.iloc[25:31]["f0_max"].to_numpy(dtype=float), decimals=4,
        )

        cls.assertTrue(np.array_equal(expected_max_f0_vals, true_max_f0_vals))

    def test_f0_mean_extraction(cls):
        expected_mean_f0_vals = np.around(
            [
                144.19247747891387,
                141.24714063080444,
                82.45323698156723,
                71.93538145114684,
                79.82018690856482,
                89.27793109999983,
            ],
            decimals=4,
        )
        true_mean_f0_vals = np.around(
            cls.tester.features.iloc[25:31]["f0_mean"].to_numpy(dtype=float),
            decimals=4,
        )

        cls.assertTrue(np.array_equal(expected_mean_f0_vals, true_mean_f0_vals))

    def test_f0_std_extraction(cls):
        expected_f0_std_vals = np.around(
            [
                29.39623269825003,
                6.204980285370182,
                8.502563974126964,
                5.771182232504161,
                2.430694437462275,
                11.699726532753346,
            ],
            decimals=4,
        )
        true_f0_std_vals = np.around(
            cls.tester.features.iloc[25:31]["f0_std"].to_numpy(dtype=float), decimals=4,
        )

        cls.assertTrue(np.array_equal(expected_f0_std_vals, true_f0_std_vals))

    def test_f0_slope_extraction(cls):
        expected_f0_slope_vals = np.around(
            [
                82.29572142068675,
                35.19382999003871,
                39.75845759512797,
                46.03455041025139,
                38.084437731771985,
                65.2757023779267,
            ],
            decimals=4,
        )
        true_f0_slope_vals = np.around(
            cls.tester.features.iloc[25:31]["f0_slope"].to_numpy(dtype=float),
            decimals=4,
        )

        cls.assertTrue(np.array_equal(expected_f0_slope_vals, true_f0_slope_vals))

    def test_f0_excursion_ip_extraction(cls):
        expected_excursion_ip_vals = np.around(
            [
                12 * np.log2(175.8310095151027 / 71.78073529645441),
                12 * np.log2(148.73301023639186 / 71.78073529645441),
                12 * np.log2(110.13415499332966 / 71.78073529645441),
                12 * np.log2(79.85128984114829 / 71.78073529645441),
                12 * np.log2(83.28485978576516 / 71.78073529645441),
                12 * np.log2(122.86386702663233 / 71.78073529645441),
            ],
            decimals=4,
        )
        true_excursion_ip_vals = np.around(
            cls.tester.features.iloc[25:31]["f0_exc_ip"].to_numpy(dtype=float),
            decimals=4,
        )

        cls.assertTrue(
            np.array_equal(expected_excursion_ip_vals, true_excursion_ip_vals)
        )

    def test_f0_excursion_utterance_extraction(cls):
        expected_excursion_utt_vals = np.around(
            [
                12 * np.log2(175.8310095151027 / 73.27927882944502),
                12 * np.log2(148.73301023639186 / 73.27927882944502),
                12 * np.log2(110.13415499332966 / 73.27927882944502),
                12 * np.log2(79.85128984114829 / 73.27927882944502),
                12 * np.log2(83.28485978576516 / 73.27927882944502),
                12 * np.log2(122.86386702663233 / 73.27927882944502),
            ],
            decimals=4,
        )
        true_excursion_utt_vals = np.around(
            cls.tester.features.iloc[25:31]["f0_exc_utt"].to_numpy(dtype=float),
            decimals=4,
        )

        cls.assertTrue(
            np.array_equal(expected_excursion_utt_vals, true_excursion_utt_vals)
        )

    def test_f0_min_pos_extraction(cls):
        expected_f0_min_pos_vals = np.around(
            [
                0.3982791524084948,
                0.629278680837722,
                0.7486519460175809,
                0.6298901393354346,
                0.005025971527465514,
                0.31988122715904505,
            ],
            decimals=4,
        )
        true_f0_min_pos_vals = np.around(
            cls.tester.features.iloc[25:31]["f0_min_pos"].to_numpy(dtype=float),
            decimals=4,
        )

        cls.assertTrue(np.array_equal(expected_f0_min_pos_vals, true_f0_min_pos_vals))

    def test_f0_max_pos_vals(cls):
        expected_f0_max_pos_vals = np.around(
            [
                0.700990392208448,
                0.1603821506720504,
                0.6703292565369189,
                0.701344230082128,
                0.19740765679102984,
                0.16600548182339994,
            ],
            decimals=4,
        )
        true_f0_max_pos_vals = np.around(
            cls.tester.features.iloc[25:31]["f0_max_pos"].to_numpy(dtype=float),
            decimals=4,
        )

        cls.assertTrue(np.array_equal(expected_f0_max_pos_vals, true_f0_max_pos_vals))
