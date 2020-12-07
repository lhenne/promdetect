import pandas as pd
import parselmouth as pm
from parselmouth import praat
from pathlib import Path
from glob import glob

SPEAKERS = {
    "dlf-nachrichten-200703250000": "m",
    "dlf-nachrichten-200703250100": "m",
    "dlf-nachrichten-200703250300": "m",
    "dlf-nachrichten-200703250400": "m",
    "dlf-nachrichten-200703250500": "m",
    "dlf-nachrichten-200703250600": "f",
    "dlf-nachrichten-200703250700": "f",
    "dlf-nachrichten-200703250800": "f",
    "dlf-nachrichten-200703250900": "f",
    "dlf-nachrichten-200703251000": "f",
    "dlf-nachrichten-200703251100": "f",
    "dlf-nachrichten-200703251200": "f",
    "dlf-nachrichten-200703251300": "f",
    "dlf-nachrichten-200703251400": "f",
    "dlf-nachrichten-200703251500": "f",
    "dlf-nachrichten-200703260000": "m",
    "dlf-nachrichten-200703260100": "m",
    "dlf-nachrichten-200703260200": "m",
    "dlf-nachrichten-200703260300": "m",
    "dlf-nachrichten-200703260400": "m",
    "dlf-nachrichten-200703260500": "m",
    "dlf-nachrichten-200703260600": "f",
    "dlf-nachrichten-200703260700": "f",
    "dlf-nachrichten-200703260800": "f",
    "dlf-nachrichten-200703260900": "f",
    "dlf-nachrichten-200703261000": "f",
    "dlf-nachrichten-200703261100": "f",
    "dlf-nachrichten-200703261200": "f",
    "dlf-nachrichten-200703261300": "f",
    "dlf-nachrichten-200703261400": "f",
    "dlf-nachrichten-200703261500": "f",
    "dlf-nachrichten-200703261600": "f",
    "dlf-nachrichten-200703261700": "f",
    "dlf-nachrichten-200703261800": "m",
    "dlf-nachrichten-200703261900": "m",
    "dlf-nachrichten-200703262000": "m",
    "dlf-nachrichten-200703262100": "m",
    "dlf-nachrichten-200703262200": "m",
    "dlf-nachrichten-200703262300": "m",
    "dlf-nachrichten-200703270000": "m",
    "dlf-nachrichten-200703270100": "m",
    "dlf-nachrichten-200703270200": "m",
    "dlf-nachrichten-200703270300": "m",
    "dlf-nachrichten-200703270400": "m",
    "dlf-nachrichten-200703270500": "m",
    "dlf-nachrichten-200703270600": "m",
    "dlf-nachrichten-200703270700": "m",
    "dlf-nachrichten-200703270800": "m",
    "dlf-nachrichten-200703270900": "m",
    "dlf-nachrichten-200703271000": "m",
    "dlf-nachrichten-200703271100": "m",
    "dlf-nachrichten-200703271200": "m",
    "dlf-nachrichten-200703271300": "m",
    "dlf-nachrichten-200703271400": "m",
    "dlf-nachrichten-200703271500": "m",
}


class FrameLevelExtractor:
    def __init__(self, wav_file, gender="f") -> None:
        self.wav_file = wav_file
        self.snd_obj = pm.Sound(self.wav_file)
        self.gender = gender
        self.features = pd.DataFrame()
        self.path = f"features/{wav_file}.frames"

        if gender == "f":
            self.__pitch_range = (75, 500)
        else:
            self.__pitch_range = (50, 300)

        self.TIME_STEP = 0.01

        self.pitch_obj = self.snd_obj.to_pitch_cc(
            time_step=self.TIME_STEP,
            pitch_floor=self.__pitch_range[0],
            pitch_ceiling=self.__pitch_range[1],
        )

        self.pitch_extraction()

    def pitch_extraction(self):
        timestamps = self.pitch_obj.ts()

        pitch_cands = self.pitch_obj.to_array().T
        best_cands = [max(frame, key=lambda x: x[1]) for frame in pitch_cands]
        f0, voicing_pr = zip(
            *best_cands
        )  # candidate strength as replacement for voicing probability, will be normalized later so high values are not problematic

        cols_to_add = ["time", "f0", "voicing_pr"]
        vals_to_add = zip(timestamps, f0, voicing_pr)

        data_to_add = pd.DataFrame(vals_to_add, columns=cols_to_add)
        self.features = pd.concat([self.features, data_to_add])

    def rms_extraction(self):
        self.features["rms"] = [
            self.snd_obj.get_rms(
                from_time=frame.time, to_time=(frame.time + self.TIME_STEP)
            )
            for frame in self.features.itertuples()
        ]

    def loudness_extraction(self):
        self.cochleagram = praat.call(
            self.snd_obj, "To Cochleagram", 0.01, 0.1, 0.03, 0.03
        )
        self.features["excitation"] = [
            praat.call(self.cochleagram, "To Excitation (slice)", frame.time)
            for frame in self.features.itertuples()
        ]

        self.features["loudness"] = [
            praat.call(frame.excitation, "Get loudness")
            for frame in self.features.itertuples()
        ]

        self.features = self.features.drop(columns="excitation")

    def zcr_extraction(self):
        self.features["snd_part"] = [
            self.snd_obj.extract_part(from_time=frame.time, to_time=frame.time + 0.01)
            for frame in self.features.itertuples()
        ]

        self.features["pp_part"] = [
            praat.call(frame.snd_part, "To PointProcess (zeroes)", 1, "yes", "yes")
            for frame in self.features.itertuples()
        ]

        self.features["zcr"] = [
            praat.call(frame.pp_part, "Get number of points")
            for frame in self.features.itertuples()
        ]

        self.features = self.features.drop(columns=["snd_part", "pp_part"])

    def hnr_extraction(self):
        self.harm_obj = self.snd_obj.to_harmonicity_cc(
            minimum_pitch=self.__pitch_range[0]
        )

        self.features["hnr"] = [
            praat.call(self.harm_obj, "Get value in frame", frame.Index)
            for frame in self.features.itertuples()
        ]

    def write_features(self):
        self.features.to_csv(self.path)


for wav in glob("*.wav"):
    recording = Path(wav).stem
    gender = SPEAKERS[recording]

    extractor = FrameLevelExtractor(wav, gender)
    extractor.rms_extraction()
    extractor.loudness_extraction()
    extractor.zcr_extraction()
    extractor.hnr_extraction()
    extractor.write_features()
