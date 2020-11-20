from glob import glob
from pathlib import Path
from promdetect.prep.process_annotations import read_file, clean_text
from io import StringIO
import numpy as np
import pandas as pd


class Segmenter:
    def __init__(self, level, directory):
        self.level = level
        self.directory = directory
        self.files = glob(f"{directory}/*.{level}")

    def read_annotations(self):
        self.annotations = {}
        for file in self.files:
            recording = Path(file).stem
            raw_content = read_file(file)

            content = StringIO(clean_text(raw_content, self.level))

            content_as_df = self.content_to_df(content)

            content_inf_bounds = self.calc_start_end(content_as_df, self.level)

            self.annotations[recording] = content_inf_bounds

    def add_frame_info(self):
        for df in self.annotations.values():
            df["start_frame"] = df["start"] * 48_000
            df["end_frame"] = df["end"] * 48_000
            df["duration_frames"] = df["duration"] * 48_000

    def save_output(self, file):
        pass

    # ANCILLARY FUNCTIONS
    def content_to_df(self, content):
        content_as_df = pd.read_csv(
            content,
            sep=" ",
            engine="python",
            quoting=3,
            names=["end", "xwaves", "label"],
        ).drop(  # Drop unnecessary xwaves column
            ["xwaves"], axis=1
        )
        return content_as_df

    def calc_start_end(self, content_as_df, level):
        content_as_df["start"] = np.nan
        content_as_df["duration"] = np.nan
        num_rows = len(content_as_df.index)

        # Add data for estimated start timestamps of each word/int.phrase, which are 1ms after the end timestamp of the previous label (to avoid overlap for now). Set the starting time of the first label to N/A.
        for i in content_as_df.index:
            if i <= num_rows and i > 0:
                end_time = content_as_df.loc[i, "end"]
                start_time = content_as_df.loc[i - 1, "end"] + 0.0001

                duration = end_time - start_time

                if level == "words":
                    if duration > 3.0:
                        content_as_df.at[i, "start"] = end_time - 3.0
                        content_as_df.at[i, "duration"] = 3.0

                    else:
                        content_as_df.at[i, "start"] = start_time
                        content_as_df.at[i, "duration"] = end_time - start_time

                elif level == "tones":
                    content_as_df.at[i, "start"] = start_time
                    content_as_df.at[i, "duration"] = end_time - start_time
            else:
                continue

        return content_as_df
