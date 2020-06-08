"""
This script detect syllable nuclei in a sound file
and returns their timestamps as annotations.

It is a loose translation of a Praat script by Nivja de Jong and Ton Wempe,
and takes aspects from David Feinberg's Python translation of that script.
"""

import parselmouth as pm
from parselmouth.praat import call

silenceThreshold = -25
minDipBetweenPeaks = 2
minPauseDuration = 0.3

soundObj = pm.Sound("/home/lukas/Dokumente/Uni/ma_thesis/\
                        quelldaten/DIRNDL-prosody/\
                        dlf-nachrichten-200703271500.wav")
intensityObj = soundObj.to_intensity(minimum_pitch=50)

totalDuration = soundObj.get_total_duration()
minIntensity = intensityObj.get_minimum()
maxIntensity = intensityObj.get_maximum()
max99Intensity = call(intensityObj, "Get quantile", 0, 0, 0.99)

threshold1 = max99Intensity + silenceThreshold
threshold2 = maxIntensity - max99Intensity
threshold3 = silenceThreshold - threshold2


