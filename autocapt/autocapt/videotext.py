import os
import numpy
import whisper
import json
import datetime

from moviepy.editor import *
import whisper_timestamped as whisper

# Change this based on which video you would like to process
video_name = "autocapt/Michael_Owen"

# How many words you want per caption section
caption_grouping = 3

type = "mp3"
filename, ext = os.path.splitext(video_name + ".mp4")
clip = VideoFileClip(video_name + ".mp4")
clip.audio.write_audiofile(f"{filename}.{type}")

audio = whisper.load_audio(video_name + ".mp3")
model = whisper.load_model("base", device="cpu")
result = whisper.transcribe(model, audio, language="en")


with open("autocapt/data.py", "w") as f:
    f.write(json.dumps(result))

segments = result["segments"]
txt_clips = []

for i in range(len(segments)):
    words = segments[i]["words"]
    j = 0
    while(j < len(words)):

        text_string = ""
        last_few = 0
        last_run = False

        if (j + caption_grouping) > len(words):
            last_few = len(words) - j
            last_run = True
            for x in range(last_few):
                text_string = text_string + str(words[j + x]["text"]) + " "
        else:
            for x in range(caption_grouping):
                text_string = text_string + str(words[j + x]["text"]) + " "

        txt_clip = TextClip(text_string, fontsize=70, color="red")
        if last_run:
            txt_clip = (
            txt_clip.set_position((0.4,0.8), relative=True)
            .set_start(words[j]["start"])
            .set_duration(words[j + (last_few - 1)]["end"] - words[j]["start"])
        )
        else:
            txt_clip = (
            txt_clip.set_position((0.4,0.8), relative=True)
            .set_start(words[j]["start"])
            .set_duration(words[j + (caption_grouping - 1)]["end"] - words[j]["start"])
        )
        txt_clips.append(txt_clip)
        if (j + caption_grouping) <= len(words):
            j += caption_grouping
        if last_run:
            break

composite_list = []
composite_list.append(clip)
composite_list.extend(txt_clips)

video = CompositeVideoClip(
    composite_list
)

video.write_videofile(
    "autocapt/Output_File.mp4",
    codec="libx264",
    bitrate=str(numpy.power(10, 7)),
    verbose=False,
    audio=True,
    ffmpeg_params=["-pix_fmt", "yuv420p"],
)