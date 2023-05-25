import os
import numpy
import whisper
import json
import datetime

from moviepy.editor import *
import whisper_timestamped as whisper

# Change this based on which video you would like to process
video_name = "autocapt/macdemarco"

# How many words you want per caption section
caption_grouping = 4

# Font Size
font_size = 50

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
    last_run = False
    num_words = len(words)
    while(j < num_words):

        text_string = ""
        last_few = num_words - j
        
        if (j + caption_grouping) > num_words:
            last_run = True

        for x in range(last_few if last_run else caption_grouping):
                text_string = text_string + str(words[j + x]["text"]) + " "

        txt_clip = TextClip(text_string, fontsize=font_size, color="red")
        txt_clip = (
            txt_clip
            .set_start(words[j]["start"])
            .set_duration(words[j + ((last_few if last_run else caption_grouping) - 1)]["end"] - words[j]["start"])
        ).set_position(("center", "bottom"))
        txt_clips.append(txt_clip)
        if (j + caption_grouping) <= num_words:
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