import os
import numpy
import whisper
import json
import datetime

from moviepy.editor import *
import whisper_timestamped as whisper

# Change this based on which video you would like to process
video_name = "autocapt/Michael_Owen"

type = "mp3"
filename, ext = os.path.splitext(video_name + ".mp4")
clip = VideoFileClip(video_name + ".mp4")
clip.audio.write_audiofile(f"{filename}.{type}")

audio = whisper.load_audio(video_name + ".mp3")
model = whisper.load_model("base", device="cpu")
result = whisper.transcribe(model, audio, language="en")

segments = result["segments"]
words = segments[0]["words"]
print(words)
with open("autocapt/data.py", "w") as f:
    f.write(json.dumps(result))

txt_clips = []
for text in words:
    txt_clip = TextClip(text["text"], fontsize=70, color="red")
    txt_clip = (
        txt_clip.set_position((0.4,0.9), relative=True)
        .set_start(text["start"])
        .set_duration(text["end"] - text["start"])
    )
    txt_clips.append(txt_clip)

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