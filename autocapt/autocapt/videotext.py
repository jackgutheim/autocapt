import os
import numpy
import whisper
import json
import datetime

from moviepy.editor import *
import whisper_timestamped as whisper

# Change this based on which video you would like to process
video_name = "src/Michael_Owen"

type = "mp3"
filename, ext = os.path.splitext(video_name + ".mp4")
clip = VideoFileClip(video_name + ".mp4")
clip.audio.write_audiofile(f"{filename}.{type}")

audio = whisper.load_audio(video_name + "mp3")
model = whisper.load_model("base", device="cpu")
result = whisper.transcribe(model, audio, language="en")

segments = result["segments"]
words = segments[0]["words"]
print(words)
with open("src/data.py", "w") as f:
    f.write(json.dumps(result))

txt_clips = []
for text in words:
    txt_clip = TextClip(text["text"], fontsize=70, color="red")
    txt_clip = (
        txt_clip.set_position((500, 750))
        .set_start(text["start"])
        .set_duration(text["end"] - text["start"])
    )
    txt_clips.append(txt_clip)

video = CompositeVideoClip(
    [
        clip,
        txt_clips[0],
        txt_clips[1],
        txt_clips[2],
        txt_clips[3],
        txt_clips[4],
        txt_clips[5],
        txt_clips[6],
        txt_clips[7],
        txt_clips[8],
        txt_clips[9],
        txt_clips[10],
        txt_clips[11],
        txt_clips[12],
        txt_clips[13],
        txt_clips[14],
        txt_clips[15],
        txt_clips[16],
        txt_clips[17],
        txt_clips[18],
        txt_clips[19],
        txt_clips[20],
        txt_clips[21],
        txt_clips[22],
        txt_clips[23],
        txt_clips[24],
        txt_clips[25],
        txt_clips[26],
        txt_clips[27],
        txt_clips[28],
        txt_clips[29],
        txt_clips[30],
    ]
)

video.write_videofile(
    "src/Output_File.mp4",
    codec="libx264",
    bitrate=str(numpy.power(10, 7)),
    verbose=False,
    audio=False,
    ffmpeg_params=["-pix_fmt", "yuv420p"],
)