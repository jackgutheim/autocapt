import os
import numpy
import whisper
import json
import datetime

from moviepy.editor import *
import whisper_timestamped as whisper

# Name of video to be processed
video_name = "autocapt/Michael_Owen"

# Number of words per caption section
caption_grouping = 4

# Number of seconds per caption chunk
caption_time = 1

# Caption Font Size
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

timer = 0

for i in range(len(segments)):
    words = segments[i]["words"]
    num_words = len(words)
    j = 0
    final_chunk = 0
    last_run = False

    while(j < num_words):
        remaining_time = caption_time
        text_string = ""


        first_word = words[j]
        start_time = first_word["start"]
        while remaining_time > 0 and j < num_words:
             curr_word = words[j]
             word_length = curr_word["end"] - curr_word["start"]
             
             text_string = text_string + str(curr_word["text"]) + " "
             remaining_time = remaining_time - word_length
             j += 1

        txt_clip = TextClip(text_string, fontsize=font_size, color="red")
        txt_clip = txt_clip.set_start(
            timer
             ).set_duration(
                caption_time
                  ).set_position(
                       ("center", "bottom")
                       )

        txt_clips.append(txt_clip)
        timer += caption_time

composite_list = []
composite_list.append(clip)
composite_list.extend(txt_clips)

video = CompositeVideoClip(
    composite_list
)

video.write_videofile(
    "autocapt/Output_File.mp4",
    codec="libx264",
    audio_codec='aac',
    bitrate=str(numpy.power(10, 7)),
    verbose=False,
    audio=True,
    ffmpeg_params=["-pix_fmt", "yuv420p"],
)