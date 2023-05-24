import os
import cv2
import numpy
import whisper
import json
import datetime

from moviepy.editor import *
import whisper_timestamped as whisper

type = "mp3"
filename, ext = os.path.splitext(".venv/Test_Movie_Audio.mp4")
clip = VideoFileClip(".venv/Test_Movie_Audio.mp4")
clip.audio.write_audiofile(f"{filename}.{type}")

# start_time = datetime.datetime.now()
# model = whisper.load_model("base")
# result = model.transcribe(".venv/Test_Movie_Audio.mp3")
# end_time = datetime.datetime.now()

# ///
audio = whisper.load_audio(".venv/Test_Movie_Audio.mp3")
model = whisper.load_model("base", device="cpu")
result = whisper.transcribe(model, audio, language="en")

# print(json.dumps(result1, indent=2, ensure_ascii=False))


# print({"start_time": start_time})
# print({"end_time": end_time})
# print({"diff": (end_time - start_time)})

segments = result["segments"]
words = segments[0]["words"]
print(words)
with open("data.py", "w") as f:
    f.write(json.dumps(result))

txt_clips = []
for text in words:
    txt_clip = TextClip(text["text"], fontsize=50, color="blue")
    txt_clip = (
        txt_clip.set_position((100, 900))
        .set_start(text["start"])
        .set_duration(text["end"] - text["start"])
    )
    txt_clips.append(txt_clip)


# clip = VideoFileClip(".venv/Test_Movie_Audio.mp4")

# Generate a text clip
# txt_clip = TextClip(result["text"], fontsize=20, color="blue")

# setting position of text in the center and duration will be 10 seconds
# txt_clip = txt_clip.set_position((0.5, 900)).set_duration(1)


# Overlay the text clip on the first video clip
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
    ]
)

# showing video
video.write_videofile(
    "New_Test_Movie_Audio.mp4",
    codec="libx264",
    bitrate=str(numpy.power(10, 7)),
    verbose=False,
    audio=False,
    ffmpeg_params=["-pix_fmt", "yuv420p"],
)


# cap = cv2.VideoCapture(".venv/Test_Movie_Audio.mp4")

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break
#     frame = cv2.resize(frame, (600, 400))
#     cv2.putText(
#         frame,
#         result["text"],
#         (0, 350),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.7,
#         (255, 50, 50),
#         3,
#     )
#     cv2.imshow("vid", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
