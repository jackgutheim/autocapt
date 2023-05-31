from django.shortcuts import render
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer

import os
import whisper

from moviepy.editor import *
import whisper_timestamped as whisper

from django.views.decorators.csrf import csrf_exempt

# Create your views here.

@action(
        detail=True,
        methods=["POST"],
    )
@api_view(('POST',))
@renderer_classes((JSONRenderer,))
def transcribe(request):
    video = request.data['videofile']
    language = request.data['language']

    # create the temporary audio file
    type = "mp3"
    filename = video.name.split('.')[0]
    clip = VideoFileClip(video.temporary_file_path())
    clip.audio.write_audiofile(f"{filename}.{type}")

    # transcribe the audio file
    audio = whisper.load_audio(filename + ".mp3")
    model = whisper.load_model("base", device="cpu")
    result = whisper.transcribe(model, audio, language)

    # remove temporary .mp3 file
    os.remove("./" + filename + ".mp3")

    return Response(result)

