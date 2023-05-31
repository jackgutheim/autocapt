from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer

import os
import numpy
import whisper
import json
import datetime

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

    type = "mp3"
    filename = "Michael_Owen"
    clip = VideoFileClip(video.temporary_file_path())
    clip.audio.write_audiofile(f"{filename}.{type}")

    audio = whisper.load_audio(filename + ".mp3")
    model = whisper.load_model("base", device="cpu")
    result = whisper.transcribe(model, audio, language)

    os.remove("./" + filename + ".mp3")

    return Response(result)

