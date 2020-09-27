# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 22:05:58 2020

@author: SSTAN
https://www.linkedin.com/pulse/pythonda-sesli-asistan-olu%C5%9Fturmak-yunus-emre-g%C3%BCndo%C4%9Fmu%C5%9F/
"""
import speech_recognition as sr
import os
import time
from gtts import gTTS
 
audioString="merhaba"
tts = gTTS(text=audioString, lang='tr')
tts.save("audio.mp3")
os.system("audio.mp3")
