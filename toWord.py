import  moviepy.editor
import speech_recognition as sr
def videoToWav(vName,fName):
    videoName = vName
    fileName = fName
    video = moviepy.editor.VideoFileClip(videoName)
    audio= video.audio
    audio.write_audiofile(fileName)
    #return fileName

def trans(aPath):
    audioPath = aPath
    r = sr.Recognizer() 
    with sr.WavFile(audioPath) as source:  #讀取wav檔
        audio = r.record(source)
    try:
        print("Transcription: " + r.recognize_google(audio,language="zh-TW"))
                                            #使用Google的服務
    except LookupError:
        print("Could not understand audio")
    return  r.recognize_google(audio,language="zh-TW")
        
'''
r=sr.Recognizer()
    
with sr.Microphone() as source:
    print("Please wait. Calibrating microphone...")
    #listen for 5 seconds and create the ambient noise energy level
    r.adjust_for_ambient_noise(source, duration=5)
    print("Say something!")
    audio=r.listen(source)'''
'''
# recognize speech using Google Speech Recognition
try:
    print("Google Speech Recognition thinks you said:")
    print(r.recognize_google(audio, language="zh-TW"))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("No response from Google Speech Recognition service: {0}".format(e))'''

