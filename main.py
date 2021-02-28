from toWord import videoToWav,trans
from spleetertest import voice
from splitWord import wordSplit
videoName="shortVideo.mp4"
fileName = videoToWav(videoName)
print("Here!!")
savePath = "output"
voice(fileName,savePath)
audioPath = "./" + savePath + "/" + fileName[:-4] + "/vocals.wav"
print(audioPath)
setence = trans(audioPath)
wordSplit(setence)


