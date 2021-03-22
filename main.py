from toWord import videoToWav,trans
from spleetertest import voice
from splitWord import wordSplit
from Bayes import Bayes_Predict
from BERT import BERT_Predict ,dataToCsv

videoName="test_vid.mp4"
fileName ="hello.wav"
videoToWav(videoName,fileName)
#print("Here!!")
savePath = "output"
voice(fileName,savePath)
audioPath = "./" + savePath + "/" + fileName[:-4] + "/vocals.wav"
print(audioPath)
setence = [trans(audioPath)]
#data_list = wordSplit(setence)
print(setence)
csvName = "BertData.csv"
dataToCsv([setence],csvName)
BERT_Predict(csvName)
Bayes_Predict([setence])


