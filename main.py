from toWord import videoToWav,voaclToWord
from spleeter_vocal import wavToVocal
#from splitWord import wordSplit
from Bayes import Bayes_Predict
from BERT import BERT_Predict 

videoName="test_vid.mp4"
fileName ="hello.wav"
videoToWav(videoName,fileName)
#print("Here!!")
savePath = "output"
wavToVocal(fileName,savePath)
audioPath = "./" + savePath + "/" + fileName[:-4] + "/vocals.wav"
print(audioPath)
setence = [voaclToWord(audioPath)]
#data_list = wordSplit(setence)
#gprint(setence)
#csvName = "BertData.csv"
#dataToCsv([setence],csvName)
BERT_Predict(setence)
Bayes_Predict(setence)


