from toWord import videoToWav,trans
from spleetertest import voice
from splitWord import wordSplit
from Bayes import Bayes_Predict
from BERT import outcome ,dataToCsv
videoName="test_vid.mp4"
fileName = videoToWav(videoName)
print("Here!!")
savePath = "output"
voice(fileName,savePath)
audioPath = "./" + savePath + "/" + fileName[:-4] + "/vocals.wav"
print(audioPath)
setence = trans(audioPath)
data_list = wordSplit(setence)
#data_list = [['凹凸不平', '的', '狀態', '去', '打破', '的', '時候', '你', '的', '聲音', '波形', '就', '會', '非常', '不', '穩定', '你', '現在', '聽起來', '就', '是', '會', '是', '非常', '不', '乾淨', '的', '那麼', '假如', '你', '今天', '是', '像', '我', '一樣', '他', '是', '用', '嘴唇', '內側', '嘴唇', '內側', '去', '打', '的話', '他', '今天', '就', '會', '是', '非常', '穩定', '的', '波形', '因為', '你', '是', '用', '平面', '去', '吃', '一', '個', '聲音', '他', '就', '不會', '有', '不', '穩定', '的', '波形', '好率', '最', '差', '就', '會', '是']] 
print(data_list)
csvName = "BertData.csv"
dataToCsv(data_list,csvName)
outcome(csvName)
#Bayes_Predict(data_list)
#print("已將結果存到output.csv")

