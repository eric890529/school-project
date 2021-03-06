from spleeter.separator import Separator

from spleeter.audio.adapter import AudioAdapter
# Using embedded configuration.
def voice(fName,sPath):
    fileName = fName
    savePath = sPath
    separator = Separator('spleeter:2stems')#選模式 要下載預訓練模型
    separator.separate_to_file(fileName, savePath)#音檔路徑,指定的存檔位置
    print("hello")

