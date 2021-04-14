import pandas as pd # 引用套件並縮寫為 pd  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import csv
import io  
def Bayes_Predict(data_list):
    df = pd.read_csv('data.csv')  
    df.columns = ['text', 'label']
    #print(df)
    # 初始化vectorizer, 使用的是bag-of-word 最基礎的 CountVectorizer
    vectorizer = CountVectorizer(analyzer='char')#以一個字一個字去訓練

    # 將 text 轉換成 bow 格式
    text = vectorizer.fit_transform(df['text'])

    # 實例化(Instantiate) 這個 Naive Bayes Classifier
    MNB_model = MultinomialNB()

    # 把資料給它，讓他根據貝氏定理，去算那些機率。
    MNB_model.fit(text, df['label'])

    #清除上一筆資料
    '''
    with open('word.csv', 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            # 寫入一列資料
            writer.writerow("")
    with open('output.csv', 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            # 寫入一列資料
            writer.writerow("")
    '''
    '''
    for i in range(len(data_list[0])):
        with open('word.csv', 'a', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            # 寫入一列資料
            writer.writerow([data_list[0][i]])
            #writer.writerow(["髒話測試"])
    '''
    df=pd.DataFrame(data_list,columns=["text"])
    data = df['text']
    
    #line = pd.read_csv('word.csv',header=None)
    #line.columns = ['text']
    word = vectorizer.transform(data)
    '''
    for i in range(len(data_list[0])):
        with open('output.csv', 'a', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            writer.writerow([line['text'][i],MNB_model.predict(word[i])])
    '''
    for i in range(len(data_list)):
        print("Bayes_Predict:",MNB_model.predict(word[i]))
#print("first:",line['text'][0])
    #print("line:",len(line['text']))
    #print("data_list:",len(data_list[0]))


 #print("word = ",line.loc[i])
        #print(word[i])
        #print(line['text'][i])
        #print(MNB_model.predict(word[i]))
'''
   #測試向量
    dff = pd.read_csv(  'testdata.csv',header=None)
    print ( len(dff) )
    dff.columns = ['text']
    word2 = vectorizer.transform(dff['text'])
    for i in range(len(dff)):
        #print("word = ",line.loc[i])
        print(word2[i])
        print(dff['text'][i])
        print(MNB_model.predict(word2[i]))
'''    
    


