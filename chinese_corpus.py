import pymongo
import zhconv
from tqdm import tqdm


myclient = pymongo.MongoClient("mongodb://admin:research21@ec768aa3-bb97-421c-98bd-4822fa7dbea1-0.b8267831955f40ef8fb5530d280e5a10.databases.appdomain.cloud:32481,ec768aa3-bb97-421c-98bd-4822fa7dbea1-1.b8267831955f40ef8fb5530d280e5a10.databases.appdomain.cloud:32481,ec768aa3-bb97-421c-98bd-4822fa7dbea1-2.b8267831955f40ef8fb5530d280e5a10.databases.appdomain.cloud:32481/ibmclouddb?authSource=admin&replicaSet=replset",
                                    ssl=True, ssl_ca_certs="./mongo_cacert_file")


# download data and save into txt, including 10 sentences each line

Corpus_collection = myclient['Chinese_Corpus']['HKFinancialStatements']

data = []
for record in tqdm(Corpus_collection.find()):
    list1 = record['text'].split('。')
    start = 0
    while(start+10 < len(list1)):
        sentence = ''
        for i in range(10):
            sentence = sentence + list1[start + i] + '。'
        data.append(sentence)
        start = start + 10

list_return = []
for line in tqdm(data):
    text = zhconv.convert(line, 'zh-cn')
    list_return.append(text)

with open('./HKFinancialStatements_zh_cn.txt', 'w', encoding='utf-8') as f:
    for sentence in tqdm(list_return):
        f.write(sentence + "\n")