# CPM_financial_sentiment
This repo is for financial sentiment on [CPM](https://github.com/orgs/TsinghuaAI/repositories), similar to FinBERT.
## Model file
Download [here](https://pan.baidu.com/s/1qwRSqSgwDvjAyhL_srC3yg), password```g5y8```
1. ```model_pretrain_distill.pth```, pretrain CPM model, converted from the downloaded file CPM_distill [here](https://cpm.baai.ac.cn/download.html)  
2. ```financial_sentiment.pth```, trained model   
## Dataset
Two part, one is translated from which FinBERT used. [FinBERT](https://github.com/yya518/FinBERT). The other is from [webpage](https://xueqiu.com/)
1. Use ```data.ipynb``` to process row data.
2. Use ```dataset_setiment.ipynb``` to convert json data.
## Train
```train_distill_sentiment.py```, only train the MLP part.
## Use
```demo_sentiment.ipynb```, remember to put ```financial_sentiment.pth``` in ```/models/``` first
