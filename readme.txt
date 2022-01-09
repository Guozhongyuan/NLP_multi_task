数据处理：
句号，去除年份、月份、公司名

正式的语句用CPM

情感分析旨在自动识别和提取文本中的倾向、立场、评价、观点等主观信息。它包含各式各样的任务，比如句子级情感分类、评价对象级情感分类、观点抽取、情绪分类等。情感分析是人工智能的重要研究方向，具有很高的学术价值。同时，情感分析在消费决策、舆情分析、个性化推荐等领域均有重要的应用，具有很高的商业价值。
近日，百度正式发布情感预训练模型SKEP（Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis）。SKEP利用情感知识增强预训练模型， 在14项中英情感分析典型任务上全面超越SOTA，此工作已经被ACL 2020录用。
论文地址：https://arxiv.org/abs/2005.05635
为了方便研发人员和商业合作伙伴共享效果领先的情感分析技术，本次百度在Senta中开源了基于SKEP的情感预训练代码和中英情感预训练模型。而且，为了进一步降低用户的使用门槛，百度在SKEP开源项目中集成了面向产业化的一键式情感分析预测工具。用户只需要几行代码即可实现基于SKEP的情感预训练以及模型预测功能。

SKEP是百度研究团队提出的基于情感知识增强的情感预训练算法，此算法采用无监督方法自动挖掘情感知识，然后利用情感知识构建预训练目标，从而让机器学会理解情感语义。SKEP为各类情感分析任务提供统一且强大的情感语义表示。
百度研究团队在三个典型情感分析任务，句子级情感分类（Sentence-level Sentiment Classification），评价对象级情感分类（Aspect-level Sentiment Classification）、观点抽取（Opinion Role Labeling），共计14个中英文数据上进一步验证了情感预训练模型SKEP的效果。实验表明，以通用预训练模型ERNIE（内部版本）作为初始化，SKEP相比ERNIE平均提升约1.2%，并且较原SOTA平均提升约2%，具体效果如下表：

安装paddlepaddle 和 Senta
python -m pip install paddlepaddle-gpu==2.2.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
git clone https://github.com/baidu/Senta.git
cd Senta
python -m pip install .

训练
模型下载
git clone https://github.com/baidu/Senta.git
cd ./model_files
sh download_ernie_1.0_skep_large_ch.sh
环境配置