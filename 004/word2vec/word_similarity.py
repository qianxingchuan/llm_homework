# -*-coding: utf-8 -*-
# 先运行 word_seg进行中文分词，然后再进行word_similarity计算
# 将Word转换成Vec，然后计算相似度 
from gensim.models import word2vec
import multiprocessing
import os
# 如果目录中有多个文件，可以使用PathLineSentences
current_dir = os.path.dirname(os.path.abspath(__file__))
segment_folder = os.path.join(current_dir, 'seg')
# 切分之后的句子合集
sentences = word2vec.PathLineSentences(segment_folder)

# 设置模型参数，进行训练
model = word2vec.Word2Vec(sentences, vector_size=128, window=3, min_count=1)
print(model.wv.similarity('孔明', '诸葛亮'))
print(model.wv.similarity('曹操', '赤壁'))
print(model.wv.most_similar(positive=['关羽', '张飞'], negative=['吕布']))
# 设置模型参数，进行训练
model2 = word2vec.Word2Vec(sentences, vector_size=128, window=5, min_count=2, workers=multiprocessing.cpu_count())
# 保存模型
model_path = os.path.join(current_dir,'models')
if not os.path.exists(model_path):
    os.makedirs(model_path)
# model2.save('./models/word2Vec.model')
print(model2.wv.similarity('孔明', '诸葛亮'))
print(model2.wv.similarity('曹操', '赤壁'))
print(model2.wv.most_similar(positive=['关羽', '张飞'], negative=['吕布']))