from gensim.models import Word2Vec
import os

# 加载已保存的模型
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'word2vec.model')
model = Word2Vec.load(model_path)

# 使用模型进行各种操作
def find_similar_words(word, topn=10):
    """查找相似词"""
    try:
        similar_words = model.wv.most_similar(word, topn=topn)
        return similar_words
    except KeyError:
        return f"词汇 '{word}' 不在模型词汇表中"

def calculate_similarity(word1, word2):
    """计算两个词的相似度"""
    try:
        similarity = model.wv.similarity(word1, word2)
        return similarity
    except KeyError as e:
        return f"词汇不在模型词汇表中: {e}"

def word_analogy(positive, negative):
    """词汇类比推理"""
    try:
        result = model.wv.most_similar(positive=positive, negative=negative)
        return result
    except KeyError as e:
        return f"词汇不在模型词汇表中: {e}"

# 示例使用
if __name__ == "__main__":
    # 查找相似词
    print("与'孔明'相似的词：")
    print(find_similar_words('孔明'))
    
    # 计算相似度
    print(f"\n'孔明'和'诸葛亮'的相似度: {calculate_similarity('孔明', '诸葛亮')}")
    
    # 词汇类比
    print("\n类比推理 (关羽 + 张飞 - 吕布):")
    print(word_analogy(['关羽', '张飞'], ['吕布']))
    
    # 获取词向量
    try:
        vector = model.wv['孔明']
        print(f"\n'孔明'的词向量维度: {vector.shape}")
        print(f"词向量前5个值: {vector[:5]}")
    except KeyError:
        print("词汇不在模型中")