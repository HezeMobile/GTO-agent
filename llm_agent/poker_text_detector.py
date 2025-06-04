import jieba
import numpy as np
from collections import Counter
from typing import List, Set, Dict

class PokerTextDetector:
    def __init__(self):
        # 初始化德州扑克相关的关键词集合
        self.poker_keywords = {
            # 基础术语
            '德州扑克', '扑克', 'poker', 'texas holdem', 'holdem',
            '牌局', 'game', '底池', 'pot', '盲注', 'blind',
            '小盲', 'small blind', '大盲', 'big blind', '庄家', 'dealer',
            '按钮', 'button', '位置', 'position',
            
            # 牌局阶段
            '翻牌', 'flop', '转牌', 'turn', '河牌', 'river',
            '公共牌', 'community cards', '手牌', 'hole cards',
            
            # 动作
            '弃牌', 'fold', '跟注', 'call', '加注', 'raise',
            '全押', 'all in', '下注', 'bet', '过牌', 'check',
            '梭哈', '诈唬', 'bluff', '偷盲', 'steal',
            
            # 牌型
            '同花顺', 'straight flush', '四条', 'four of a kind',
            '葫芦', 'full house', '同花', 'flush', '顺子', 'straight',
            '三条', 'three of a kind', '两对', 'two pair',
            '一对', 'one pair', '高牌', 'high card',
            
            # 牌面
            'AA', 'KK', 'QQ', 'JJ', 'TT', 'AK', 'AQ', 'AJ',
            'KQ', 'KJ', 'QJ', '红桃', 'hearts', '黑桃', 'spades',
            '梅花', 'clubs', '方块', 'diamonds',
            'A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2'
        }
        
    def add_keywords(self, new_keywords: Set[str]) -> None:
        """添加新的关键词到词袋中"""
        self.poker_keywords.update(new_keywords)
    
    def calculate_tf(self, text: str) -> Dict[str, float]:
        """计算文本的词频（TF）"""
        # 使用结巴分词
        words = list(jieba.cut(text))
        word_count = Counter(words)
        total_words = len(words)
        
        # 计算TF
        tf = {word: count/total_words for word, count in word_count.items()}
        return tf
    
    def calculate_idf(self, text: str) -> Dict[str, float]:
        """计算逆文档频率（IDF）"""
        words = set(jieba.cut(text))
        # 计算每个词在关键词集合中出现的次数
        idf = {}
        for word in words:
            if word in self.poker_keywords:
                idf[word] = np.log(len(self.poker_keywords) / (1 + 1))  # 简化版IDF计算
            else:
                idf[word] = 0
        return idf
    
    def calculate_tfidf(self, text: str) -> float:
        """计算文本的TF-IDF得分"""
        tf = self.calculate_tf(text)
        idf = self.calculate_idf(text)
        
        # 计算TF-IDF得分
        tfidf_score = sum(tf.get(word, 0) * idf.get(word, 0) for word in set(jieba.cut(text)))
        return tfidf_score
    
    def is_poker_related(self, text: str, threshold: float = 0.1) -> bool:
        """判断文本是否与德州扑克相关"""
        score = self.calculate_tfidf(text)
        return score > threshold

    def analyze_text(self, text: str) -> None:
        """分析文本，显示分词结果和每个词的得分"""
        # 分词
        words = list(jieba.cut(text))
        print("\n分词结果:")
        print(" ".join(words))
        
        # 计算TF和IDF
        tf = self.calculate_tf(text)
        idf = self.calculate_idf(text)
        
        # 计算每个词的TF-IDF得分
        word_scores = {}
        for word in set(words):
            tf_score = tf.get(word, 0)
            idf_score = idf.get(word, 0)
            tfidf_score = tf_score * idf_score
            word_scores[word] = {
                'TF': tf_score,
                'IDF': idf_score,
                'TF-IDF': tfidf_score,
                '是否关键词': word in self.poker_keywords
            }
        
        # 按TF-IDF得分排序
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1]['TF-IDF'], reverse=True)
        
        print("\n词得分详情:")
        print("-" * 60)
        print(f"{'词语':<10} {'TF':<10} {'IDF':<10} {'TF-IDF':<10} {'是否关键词':<10}")
        print("-" * 60)
        for word, scores in sorted_words:
            print(f"{word:<10} {scores['TF']:<10.4f} {scores['IDF']:<10.4f} {scores['TF-IDF']:<10.4f} {str(scores['是否关键词']):<10}")
        print("-" * 60)
        
        # 显示总体得分
        total_score = self.calculate_tfidf(text)
        print(f"\n文本总体TF-IDF得分: {total_score:.4f}")

# test
if __name__ == "__main__":
    detector = PokerTextDetector()
    
    new_keywords = {'梭哈', '诈唬', '偷盲', 'bluff', 'steal'}
    detector.add_keywords(new_keywords)
    
    # test_text = "这把牌我拿到了AA，在flop前raise，对手call后flop是KQJ，我继续bet，对手fold了。"
    test_text = "玩家位置UTG，对手位置UTG+1。玩家手牌Jd8d。翻牌是Td6c4S，对手Bet(6)，玩家Bet(12),对手Ca11。转牌是Ah，对手Bet(12)。"
    detector.analyze_text(test_text) 