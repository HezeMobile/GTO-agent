from .utils import card_value


class Hand:
    def __init__(self, cards):
        """
        初始化手牌类。
        参数：
        - cards: 一个包含两张牌的列表，每张牌是一个字符串，如 'Ah' 表示红桃A。
        """
        if not isinstance(cards, list) or len(cards) != 2:
            raise ValueError("必须提供两张牌的列表")

        valid_ranks = ["A", "K", "Q", "J", "T"] + [str(n) for n in range(2, 10)]
        valid_suits = ["h", "d", "s", "c"]

        # 解析输入的牌
        parsed_cards = []
        for card in cards:
            if not isinstance(card, str) or len(card) < 2:
                raise ValueError(f"无效的牌格式: {card}")
            rank = card[:-1]  # 点数 (比如 'A', 'K', '5' 等)
            suit = card[-1]  # 花色 (比如 'h', 'd', 's', 'c')

            if rank not in valid_ranks:
                raise ValueError(f"无效的牌点数: {rank}")
            if suit not in valid_suits:
                raise ValueError(f"无效的牌花色: {suit}")

            parsed_cards.append((rank, suit))

        self.cards = parsed_cards
        # 将rank和suit直接存储为self.ranks和self.suits(ranks和suits长度相同
        self.ranks = [card_value(card[0]) for card in parsed_cards]  # 只存储rank
        self.suits = [card[1] for card in parsed_cards]  # 只存储suit

    def __repr__(self):
        return f"Hand(cards={self.cards})"
