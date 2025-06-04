# utils.py
import numpy as np
import pandas as pd


def evaluate_hand(hand, board):
    # 合并手牌和公共牌
    all_ranks = hand.ranks + board.ranks
    all_suits = hand.suits + board.suits
    all_cards = hand.cards + board.cards

    # 统计点数和花色的频率
    value_counts = get_counts(all_ranks)
    suit_counts = get_counts(all_suits)

    # 判断同花
    is_flush, flush_suit = check_flush(suit_counts)
    flush_rank = -1
    if is_flush:
        # 获取所有同花花色的牌点数
        flush_ranks = [rank for rank, suit in zip(all_ranks, all_suits) if suit == flush_suit]
        # 找到最大的牌点数
        flush_rank = max(flush_ranks)
        is_flush_draw = False

    # 判断顺子
    is_straight, straight_high = check_straight(all_ranks)

    # 判断同花顺
    is_straight_flush = False
    straight_flush = None
    if is_flush:
        flush_values = [rank for rank, suit in zip(all_ranks, all_suits) if suit == flush_suit]
        is_straight_flush, straight_flush = check_straight(flush_values)

    # 分析点数频率
    pairs, three_of_a_kind, four_of_a_kind = check_counts(value_counts)
    is_full_house, full_house = check_full_house(three_of_a_kind, pairs)
    sorted_pairs = sorted(pairs, reverse=True)  # 按大小降序排列

    high_card = max(all_ranks)
    # one_pair = pairs[0] if len(pairs) >= 1 else None
    two_pair = sorted_pairs[:2] if len(pairs) >= 2 else None
    three = max(three_of_a_kind) if len(three_of_a_kind) >= 1 else None
    four = max(four_of_a_kind) if len(four_of_a_kind) >= 1 else None

    one_pair = None

    # 获取公共牌中点数的最大值
    max_board_rank = max(board.ranks)

    # 计算 one_pair
    if len(pairs) >= 1:
        pair_rank = pairs[0]  # 形成对子牌的点数
        if pair_rank > max_board_rank:
            pair_type = "over_pair"  # 超对
        elif pair_rank == max_board_rank:
            pair_type = "top_pair"  # 顶对
        else:
            pair_type = "other"  # 其他
        one_pair = (pair_rank, pair_type)

    # 判断最大组合
    if is_straight_flush:
        max_comb = "straight_flush"
    elif four:
        max_comb = "four_of_a_kind"
    elif is_full_house:
        max_comb = "full_house"
    elif is_flush:
        max_comb = "flush"
    elif is_straight:
        max_comb = "straight"
    elif three:
        max_comb = "three_of_a_kind"
    elif len(pairs) >= 2:
        max_comb = "two_pair"
    elif one_pair:
        max_comb = "one_pair"
    else:
        max_comb = "high_card"

    # Initialize variables
    flush_draw_list = None
    flush_draw_suit = None
    flush_draw_num = None

    # 判断听牌情况
    "Todo: 需要排除已经出现的顺子和同花顺的情况"
    is_straight_draw, straight_draw = check_straight_draw(all_ranks, "hand")
    if not is_flush:
        is_flush_draw, flush_draw_num, flush_draw_suit, flush_draw_list = check_flush_draw(
            suit_counts, len(board), all_cards
        )
    is_straight_flush_draw, straight_flush_draw = check_straight_flush_draw(all_cards)

    # 输出格式转换
    high_card = card_value_inverse(high_card)
    if one_pair:
        one_pair = (card_value_inverse(one_pair[0]), one_pair[1])  # 对子点数转为字符
    if two_pair:
        two_pair = [card_value_inverse(rank) for rank in two_pair]
    if three:
        three = card_value_inverse(three)
    if four:
        four = card_value_inverse(four)
    if straight_high:
        straight_high = card_value_inverse(straight_high)
    if flush_rank != -1:
        flush_rank = card_value_inverse(flush_rank)
    if straight_flush:
        straight_flush = card_value_inverse(straight_flush)
    if full_house:
        full_house = (card_value_inverse(full_house[0]), card_value_inverse(full_house[1]))
    if straight_draw:
        straight_draw = [[card_value_inverse(rank) for rank in combo] for combo in straight_draw]
    if flush_draw_list:
        flush_draw_list = [card_value_inverse(rank) for rank in flush_draw_list]
    if straight_flush_draw:
        straight_flush_draw = [
            [(card_value_inverse(rank), suit) for rank, suit in combo] for combo in straight_flush_draw
        ]

    # 返回结果
    hand_rankings = {
        "max_comb": max_comb,  # 最大组合
        "high_card": high_card,
        "one_pair": one_pair,
        "two_pair": two_pair,
        "three_of_a_kind": three,
        "straight": straight_high,
        "flush": [flush_rank, flush_suit],
        "full_house": full_house,
        "four_of_a_kind": four,
        "straight_flush": straight_flush,
    }

    draws = {
        "straight_draw": straight_draw,
        "flush_draw": [flush_draw_list, flush_draw_suit, flush_draw_num],
        "straight_flush_draw": straight_flush_draw,
    }

    return hand_rankings, draws


def evaluate_board(board):
    # 合并手牌和公共牌
    all_ranks = board.ranks
    all_suits = board.suits
    all_cards = board.cards

    # 统计点数和花色的频率
    value_counts = get_counts(all_ranks)
    suit_counts = get_counts(all_suits)

    # 判断同花
    is_flush, flush_suit = check_flush(suit_counts)
    flush_rank = -1
    if is_flush:
        # 获取所有同花花色的牌点数
        flush_ranks = [rank for rank, suit in zip(all_ranks, all_suits) if suit == flush_suit]
        # 找到最大的牌点数
        flush_rank = max(flush_ranks)
        is_flush_draw = False

    # 判断顺子
    is_straight, straight_high = check_straight(all_ranks)

    # 判断同花顺
    is_straight_flush = False
    straight_flush = None
    if is_flush:
        flush_values = [rank for rank, suit in zip(all_ranks, all_suits) if suit == flush_suit]
        is_straight_flush, straight_flush = check_straight(flush_values)

    # 分析点数频率
    pairs, three_of_a_kind, four_of_a_kind = check_counts(value_counts)
    is_full_house, full_house = check_full_house(three_of_a_kind, pairs)
    sorted_pairs = sorted(pairs, reverse=True)  # 按大小降序排列

    high_card = max(all_ranks)
    one_pair = pairs[0] if len(pairs) >= 1 else None
    two_pair = sorted_pairs[:2] if len(pairs) >= 2 else None
    three = max(three_of_a_kind) if len(three_of_a_kind) >= 1 else None
    four = max(four_of_a_kind) if len(four_of_a_kind) >= 1 else None

    # one_pair = None

    # # 获取公共牌中点数的最大值
    # max_board_rank = max(board.ranks)

    # # 计算 one_pair
    # if len(pairs) >= 1:
    #     pair_rank = pairs[0]  # 形成对子牌的点数
    #     if pair_rank > max_board_rank:
    #         pair_type = "over_pair"  # 超对
    #     elif pair_rank == max_board_rank:
    #         pair_type = "top_pair"  # 顶对
    #     else:
    #         pair_type = "other"  # 其他
    #     one_pair = (pair_rank, pair_type)

    # 判断最大组合
    if is_straight_flush:
        max_comb = "straight_flush"
    elif four:
        max_comb = "four_of_a_kind"
    elif is_full_house:
        max_comb = "full_house"
    elif is_flush:
        max_comb = "flush"
    elif is_straight:
        max_comb = "straight"
    elif three:
        max_comb = "three_of_a_kind"
    elif len(pairs) >= 2:
        max_comb = "two_pair"
    elif one_pair:
        max_comb = "one_pair"
    else:
        max_comb = "high_card"

    # Initialize variables
    flush_draw_list = None
    flush_draw_suit = None
    flush_draw_num = None

    # 判断听牌情况
    "Todo: 需要排除已经出现的顺子和同花顺的情况"
    is_straight_draw, straight_draw = check_straight_draw(all_ranks, "hand")
    if not is_flush:
        is_flush_draw, flush_draw_num, flush_draw_suit, flush_draw_list = check_flush_draw(
            suit_counts, len(board), all_cards
        )
    is_straight_flush_draw, straight_flush_draw = check_straight_flush_draw(all_cards)

    # 输出格式转换
    high_card = card_value_inverse(high_card)
    if one_pair:
        one_pair = (card_value_inverse(one_pair[0]), one_pair[1])  # 对子点数转为字符
    if two_pair:
        two_pair = [card_value_inverse(rank) for rank in two_pair]
    if three:
        three = card_value_inverse(three)
    if four:
        four = card_value_inverse(four)
    if straight_high:
        straight_high = card_value_inverse(straight_high)
    if flush_rank != -1:
        flush_rank = card_value_inverse(flush_rank)
    if straight_flush:
        straight_flush = card_value_inverse(straight_flush)
    if full_house:
        full_house = (card_value_inverse(full_house[0]), card_value_inverse(full_house[1]))
    if straight_draw:
        straight_draw = [[card_value_inverse(rank) for rank in combo] for combo in straight_draw]
    if flush_draw_list:
        flush_draw_list = [card_value_inverse(rank) for rank in flush_draw_list]
    if straight_flush_draw:
        straight_flush_draw = [
            [(card_value_inverse(rank), suit) for rank, suit in combo] for combo in straight_flush_draw
        ]

    # 返回结果
    hand_rankings = {
        "max_comb": max_comb,  # 最大组合
        "high_card": high_card,
        "one_pair": one_pair,
        "two_pair": two_pair,
        "three_of_a_kind": three,
        "straight": straight_high,
        "flush": [flush_rank, flush_suit],
        "full_house": full_house,
        "four_of_a_kind": four,
        "straight_flush": straight_flush,
    }

    draws = {
        "straight_draw": straight_draw,
        "flush_draw": [flush_draw_list, flush_draw_suit, flush_draw_num],
        "straight_flush_draw": straight_flush_draw,
    }

    return hand_rankings, draws


def evaluate_hand_with_board_filter(hand, board):
    """
    结合 evaluate_hand 和 evaluate_board 的结果，过滤掉 board 的听牌
    """
    # 调用 evaluate_hand 获取手牌+公共牌的结果
    hand_rankings, hand_draws = evaluate_hand(hand, board)

    # 调用 evaluate_board 获取 board 自身的听牌情况
    board_rankings, board_draws = evaluate_board(board)

    # 过滤 hand_draws 中与 board_draws 重叠的听牌
    filtered_draws = {}
    for key in hand_draws:
        # 遍历每种听牌类型
        if key in board_draws:
            # 处理 flush_draw 的去重逻辑
            if key == "flush_draw":
                hand_flush_list, hand_suit, hand_num = hand_draws[key]
                board_flush_list, board_suit, board_num = board_draws[key]

                # 如果花色和缺牌数一致，清空整个 flush_draw
                if hand_suit == board_suit and hand_num == board_num:
                    filtered_draws[key] = [None, None, None]  # 清空整个 flush_draw
                else:
                    filtered_draws[key] = hand_draws[key]  # 保留其他内容

            # 处理 straight_draw 的去重逻辑
            elif key == "straight_draw":
                hand_combinations = hand_draws[key] or []
                board_combinations = board_draws[key] or []

                # 使用集合过滤出不在 board 的听牌组合
                filtered_combinations = [combo for combo in hand_combinations if combo not in board_combinations]
                filtered_draws[key] = filtered_combinations if filtered_combinations else None  # 返回空列表

            # 处理 straight_flush_draw 的去重逻辑
            elif key == "straight_flush_draw":
                hand_combinations = hand_draws[key] or []
                board_combinations = board_draws[key] or []

                # 使用集合过滤出不在 board 的听牌组合
                filtered_combinations = [combo for combo in hand_combinations if combo not in board_combinations]
                filtered_draws[key] = filtered_combinations if filtered_combinations else None  # 返回空列表

            else:
                # 保留其他听牌
                filtered_draws[key] = hand_draws[key]
        else:
            # 不在 board_draws 中，直接保留
            filtered_draws[key] = hand_draws[key]

    return hand_rankings, filtered_draws


def evaluate_group_with_board_filter(grouped_data, board):
    """
    计算分组结果中手牌和公共牌的结果，并根据 board 进行滥选
    :param grouped_data: [[hand1, hand2], [hand3], ...] 格式的分组手牌
    :param board: 公共牌
    :return: 过滤后的分组结果，格式与 grouped_data 一致
    """
    filtered_group = []

    for group in grouped_data:
        filtered_subgroup = []
        for hand in group:
            hand_instance = Hand(hand)  # 将字符串应用为 Hand 类
            hand_rankings, filtered_draws = evaluate_hand_with_board_filter(hand_instance, board)

            # 只保留 max_comb 和其对应的结构
            max_comb = hand_rankings["max_comb"]
            value = hand_rankings.get(max_comb, None)
            processed_rankings = {"max_comb": max_comb, "value": value}

            filtered_subgroup.append((hand, processed_rankings, filtered_draws))
        filtered_group.append(filtered_subgroup)

    return filtered_group


def card_value(rank):
    if rank == "A":
        return 14  # A 可以作为最大点数
    elif rank == "K":
        return 13
    elif rank == "Q":
        return 12
    elif rank == "J":
        return 11
    elif rank == "T":
        return 10
    else:
        return int(rank)


def card_value_inverse(value):
    if value == 14:
        return "A"
    elif value == 13:
        return "K"
    elif value == 12:
        return "Q"
    elif value == 11:
        return "J"
    elif value == 10:
        return "T"
    else:
        return str(value)


def get_counts(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def check_counts(value_counts):
    pairs = []
    three_of_a_kind = []
    four_of_a_kind = []
    for value, count in value_counts.items():
        if count == 2:
            pairs.append(value)
        elif count == 3:
            three_of_a_kind.append(value)
        elif count == 4:
            four_of_a_kind.append(value)
    return pairs, three_of_a_kind, four_of_a_kind


def check_flush(suit_counts):
    for suit, count in suit_counts.items():
        if count >= 5:
            return True, suit
    return False, None


def check_straight(values):
    unique_values = set(values)
    value_list = list(unique_values)
    value_list.sort()

    # 考虑A作为1的情况
    if 14 in value_list:
        value_list.append(1)
        value_list.sort()

    for i in range(len(value_list) - 4):
        window = value_list[i : i + 5]
        if window[4] - window[0] == 4 and len(window) == 5:
            return True, window[4]
    return False, None


def check_full_house(three_of_a_kind, pairs):
    # 确保至少有一组三条
    if len(three_of_a_kind) >= 1:
        # 如果有多个三条，选择最大的
        best_three = max(three_of_a_kind)

        # 如果有对子，选择最大的对子
        if len(pairs) >= 1:
            best_pair = max(pairs)
            return True, (best_three, best_pair)

        # 如果没有对子，但有两个三条，可以把一个三条当作对子
        elif len(three_of_a_kind) > 1:
            # 选择第二大的三条作为对子
            second_best_three = sorted(three_of_a_kind)[-2]
            return True, (best_three, second_best_three)

    return False, None


def check_straight_draw(values, mode="hand"):
    # 将 A 视为 1 和 14
    hand_ranks = set(values)
    N = 1
    if 14 in values:
        hand_ranks.add(1)

    if mode == "hand":
        # 计算剩余的牌数量
        total_cards = 7  # 2 手牌 + 5 公共牌
        N = total_cards - len(values)
    elif mode == "board":
        # 计算剩余的牌数量
        total_cards = 5
        N = total_cards - len(values)

    # 如果没有剩余的牌，直接返回 False 和空列表
    if N <= 0:
        return False, None

    # 存储结果的列表
    missing_one_card_combinations = []
    missing_two_cards_combinations = []

    for start in range(1, 11):  # 顺子的起始点数
        straight = set(range(start, start + 5))
        missing_cards = straight - hand_ranks
        # 只考虑牌面在 1 到 13 的牌
        missing_cards = set(filter(lambda x: 1 <= x <= 13, missing_cards))
        num_missing = len(missing_cards)
        if 0 < num_missing <= N:
            sorted_missing = sorted(missing_cards)
            if num_missing == 1:
                if sorted_missing not in missing_one_card_combinations:
                    missing_one_card_combinations.append(sorted_missing)
            elif num_missing == 2:
                if sorted_missing not in missing_two_cards_combinations:
                    missing_two_cards_combinations.append(sorted_missing)

    # 遍历比较，删除重复的较大牌
    for combo_two in missing_two_cards_combinations[:]:
        larger_card = combo_two[-1]
        for combo_one in missing_one_card_combinations:
            if larger_card in combo_one:
                missing_two_cards_combinations.remove(combo_two)
                break  # 跳出内层循环，继续检查下一个组合

    # 合并两个缺失牌组合的列表
    missing_combinations = missing_one_card_combinations + missing_two_cards_combinations

    # 判断是否存在听顺
    has_straight_draw = bool(missing_combinations)

    return has_straight_draw, missing_combinations


def check_straight_flush_draw(cards, mode="hand"):
    N = 1
    if mode == "hand":
        # 计算剩余的牌数量
        total_cards = 7  # 2 手牌 + 5 公共牌
        N = total_cards - len(cards)
    elif mode == "board":
        # 计算剩余的牌数量
        total_cards = 5
        N = total_cards - len(cards)

    if N <= 0:
        return False, None

    # 按花色分类牌
    suits = {}
    for value, suit in cards:
        suits.setdefault(suit, []).append(card_value(value[0]))

    has_straight_flush_draw = False
    missing_combinations = []

    # 对每个花色的牌进行听顺检测
    for suit, values in suits.items():
        num_cards_same_suit = len(values)
        if num_cards_same_suit + N < 5:
            continue  # 不可能形成同花，检查下一个花色
        # 调用 check_straight_draw 函数
        has_draw, missing_cards = check_straight_draw(values, mode)

        if has_draw:
            has_straight_flush_draw = True
            # 将缺失的牌附加花色信息
            missing_cards_with_suit = [[(rank, suit) for rank in combo] for combo in missing_cards]
            # 合并结果
            missing_combinations.extend(missing_cards_with_suit)

    return has_straight_flush_draw, missing_combinations


def check_flush_draw(suit_counts, board_size, cards):
    # 标准扑克牌的数字
    all_ranks = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"}

    # 收集已经出现的花色和数字
    seen_cards = {"c": set(), "d": set(), "h": set(), "s": set()}  # 各花色的已出现牌
    for rank, suit in cards:
        seen_cards[suit].add(rank)

    # 检查是否满足听牌条件
    for suit, count in suit_counts.items():
        if count >= board_size:  # 满足听同花的花色
            missing_cards = all_ranks - seen_cards[suit]  # 未出现的牌
            return True, (5 - count), suit, list(missing_cards)  # 听牌状态，缺几张，听花色，听哪些牌

    return False, None, None, None  # 不满足听牌


def get_top_combinations(hands, weights, ev, effective_stack, top_n):
    """
    根据手牌范围、权重和EV，计算最靠近二维特征右上角的手牌组合
    :param hands: 手牌范围列表
    :param weights: 对应手牌的权重列表
    :param ev: 对应手牌的EV列表
    :param effective_stack: 用于归一化EV的有效筹码量
    :param top_n: 返回的组合数
    :return: 最靠近右上角的组合及其对应的weights和ev
    """

    # 1. 将输入数据加载为DataFrame
    data = pd.DataFrame({"hands": hands, "weights": weights, "ev": ev})

    # 新增标签列label
    data["label"] = np.where(
        data["hands"].str[1] == data["hands"].str[3],
        data["hands"].str[0] + data["hands"].str[2] + "s",
        data["hands"].str[0] + data["hands"].str[2] + "o",
    )

    # 2. 对EV进行归一化
    data["ev_scaled"] = data["ev"] / effective_stack

    # 3. 对weights进行归一化（min-max归一化）
    weight_max = data["weights"].max()
    weight_min = data["weights"].min()
    if weight_max != weight_min:  # 避免除以零
        data["weights_scaled"] = (data["weights"] - weight_min) / (weight_max - weight_min)
    else:
        data["weights_scaled"] = data["weights"]  # 如果最大值等于最小值，直接使用原值

    # 4. 计算每个组合到(0, 0)的距离
    data["distance"] = np.sqrt(data["weights_scaled"] ** 2 + data["ev_scaled"] ** 2)

    # 5. 按距离排序
    data = data.sort_values(by="distance", ascending=False)

    # 6. 选择前 top_n + 1 个不同 label 的组合（允许重复，但同一label不计入计数）
    unique_labels = set()
    top_combinations = []

    for _, row in data.iterrows():
        label = row["label"]
        if label not in unique_labels:
            unique_labels.add(label)
            top_combinations.append(row)
            if len(unique_labels) == top_n + 1:
                break
        else:
            top_combinations.append(row)

    # 7. 去掉最后一个组合，确保仅有 top_n 个不同的label
    if len(unique_labels) > top_n:
        top_combinations = top_combinations[:-1]

    # 8. 转换为 DataFrame 并返回 hands, weights, ev 和 label
    top_combinations_df = pd.DataFrame(top_combinations)

    # 9. 根据label对组合进行分组
    grouped_data = []
    for label, group in top_combinations_df.groupby("label", sort=False):
        group = group.sort_values(by="distance", ascending=False)
        distances = group["distance"].values

        # 检查距离并分组
        split_indices = [0]
        for i in range(len(distances) - 1):
            if distances[i] - distances[i + 1] > 0.04:
                split_indices.append(i + 1)

        split_indices.append(len(distances))

        # 分组后将分组添加到结果集合
        for i in range(len(split_indices) - 1):
            start, end = split_indices[i], split_indices[i + 1]
            grouped_data.append(group.iloc[start:end]["hands"].tolist())

    # 10. 返回分组结果
    return top_combinations_df[["hands", "weights", "ev", "label", "distance"]], grouped_data
