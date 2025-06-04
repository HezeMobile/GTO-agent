import sys
import os
import argparse
import time
import json
import requests
import copy

import openai

os.environ["OPENAI_API_KEY"] = "sk-8c6fd7cbaf5842d09cdd23be37ba0ecf"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gto_facts_converter import Board
from gto_facts_converter import Hand
from gto_facts_converter import (
    evaluate_board,
    evaluate_hand_with_board_filter,
    get_top_combinations,
)
from poker_text_detector import PokerTextDetector
from game_info_extractor import extract_poker_info


TERM_MAP = {
    "high_card": "高牌",
    "one_pair": "对子",
    "two_pair": "两对",
    "three_of_a_kind": "三条",
    "straight": "顺子",
    "flush": "同花",
    "full_house": "葫芦",
    "four_of_a_kind": "四条",
    "straight_flush": "同花顺",
    "straight_draw": "顺子听牌",
    "flush_draw": "同花听牌",
    "straight_flush_draw": "同花顺听牌",
    "over_pair": "超对",
    "top_pair": "顶对",
}

STAGE_MAP = ["翻牌", "转牌", "河牌"]

QUERY = """有效筹码量：{effective_stack}BB
玩家位置：{player_position}
对手位置：{op_position}
玩家手牌：{hand}
翻后：{game}
对手手牌范围：{op_range}
GTO的结果：{gto}"""

SYSTEM_PROMPT = """你是一个德州扑克游戏中的解释器。如果用户的输入和德州扑克无关，请提醒用户输入与德州扑克有关的信息！"""

# pylint: disable=line-too-long
USER_PROMPT = """结合玩家位置、对手位置、玩家手牌、对手手牌范围等信息，你需要通过下面几点对GTO结果中各项行动的原因进行解释：
1. 分析公共牌面的干燥性和连接性
{analysis1}
2. 根据对手的行动，分析对手的范围可能包括了哪些价值组合，听牌组合，需要关注对手的组合有：
{analysis2}
3. 分析各阶段玩家手牌的强弱，以及玩家手牌对对手牌力的影响
{analysis3}

请基于下面的局面信息，按照初步分析，解释GTO结果中各项动作的原因：
{query}
"""


def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Explain the GTO actions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek",
        help="The model used to generate explanation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="llm_agent/outcome.txt",
        help="Where to store the output.",
    )
    return parser.parse_args()


def prepare_analysis1(board: list) -> str:
    analysis1 = f"翻牌面是{board[0]}，{board[1]}，{board[2]}\n"
    if len(board) >= 4:
        analysis1 += f"转牌面是{board[3]}，{board[0]}，{board[1]}，{board[2]}\n"
    if len(board) == 5:
        analysis1 += (
            f"河牌面是{board[4]}，{board[3]}，{board[0]}，{board[1]}，{board[2]}\n"
        )
    _, draws = evaluate_board(Board(board))
    stage = STAGE_MAP[len(board) - 3]
    if draws["straight_draw"]:
        analysis1 += f"{stage}面有成顺的可能\n"
    else:
        analysis1 += f"{stage}面没有成顺的可能\n"
    if draws["flush_draw"][0]:
        analysis1 += f"{stage}面有同花的可能\n"
    else:
        analysis1 += f"{stage}面没有同花的可能\n"
    if draws["straight_flush_draw"]:
        analysis1 += f"{stage}面有成同花顺的可能\n"
    elif draws["straight_draw"] and draws["flush_draw"][0]:
        analysis1 += f"{stage}面没有成同花顺的可能\n"
    return analysis1


def prepare_analysis2(
    op_range: list, op_weight: list, op_ev: list, board: list, effective_stack: int
) -> str:
    # TODO
    top_n = 5
    tops = get_top_combinations(
        op_range, op_weight, op_ev, effective_stack=effective_stack, top_n=top_n
    )
    hands_label = tops[0]["label"].drop_duplicates().tolist()
    hands_group = tops[1]

    assert len(hands_label) == top_n

    analysis2 = ""
    for i in range(top_n):
        hand = hands_group[i][0]
        op_hand_rankings, op_draws = evaluate_hand_with_board_filter(
            Hand([hand[:2], hand[2:]]), Board(board)
        )
        analysis2 = (
            analysis2 + hands_label[i] + "，" + TERM_MAP[op_hand_rankings["max_comb"]]
        )
        if op_draws["straight_flush_draw"]:
            analysis2 += "，同花顺听牌"
        else:
            if op_draws["straight_draw"]:
                analysis2 += "，顺子听牌"
            if op_draws["flush_draw"][0]:
                analysis2 += "，同花听牌"
        analysis2 += "\n"
    return analysis2


def prepare_analysis3(hand: str, board: list) -> str:
    analysis3 = ""
    stage_id = len(board) - 3
    for i in range(stage_id + 1):
        if i > 0:
            analysis3 += "\n\n"
        hand_rankings, draws = evaluate_hand_with_board_filter(
            Hand([hand[:2], hand[2:]]), Board(board[: 3 + i])
        )
        analysis3 = (
            analysis3
            + STAGE_MAP[i]
            + "时玩家已经成牌："
            + TERM_MAP[hand_rankings["max_comb"]]
            + str(hand_rankings[hand_rankings["max_comb"]])
        )
        if draws["straight_draw"]:
            analysis3 = (
                analysis3
                + "\n玩家顺子听牌，听"
                + "，或".join(["".join(x) for x in draws["straight_draw"]])
            )
        else:
            analysis3 += "\n玩家不可能成顺子"
        if draws["flush_draw"][0]:
            data = draws["flush_draw"]
            analysis3 = (
                analysis3
                + "\n玩家同花听牌，听"
                + ", ".join(f"{num}{data[1]}" for num in data[0])
                + "中的"
                + str(data[2])
                + "张"
            )
        else:
            analysis3 += "\n玩家不可能成同花"
        if draws["straight_flush_draw"]:
            analysis3 = (
                analysis3
                + "\n玩家同花顺听牌，听"
                + "，或".join(
                    [
                        f"{card1[0]}{card1[1]}{card2[0]}{card2[1]}"
                        for card1, card2 in draws["straight_flush_draw"]
                    ]
                )
            )
        elif draws["straight_draw"] and draws["flush_draw"][0]:
            analysis3 += "\n玩家不可能成同花顺"
    return analysis3


def prepare_prompt(user_data: dict) -> str:
    # default variable for now
    effective_stack = 100
    player_id = "ip"

    # contruct rust API inputs
    actions = copy.deepcopy(user_data["actions"])
    if len(actions) >= 4:
        actions.insert(4, "river")
        actions.insert(2, "turn")
    elif len(actions) >= 2:
        actions.insert(2, "turn")

    data = {
        "user_spt": user_data["user_position"],
        "opponent_spt": user_data["opponent_position"],
        "user_hand": "".join(user_data["user_hand"]),
        "flop": "".join(user_data["flop"]),
        "turn": user_data["turn"][0] if user_data["turn"] else "",
        "river": user_data["river"][0] if user_data["river"] else "",
        "actions": actions,
    }
    print(data)

    response = requests.post("http://127.0.0.1:8081/demo/getGto", json=data)
    response = response.json()
    # print(response)

    board_str = data["flop"] + data["turn"] + data["river"]
    board = [board_str[i : i + 2] for i in range(0, len(board_str), 2)]

    # game history
    assert player_id == "ip"
    action_history = user_data["actions"]
    game = f"翻牌面是{board[0]}，{board[1]}，{board[2]}，对手{action_history[0]}"
    if len(board) >= 4:
        game += f"，玩家{action_history[1]}\n"
        game += f"转牌面是{board[3]}，{board[0]}，{board[1]}，{board[2]}，对手{action_history[2]}"
    if len(board) == 5:
        game += f"，玩家{action_history[3]}\n"
        game += f"河牌面是{board[4]}，{board[3]}，{board[0]}，{board[1]}，{board[2]}，对手{action_history[4]}"

    action_space = response["available_actions"]
    action_probs = response["available_actions_probability"]
    gto = ", ".join(
        [f"{action_space[n]}:{action_probs[n]*100}%" for n in range(len(action_space))]
    )

    nonzero_index = [
        i for i, x in enumerate(response["opponent_hands_weights"]) if x > 1e-4
    ]

    # prepare analysis
    analysis1 = prepare_analysis1(board)
    analysis2 = prepare_analysis2(
        [response["opponent_hands_range"][i] for i in nonzero_index],
        [response["opponent_hands_weights"][i] for i in nonzero_index],
        [response["opponent_hands_ev"][i] for i in nonzero_index],
        board,
        effective_stack,
    )
    analysis3 = prepare_analysis3(data["user_hand"], board)

    # prepare query
    query = QUERY.format(
        effective_stack=effective_stack,
        player_position=data["user_spt"],
        op_position=data["opponent_spt"],
        hand=data["user_hand"],
        game=game,
        op_range=response["opponent_hands_range"],
        gto=gto,
    )

    user_prompt = USER_PROMPT.format(
        analysis1=analysis1,
        analysis2=analysis2,
        analysis3=analysis3,
        query=query,
    )

    return user_prompt


def explain(messages: str, model: str) -> str:
    if model == "deepseek":
        client = openai.OpenAI(base_url="https://api.deepseek.com")
        try:
            chat_completion = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False,
                temperature=1.3,
                max_tokens=2048,
            )
            return chat_completion.choices[0].message.content
        except Exception as ex:
            print(ex)
            time.sleep(3)
    return "LLM model error"


def force_correct(content: str) -> str:
    # TODO: force correct terms and examples in a second LLM call
    return content
