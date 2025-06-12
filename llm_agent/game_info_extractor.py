from typing import Dict, Any, List
import re
import json
import random

import openai


def format_card(card: str) -> str:
    """Format a single card to have valid rank and suit."""
    valid_ranks = ["A", "K", "Q", "J", "T"] + [str(n) for n in range(2, 10)]
    valid_suits = ["h", "d", "s", "c"]

    card = card.lower().strip()

    if not card:
        return ""

    rank = card[0].upper()
    suit = card[-1].lower()

    if rank in valid_ranks and suit in valid_suits:
        return f"{rank}{suit}"
    return ""


def format_cards_array(cards: List[str]) -> List[str]:
    """Format an array of cards, ensuring each card is valid."""
    if not cards:
        return []
    formatted_cards = [format_card(card) for card in cards]
    valid_cards = [card for card in formatted_cards if card]
    return valid_cards


def check_duplicate_cards(cards_list: List[str]) -> bool:
    """Check if there are any duplicate cards in the given list."""
    return len(cards_list) != len(set(cards_list))


def format_game_info(game_info: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    """Format game information according to specific rules.

    Returns:
        tuple: (formatted_info, status_message)
            - formatted_info: Dict containing the formatted game information
            - status_message: String indicating the status of the formatting
    """
    formatted_info = game_info.copy()
    status_message = "Success"

    # Check positions
    user_pos = game_info["user_position"].strip().upper()
    opp_pos = game_info["opponent_position"].strip().upper()

    if not user_pos:
        status_message = "Error: User position is missing."
    elif not opp_pos:
        status_message = "Error: Opponent position is missing."

    formatted_info["user_position"] = user_pos
    formatted_info["opponent_position"] = opp_pos

    if formatted_info["user_position"] == formatted_info["opponent_position"]:
        status_message = "Error: User and opponent positions could not be the same."

    # Check cards
    if not game_info.get("user_hand"):
        status_message = "Error: User hand is missing."
    elif len(game_info.get("user_hand")) != 2:
        status_message = "Error: User hand must contain exactly 2 cards."

    if game_info.get("flop") and len(game_info.get("flop")) != 3:
        status_message = "Error: Flop must contain exactly 3 cards."

    if game_info.get("turn") and len(game_info.get("turn")) != 1:
        status_message = "Error: Turn must contain exactly 1 card."

    if game_info.get("river") and len(game_info.get("river")) != 1:
        status_message = "Error: River must contain exactly 1 card."

    card_fields = ["user_hand", "flop", "turn", "river"]
    for field in card_fields:
        if game_info.get(field):
            formatted_info[field] = format_cards_array(game_info[field])

    all_cards = []
    if formatted_info.get("user_hand"):
        all_cards.extend(formatted_info["user_hand"])
    if formatted_info.get("flop"):
        all_cards.extend(formatted_info["flop"])
    if formatted_info.get("turn"):
        all_cards.extend(formatted_info["turn"])
    if formatted_info.get("river"):
        all_cards.extend(formatted_info["river"])

    if check_duplicate_cards(all_cards):
        status_message = "Error: Duplicate cards found."

    # Check actions
    plain_actions = ["Call", "Check"]

    def format_actions(actions_list):
        formatted_actions = []
        for action in actions_list:
            if action.startswith(("Bet(", "Raise(", "AllIn(")):
                try:
                    amount = int(action.split("(")[1].rstrip(")"))
                    action_type = action.split("(")[0]
                    formatted_actions.append(f"{action_type}({amount})")
                except (ValueError, IndexError):
                    continue
            # Handle simple actions
            elif action in plain_actions:
                formatted_actions.append(action)
        return formatted_actions

    # Get actions for each street
    flop_actions = game_info.get("flop_actions", [])
    turn_actions = game_info.get("turn_actions", [])
    river_actions = game_info.get("river_actions", [])

    formatted_info["flop_actions"] = format_actions(flop_actions)
    formatted_info["turn_actions"] = format_actions(turn_actions)
    formatted_info["river_actions"] = format_actions(river_actions)

    return formatted_info, status_message


def extract_poker_info(input_text: str) -> tuple[Dict[str, Any], str]:
    game_info_empty = {
        "user_position": "",
        "opponent_position": "",
        "user_hand": [],
        "flop": "",
        "turn": "",
        "river": "",
        "flop_actions": [],
        "turn_actions": [],
        "river_actions": [],
    }

    prompt = f"""Extract the following poker information from the given text and format it as a JSON object:
    - User's position (BTN, SB, BB, etc.)
    - Opponent's position (BTN, SB, BB, etc.)
    - User's hand (2 cards, from the largest to smallest)
    - Flop cards (3 cards, from the largest to smallest)
    - Turn card (1 card, optional)
    - River card (1 card, optional)
    - Flop actions (list of actions in forms like Bet(6), Bet(10), Call, etc.)
    - Turn actions (list of actions in forms like Bet(6), Bet(10), Call, etc.)
    - River actions (list of actions in forms like Bet(6), Bet(10), Call, etc.)

    Input text: {input_text}

    Return only the JSON object in the following format:
    {{
        "user_position": "position",
        "opponent_position": "position",
        "user_hand": ["user_card1","user_card2"],
        "flop": ["cards1","cards2","cards3"],
        "turn": ["cards4"],
        "river": ["cards5"],
        "flop_actions": ["action1", "action2", ...],
        "turn_actions": ["action1", "action2", ...],
        "river_actions": ["action1", "action2", ...],
    }}
    """

    client = openai.OpenAI(base_url="https://api.deepseek.com")

    try:
        chat_completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a poker information extraction assistant. Extract and format poker game information into JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=1.3,
            max_tokens=512,
        )
        result = chat_completion.choices[0].message.content
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            game_info = json.loads(match.group(0))
            print("game_info:\n", game_info)
            formatted_info, status_message = format_game_info(game_info)
            return formatted_info, status_message
        else:
            return game_info_empty, "Error: No JSON object found in the response."

    except Exception as e:
        print(f"Error extracting poker information: {str(e)}")
        return game_info_empty
