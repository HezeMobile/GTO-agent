from typing import Dict, Any, List
import re
import json

import openai


def format_card(card: str) -> str:
    """Format a single card to have valid rank and suit."""
    valid_ranks = ["A", "K", "Q", "J", "T"] + [str(n) for n in range(2, 10)]
    valid_suits = ["h", "d", "s", "c"]

    # Convert to lowercase and remove any whitespace
    card = card.lower().strip()

    # Handle empty or invalid cards
    if not card:
        return ""

    # Extract rank and suit
    rank = card[0].upper()  # First character as rank
    suit = card[-1].lower()  # Last character as suit

    # Validate and format
    if rank in valid_ranks and suit in valid_suits:
        return f"{rank}{suit}"
    return ""


def format_cards_array(cards: List[str]) -> List[str]:
    """Format an array of cards, ensuring each card is valid."""
    if not cards:
        return []

    # Format each card and filter out invalid ones
    formatted_cards = [format_card(card) for card in cards]
    valid_cards = [card for card in formatted_cards if card]
    return valid_cards


def format_game_info(game_info: Dict[str, Any]) -> Dict[str, Any]:
    """Format game information according to specific rules."""
    formatted_info = game_info.copy()

    # Format positions to uppercase
    formatted_info["user_position"] = game_info["user_position"].upper()
    formatted_info["opponent_position"] = game_info["opponent_position"].upper()

    # Format cards arrays and validate
    for card_field in ["flop", "turn", "river"]:
        if game_info.get(card_field):
            formatted_info[card_field] = format_cards_array(game_info[card_field])

    # Format actions
    plain_actions = ["Fold", "Call", "Check"]
    formatted_actions = []

    for action in game_info.get("actions", []):
        # Handle Bet/Raise/AllIn with amounts
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

    formatted_info["actions"] = formatted_actions
    return formatted_info


def extract_poker_info(input_text: str) -> Dict[str, Any]:
    game_info_empty = {
        "user_position": "",
        "opponent_position": "",
        "user_hand": "",
        "flop": "",
        "turn": "",
        "river": "",
        "actions": [],
    }

    prompt = f"""Extract the following poker information from the given text and format it as a JSON object:
    - User's position (BTN, SB, BB, etc.)
    - Opponent's position (BTN, SB, BB, etc.)
    - User's hand (2 cards)
    - Flop cards (3 cards)
    - Turn card (1 card, optional)
    - River card (1 card, optional)
    - All Actions (list of actions in forms like Bet(6), Bet(10), Call, etc.)

    Input text: {input_text}

    Return only the JSON object in the following format:
    {{
        "user_position": "position",
        "opponent_position": "position",
        "flop": ["cards1","cards2","cards3"],
        "turn": ["cards4"],
        "river": ["cards5"],
        "actions": ["action1", "action2", ...],
        "user_hand": "cards"
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
            formatted_info = format_game_info(game_info)
            # json.dump(formatted_info, open("game_info.json", "w"))
            return formatted_info
        else:
            return "Error extracting poker information: {str(e)}"

    except Exception as e:
        print(f"Error extracting poker information: {str(e)}")
        return game_info_empty


def main():
    # Example usage
    input_text = input(
        "输入当前信息，请至少包含双方位置、玩家手牌、公共牌面和双方动作："
    )
    result = extract_poker_info(input_text)
    # json.dump(result, open("game_info.json", "w"))
    print("\nExtracted Information:")
    print(result)


if __name__ == "__main__":
    main()
