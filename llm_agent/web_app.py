from flask import Flask, render_template, request, jsonify, session
import json
from llm import prepare_prompt, explain, SYSTEM_PROMPT
from game_info_extractor import extract_poker_info
from poker_text_detector import PokerTextDetector

app = Flask(__name__)
app.secret_key = (
    "sk-8c6fd7cbaf5842d09cdd23be37ba0ecf"  # Required for session management
)


@app.route("/")
def index():
    # Initialize empty conversation history when starting new session
    if "conversation_history" not in session:
        session["conversation_history"] = []
    return render_template(
        "index.html", conversation_history=session["conversation_history"]
    )


@app.route("/process", methods=["POST"])
def process():
    # Get the input text from the form
    input_text = request.json.get("text", "")

    # Get conversation history from session
    conversation_history = session.get("conversation_history", [])

    # Add user message to history
    conversation_history.append({"role": "user", "content": input_text})

    # Only check if poker related on first message
    if len(conversation_history) == 1:
        poker_text_detector = PokerTextDetector()
        if not poker_text_detector.is_poker_related(input_text):
            content = "请输入与德州扑克有关的信息！"
            print(content)
            session["conversation_history"] = []
            return jsonify(
                {
                    "success": True,
                    "result": content,
                    "conversation_history": [],
                }
            )

        try:
            # Extract poker information
            user_data, status_message = extract_poker_info(input_text)

            # Check for errors in extraction
            if "Error" in status_message:
                print(status_message)
                session["conversation_history"] = []
                return jsonify(
                    {
                        "success": True,
                        "result": status_message,
                        "conversation_history": [],
                    }
                )

            # Process the poker information
            prompt = prepare_prompt(user_data)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            content = explain(messages, "deepseek")

        except Exception as e:
            error_message = f"处理过程中出现错误: {str(e)}"
            print(error_message)
            session["conversation_history"] = []
            return jsonify(
                {
                    "success": True,
                    "result": error_message,
                    "conversation_history": [],
                }
            )
    else:
        # Use the entire conversation history as context
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *conversation_history,
        ]
        content = explain(messages, "deepseek")

    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": content})

    # Update session with new history
    session["conversation_history"] = conversation_history

    return jsonify(
        {
            "success": True,
            "result": content,
            "conversation_history": conversation_history,
        }
    )


@app.route("/clear", methods=["POST"])
def clear_history():
    # Clear the conversation history
    session["conversation_history"] = []
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True)
