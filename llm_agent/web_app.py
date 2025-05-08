from flask import Flask, render_template, request, jsonify
import json
from llm import prepare_prompt, explain, SYSTEM_PROMPT
from game_info_extractor import extract_poker_info

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    try:
        # Get the input text from the form
        input_text = request.json.get("text", "")

        # Process the input using extract_poker_info
        user_data = extract_poker_info(input_text)

        # Process the input using the existing functions
        prompt = prepare_prompt(user_data)

        # Get the explanation from the LLM
        content = explain(
            sys_prompt=SYSTEM_PROMPT, user_prompt=prompt, model="deepseek"
        )

        return jsonify({"success": True, "result": content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
