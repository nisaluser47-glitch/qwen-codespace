from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Qwen Model සහ Tokenizer Load කරගැනීම
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print("⏳ Loading Qwen2.5-0.5B Model into Cloud Memory...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)
print("✅ Qwen Model Loaded Successfully!")

@app.route("/", methods=["GET"])
def home():
    return "Thinext AI Qwen Codespace API is Running Successfully! 🚀"

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Message required"}), 400

        prompt = data.get("message", "")
        
        # Qwen Chat Template සකස් කිරීම
        messages = [
            {"role": "system", "content": "You are Thinext AI. Give helpful, accurate responses in Markdown format."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt")
        
        # Text Generation
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=250,
            temperature=0.6,
            top_p=0.9
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return jsonify({
            "reply": response.strip(),
            "source": "Cloud Qwen Model"
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
