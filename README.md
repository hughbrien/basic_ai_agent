
# OpenAI
python chatbox.py --provider openai   --model gpt-4o-mini   --session dev

# Anthropic
python chatbox.py --provider anthropic --model claude-3-5-sonnet --session dev

# Groq (OpenAI-compatible tool calling with Groq models)
python chatbox.py --provider groq     --model llama-3.1-70b   --session dev

# Ollama (local Llama 3)
python chatbox.py --provider ollama   --model llama3:latest    --session dev

# Grok (xAI; OpenAI-compatible endpoint)
python chatbox.py --provider grok     --model grok-2           --session dev
