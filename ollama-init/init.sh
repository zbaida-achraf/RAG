#!/bin/sh

ollama serve &

OLLAMA_PID=$!

echo "⏳ Attente de disponibilité d'Ollama..."
until ollama list >/dev/null 2>&1; do
  sleep 2
done

echo "📥 Pulling models..."
ollama pull mistral
ollama pull llama3.1
ollama pull llama3.2
ollama pull deepseek-r1:7b
ollama pull deepseek-r1:8b
ollama pull wizardlm2
ollama pull llama3-chatqa
echo "✅ Models pulled."

wait $OLLAMA_PID