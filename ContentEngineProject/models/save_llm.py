from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model to download
model_name = "distilgpt2"  # Replace with another model if needed
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer locally
model.save_pretrained('models/local_llm')
tokenizer.save_pretrained('models/local_llm')
print(f"Model '{model_name}' saved successfully in 'models/local_llm'")
