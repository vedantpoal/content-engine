from sentence_transformers import SentenceTransformer

# Load and save the model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
model.save('models/sentence_transformer')
print(f"Model '{model_name}' saved successfully in 'models/sentence_transformer'")
