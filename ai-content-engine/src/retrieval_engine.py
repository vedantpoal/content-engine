import torch
from typing import List, Dict, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline, 
    GenerationConfig
)

class RetrievalEngine:
    SUPPORTED_MODELS = {
        'opt': 'facebook/opt-350m',
        'gpt2': 'gpt2-medium',
        'llama2': 'meta-llama/Llama-2-7b-chat-hf',
        'mistral': 'mistralai/Mistral-7B-Instruct-v0.1'
    }

    def __init__(
        self, 
        model_name: str = 'opt', 
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """
        Initialize retrieval engine with configurable model
        
        Args:
            model_name (str): Predefined model identifier
            max_tokens (int): Maximum token generation length
            temperature (float): Sampling temperature for generation
        """
        self.model_name = model_name.lower()
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self._validate_model()
        self._load_model()
    
    def _validate_model(self):
        """Validate and resolve model name"""
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {self.model_name}. "
                             f"Supported models: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.full_model_name = self.SUPPORTED_MODELS[self.model_name]
    
    def _load_model(self):
        """Load tokenizer and model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.full_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.full_model_name, 
                torch_dtype=torch.float16,
                device_map='auto',
                low_cpu_memory_usage=True
            )
            
            # Fallback to pipeline if direct model loading fails
            if not self.model or not self.tokenizer:
                self.generation_pipeline = pipeline(
                    'text-generation', 
                    model=self.full_model_name
                )
        except Exception as e:
            print(f"Model loading error: {e}")
            self.generation_pipeline = pipeline(
                'text-generation', 
                model='gpt2-medium'  # Fallback model
            )
    
    def generate_query_embedding(
        self, 
        query: str, 
        embedding_model
    ) -> List[float]:
        """
        Generate embedding for query
        
        Args:
            query (str): Input query
            embedding_model: Embedding model
        
        Returns:
            Query embedding
        """
        try:
            return embedding_model.encode([query])[0]
        except Exception as e:
            print(f"Embedding generation error: {e}")
            return [0.0] * 384  # Default empty embedding
    
    def retrieve_context(
        self, 
        query: str, 
        vector_store, 
        embedding_model,
        top_k: int = 3
    ) -> List[str]:
        """
        Retrieve relevant context for query
        
        Args:
            query (str): User query
            vector_store: Vector store manager
            embedding_model: Embedding generator
            top_k (int): Number of context chunks
        
        Returns:
            List of retrieved context chunks
        """
        try:
            query_embedding = self.generate_query_embedding(query, embedding_model)
            results = vector_store.query(query_embedding, top_k)
            return results.get('documents', [[]])[0]
        except Exception as e:
            print(f"Context retrieval error: {e}")
            return []
    
    def generate_response(
        self, 
        query: str, 
        context: List[str],
        max_length: Optional[int] = None
    ) -> str:
        """
        Generate response using retrieved context
        
        Args:
            query (str): User query
            context (List[str]): Retrieved context
            max_length (Optional[int]): Override default max tokens
        
        Returns:
            Generated response
        """
        max_length = max_length or self.max_tokens
        
        prompt = f"""Context: {' '.join(context)}

Question: {query}
Helpful Answer:"""
        
        try:
            # Use pipeline if direct model generation fails
            if hasattr(self, 'generation_pipeline'):
                response = self.generation_pipeline(
                    prompt, 
                    max_length=max_length,
                    temperature=self.temperature
                )[0]['generated_text']
                return response
            
            # Direct model generation
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True
            ).to(self.model.device)
            
            generation_config = GenerationConfig(
                max_new_tokens=max_length,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9
            )
            
            outputs = self.model.generate(
                **inputs, 
                generation_config=generation_config
            )
            
            return self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
        
        except Exception as e:
            print(f"Response generation error: {e}")
            return f"I encountered an error generating a response: {e}"