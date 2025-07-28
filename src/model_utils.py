import os
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import LEDTokenizer, LEDForConditionalGeneration

class ModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.logger = logging.getLogger(__name__)
        
        # Set device to CPU only
        self.device = torch.device('cpu')
        
    def load_all_models(self):
        """Load all required models for Round 1B"""
        try:
            self.load_cross_encoder()
            self.load_led_model()
            self.load_embedding_model()
            self.logger.info("All models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            # Don't raise - use fallback methods instead
            self.logger.info("Using fallback methods without ML models")
        
    def load_embedding_model(self):
        """Load embedding model for semantic similarity"""
        try:
            # Get HF token from environment
            hf_token = os.environ.get('HF_TOKEN', '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')
            
            # Use sentence-transformers with HF token for secure access
            self.models["embedding"] = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                token=hf_token
            )
            self.logger.info("Embedding model loaded from sentence-transformers with HF token")
        except Exception as e:
            self.logger.warning(f"Failed to load embedding model: {e}")
            # Fallback without token
            try:
                self.models["embedding"] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                self.logger.info("Embedding model loaded from sentence-transformers (fallback)")
            except Exception as e2:
                self.logger.error(f"Failed to load embedding model (fallback): {e2}")
                self.models["embedding"] = None
        

        
    def load_cross_encoder(self):
        """Load Cross-Encoder model for relevance scoring"""
        try:
            model_path = self.models_dir / "ce_minilm_l6_int8.pt"
            if model_path.exists():
                # Load quantized cross-encoder model
                self.models["cross_encoder"] = torch.load(str(model_path), map_location='cpu')
                self.logger.info("Cross-encoder model loaded successfully")
            else:
                # Fallback to sentence-transformers
                try:
                    self.models["cross_encoder"] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                    self.logger.info("Cross-encoder model loaded from sentence-transformers")
                except Exception as e:
                    self.logger.warning(f"Failed to load cross-encoder model: {e}")
                    self.models["cross_encoder"] = None
        except Exception as e:
            self.logger.error(f"Failed to load cross-encoder model: {e}")
            self.models["cross_encoder"] = None
        
    def load_led_model(self):
        """Load LED model for text summarization"""
        try:
            model_path = self.models_dir / "led_base16k_int8.pt"
            if model_path.exists():
                # Load quantized LED model
                self.models["led"] = torch.load(str(model_path), map_location='cpu')
                self.logger.info("LED model loaded successfully")
            else:
                # Fallback to transformers
                try:
                    self.models["led"] = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
                    self.models["led_tokenizer"] = LEDTokenizer.from_pretrained('allenai/led-base-16384')
                    self.logger.info("LED model loaded from transformers")
                except Exception as e:
                    self.logger.warning(f"Failed to load LED model: {e}")
                    self.models["led"] = None
                    self.models["led_tokenizer"] = None
        except Exception as e:
            self.logger.error(f"Failed to load LED model: {e}")
            self.models["led"] = None
            self.models["led_tokenizer"] = None
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using the loaded model"""
        if self.models.get("embedding") is not None:
            try:
                embedding = self.models["embedding"].encode(text)
                return embedding
            except Exception as e:
                self.logger.warning(f"Failed to get embedding: {e}")
        
        # Fallback: return zero vector
        return np.zeros(384)
    
    def rank_sections(self, sections: List[Dict], query: str) -> List[Dict]:
        """Rank sections based on relevance to query"""
        if not sections:
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Calculate similarity scores
        for section in sections:
            section_embedding = self.get_embedding(section["text"])
            similarity = np.dot(query_embedding, section_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(section_embedding) + 1e-8
            )
            section["relevance_score"] = float(similarity)
        
        # Sort by relevance score
        sections.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return sections
    
    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """Generate summary using LED model with enhanced fallback"""
        if not text or len(text.strip()) < 10:
            return text
        
        try:
            if self.models.get("led") is not None and self.models.get("led_tokenizer") is not None:
                # Use LED model for summarization
                inputs = self.models["led_tokenizer"](
                    text, 
                    max_length=1024, 
                    truncation=True, 
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    outputs = self.models["led"].generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=2
                    )
                
                summary = self.models["led_tokenizer"].decode(outputs[0], skip_special_tokens=True)
                
                # Validate summary quality
                if len(summary.strip()) < 20:
                    self.logger.warning("LED summary too short, using fallback")
                    return self._generate_fallback_summary(text, max_length)
                
                return summary
            else:
                # Enhanced fallback summarization
                return self._generate_fallback_summary(text, max_length)
                
        except Exception as e:
            self.logger.warning(f"Failed to generate summary: {e}")
            # Enhanced fallback: return meaningful summary
            return self._generate_fallback_summary(text, max_length)
    
    def _generate_fallback_summary(self, text: str, max_length: int) -> str:
        """Generate fallback summary using extractive methods"""
        try:
            # Split into sentences
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Filter meaningful sentences (not too short, not too long)
            meaningful_sentences = []
            for sentence in sentences:
                if 20 <= len(sentence) <= 200:
                    meaningful_sentences.append(sentence)
            
            if meaningful_sentences:
                # Take first 3-4 meaningful sentences
                summary = '. '.join(meaningful_sentences[:4]) + '.'
                return summary[:max_length]
            else:
                # If no good sentences, create bullet points from lines
                lines = text.split('\n')
                key_points = []
                for line in lines[:6]:
                    line = line.strip()
                    if len(line) > 15 and not line.startswith(('•', '-', '*', '1.', '2.')):
                        key_points.append(f"• {line}")
                
                if key_points:
                    summary = '\n'.join(key_points[:4])
                    return summary[:max_length]
                else:
                    # Last resort: return first part
                    return text[:max_length]
                    
        except Exception as e:
            self.logger.warning(f"Fallback summary failed: {e}")
            return text[:max_length]
    
 