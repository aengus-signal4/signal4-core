#!/usr/bin/env python3
"""
MPS-Enhanced Embedding Utilities
================================

GPU-accelerated embedding generation with MPS (Metal Performance Shaders) support.
Falls back to ONNX models when MPS is not available.
"""

import sys
from pathlib import Path
import logging
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Union
import time

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModel
    from optimum.onnxruntime import ORTModelForFeatureExtraction
except ImportError:
    logger.error("Error: transformers or optimum is not installed.")
    logger.error("Please install them: pip install transformers optimum[onnxruntime]")
    sys.exit(1)

def _mean_pooling(model_output, attention_mask):
    """Performs mean pooling on the last hidden state using the attention mask."""
    if hasattr(model_output, 'last_hidden_state'):
        token_embeddings = model_output.last_hidden_state
    elif isinstance(model_output, (dict, tuple)) and len(model_output) > 0:
        token_embeddings = model_output[0]
    else:
        raise TypeError("Model output structure not recognized for mean pooling.")
    
    if not isinstance(token_embeddings, torch.Tensor):
        raise TypeError(f"Expected token_embeddings to be a torch.Tensor, got {type(token_embeddings)}")
    
    # Ensure attention_mask is on the same device as token_embeddings
    if attention_mask.device != token_embeddings.device:
        attention_mask = attention_mask.to(token_embeddings.device)
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class MPSEmbeddingGenerator:
    """
    Enhanced embedding generator with MPS support.
    Uses PyTorch models directly when MPS is available, falls back to ONNX otherwise.
    """
    
    def __init__(self, config_path: Path, model_type: str = "e5"):
        """
        Initialize the embedding generator with MPS support.
        
        Args:
            config_path: Path to the main YAML configuration file.
            model_type: Type of model to load ("e5" or "xlm-r")
        """
        self.config = self._load_config(config_path)
        self.model_type = model_type
        
        if model_type == "e5":
            embedding_config = self.config.get('embedding', {})
            self.model_name_or_path = embedding_config.get('model_id', 'intfloat/e5-large-v2')
            self.local_model_path = embedding_config.get('local_model_path')
            # Check for PyTorch model first
            base_path = Path(self.local_model_path).parent if self.local_model_path else None
            if base_path and (base_path / "pytorch").exists():
                self.pytorch_model_path = str(base_path / "pytorch")
            else:
                self.pytorch_model_path = None
            self.embedding_dimension = embedding_config.get('dimension', 1024)
        else:  # xlm-r for similarity
            similarity_config = self.config.get('embedding', {}).get('stitch_processing', {}).get('semantic_similarity', {})
            self.model_name_or_path = similarity_config.get('model_id', 'sentence-transformers/paraphrase-xlm-r-multilingual-v1')
            self.local_model_path = similarity_config.get('onnx_model_path')
            # Check for PyTorch model first
            base_path = Path(self.local_model_path).parent if self.local_model_path else None
            if base_path and (base_path / "pytorch").exists():
                self.pytorch_model_path = str(base_path / "pytorch")
            else:
                self.pytorch_model_path = None
            self.embedding_dimension = similarity_config.get('dimension', 768)
        
        self.tokenizer = None
        self.model = None
        self.device = None
        self.use_pytorch = False
        
        self._setup_model_and_tokenizer()
    
    def _load_config(self, config_path: Path) -> dict:
        """Loads the YAML configuration file."""
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    
    def _setup_model_and_tokenizer(self):
        """Load model and tokenizer with MPS support."""
        # Determine device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.use_pytorch = True
            logger.info(f"MPS (Metal GPU) detected! Using PyTorch model for {self.model_type}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.use_pytorch = True
            logger.info(f"CUDA GPU detected! Using PyTorch model for {self.model_type}")
        else:
            self.device = torch.device("cpu")
            self.use_pytorch = False
            logger.info(f"No GPU detected, using ONNX model for {self.model_type}")
        
        # Load tokenizer
        try:
            # Try loading from local path first
            if self.local_model_path and Path(self.local_model_path).exists():
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
                logger.info(f"Loaded tokenizer from local path: {self.local_model_path}")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
                logger.info(f"Loaded tokenizer from: {self.model_name_or_path}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Load model
        if self.use_pytorch:
            # Load PyTorch model for GPU acceleration
            try:
                logger.info(f"Loading PyTorch model for {self.model_type}...")
                
                # Try PyTorch path first, then local path, then model ID
                if self.pytorch_model_path and Path(self.pytorch_model_path).exists():
                    model_path = self.pytorch_model_path
                    logger.info(f"Using PyTorch model from: {model_path}")
                elif self.local_model_path and Path(self.local_model_path).exists():
                    model_path = self.local_model_path
                    logger.info(f"Using local model from: {model_path}")
                else:
                    model_path = self.model_name_or_path
                    logger.info(f"Downloading model from HuggingFace: {model_path}")
                
                self.model = AutoModel.from_pretrained(model_path)
                self.model = self.model.to(self.device)
                self.model.eval()  # Set to evaluation mode
                
                logger.info(f"Successfully loaded PyTorch model on {self.device}")
                
                # Run a test to ensure MPS is working
                if self.device.type == "mps":
                    self._test_mps()
                    
            except Exception as e:
                logger.error(f"Failed to load PyTorch model, falling back to ONNX: {e}")
                self.use_pytorch = False
                self.device = torch.device("cpu")
        
        # Fall back to ONNX if PyTorch loading failed or CPU only
        if not self.use_pytorch:
            self._load_onnx_model()
    
    def _test_mps(self):
        """Test that MPS is working correctly."""
        try:
            with torch.no_grad():
                test_input = torch.randn(1, 10, self.embedding_dimension).to(self.device)
                _ = test_input.mean()
            logger.info("MPS test successful - GPU acceleration is working!")
        except Exception as e:
            logger.warning(f"MPS test failed, falling back to CPU: {e}")
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
    
    def _load_onnx_model(self):
        """Load ONNX model for CPU inference."""
        if not self.local_model_path or not Path(self.local_model_path).exists():
            logger.error(f"ONNX model path not found: {self.local_model_path}")
            raise FileNotFoundError(f"ONNX model not found at: {self.local_model_path}")
        
        try:
            logger.info(f"Loading ONNX model from: {self.local_model_path}")
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                self.local_model_path,
                provider="CPUExecutionProvider"
            )
            logger.info("Successfully loaded ONNX model")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
        """
        Generate embeddings for a list of texts with GPU acceleration when available.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (larger for GPU)
            
        Returns:
            Numpy array of embeddings or None if generation fails
        """
        if not texts:
            return None
        
        # Adjust batch size based on device
        if self.device.type in ["mps", "cuda"]:
            batch_size = min(64, batch_size * 2)  # Larger batches for GPU
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Tokenize
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                )
                
                # Move to device if using PyTorch
                if self.use_pytorch:
                    encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    start_time = time.time()
                    model_output = self.model(**encoded_input)
                    
                    # Perform mean pooling
                    embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])
                    
                    # Normalize embeddings
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                    # Convert to CPU numpy
                    if self.use_pytorch:
                        embeddings = embeddings.cpu().numpy()
                    else:
                        embeddings = embeddings.numpy()
                    
                    elapsed = time.time() - start_time
                    logger.debug(f"Batch {i//batch_size + 1}: Generated {len(batch_texts)} embeddings in {elapsed:.3f}s on {self.device}")
                
                all_embeddings.append(embeddings)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                return None
        
        # Concatenate all batches
        final_embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated {len(final_embeddings)} embeddings on {self.device} ({self.model_type})")
        return final_embeddings
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embeddings = self.generate_embeddings([text1, text2])
        if embeddings is None or len(embeddings) != 2:
            return 0.0
        
        # Cosine similarity (embeddings are already normalized)
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)
    
    def generate_similarity_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Alias for generate_embeddings to match the interface expected by segment_embeddings.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.generate_embeddings(texts)