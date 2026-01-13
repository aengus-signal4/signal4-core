#!/usr/bin/env python3
"""
GPU-Accelerated Embedding Utilities
===================================

Simple, direct PyTorch implementation for GPU-accelerated embeddings.
No ONNX, just pure PyTorch with MPS/CUDA support.
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
except ImportError:
    logger.error("Error: transformers is not installed.")
    logger.error("Please install: pip install transformers torch")
    sys.exit(1)

def _mean_pooling(model_output, attention_mask):
    """Performs mean pooling on the last hidden state using the attention mask."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class GPUEmbeddingGenerator:
    """
    Simple GPU-accelerated embedding generator using PyTorch models directly.
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None, hf_token: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device to use ('mps', 'cuda', 'cpu', or None for auto-detect)
            hf_token: HuggingFace API token for authentication
        """
        self.model_name = model_name
        self.hf_token = hf_token
        self._setup_device(device)
        self._load_model()
        
    def _setup_device(self, device: Optional[str]):
        """Set up the compute device."""
        if device:
            self.device = torch.device(device)
        else:
            # Auto-detect best available device
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS (Metal GPU) for acceleration")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA GPU for acceleration")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU (no GPU detected)")
    
    def _load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            use_auth_token=self.hf_token
        )
        
        # Load model
        self.model = AutoModel.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_token
        )
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_input = self.tokenizer("test", return_tensors='pt')
            dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
            dummy_output = self.model(**dummy_input)
            self.embedding_dimension = dummy_output.last_hidden_state.shape[-1]
        
        logger.info(f"Model loaded successfully on {self.device} (dim={self.embedding_dimension})")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Adjust batch size based on device
        if self.device.type in ["mps", "cuda"]:
            batch_size = min(64, batch_size * 2)  # Larger batches for GPU
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            if show_progress:
                batch_num = i // batch_size + 1
                logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            
            # Move to device
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
                embeddings = embeddings.cpu().numpy()
                
                elapsed = time.time() - start_time
                if show_progress:
                    logger.debug(f"Generated {len(batch_texts)} embeddings in {elapsed:.3f}s")
            
            all_embeddings.append(embeddings)
        
        # Concatenate all batches
        final_embeddings = np.vstack(all_embeddings)
        return final_embeddings
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embeddings = self.generate_embeddings([text1, text2])
        if len(embeddings) != 2:
            return 0.0
        
        # Cosine similarity (embeddings are already normalized)
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)
    
    # Alias for compatibility
    def generate_similarity_embeddings(self, texts: List[str]) -> np.ndarray:
        """Alias for generate_embeddings to match expected interface."""
        return self.generate_embeddings(texts)


class DualModelEmbeddingSegmenter:
    """
    Embedding segmenter using two models:
    - E5 for final embeddings
    - XLM-R for similarity calculations during segmentation
    """
    
    def __init__(self, config_path: Path):
        """Initialize with configuration."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Get HF token from config
        hf_token = self.config.get('processing', {}).get('hf_token')
        
        # Initialize E5 model for final embeddings
        e5_model = self.config.get('embedding', {}).get('model_id', 'intfloat/multilingual-e5-large-instruct')
        self.e5_generator = GPUEmbeddingGenerator(e5_model, hf_token=hf_token)
        
        # Initialize XLM-R model for similarity
        xlmr_model = self.config.get('embedding', {}).get('stitch_processing', {}).get('semantic_similarity', {}).get(
            'model_id', 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
        )
        self.xlmr_generator = GPUEmbeddingGenerator(xlmr_model, hf_token=hf_token)
        
        logger.info(f"Initialized dual-model setup:")
        logger.info(f"  E5: {e5_model} on {self.e5_generator.device}")
        logger.info(f"  XLM-R: {xlmr_model} on {self.xlmr_generator.device}")
    
    @property
    def embed_generator(self):
        """E5 generator for final embeddings."""
        return self.e5_generator
    
    @property
    def similarity_generator(self):
        """XLM-R generator for similarity calculations."""
        return self.xlmr_generator