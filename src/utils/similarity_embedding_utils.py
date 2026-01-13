#!/usr/bin/env python3
"""
Similarity Embedding Utilities
===============================

Utilities for generating embeddings specifically for semantic similarity tasks
using the XLM-R multilingual similarity model. This is used during beam search
segmentation to find optimal semantic boundaries.

Separate from the main embedding_utils.py which handles final embeddings.
"""

import logging
import sys
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

logger = logging.getLogger(__name__)

def _mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings using attention mask.
    Same implementation as in embedding_utils.py for consistency.
    """
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SimilarityEmbeddingGenerator:
    """Handles loading and using the ONNX similarity model for semantic similarity tasks."""

    def __init__(self, config_path: Path):
        """
        Initialize the SimilarityEmbeddingGenerator using the stitch_processing.semantic_similarity config.

        Args:
            config_path: Path to the main YAML configuration file.
        """
        self.config = self._load_config(config_path)
        similarity_config = self.config.get('embedding', {}).get('stitch_processing', {}).get('semantic_similarity', {})

        if not similarity_config:
            raise ValueError("No semantic_similarity configuration found in embedding.stitch_processing section")

        self.local_model_path = similarity_config.get('onnx_model_path')
        self.tokenizer_id = similarity_config.get('tokenizer_id')
        self.embedding_dimension = similarity_config.get('dimension', 768)

        self.tokenizer = None
        self.embedding_model = None
        self.device = None

        self._setup_model_and_tokenizer(similarity_config)

    def _load_config(self, config_path: Path) -> dict:
        """Load the YAML configuration file."""
        if not config_path.is_file():
            logger.error(f"Configuration file not found at: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load or parse config file {config_path}: {e}", exc_info=True)
            raise

    def _setup_model_and_tokenizer(self, similarity_config: dict):
        """Load the tokenizer and ONNX model for similarity tasks."""
        if not self.local_model_path:
            logger.error("Missing 'onnx_model_path' in semantic_similarity config section.")
            raise ValueError("Similarity model path not configured.")

        local_model_path_obj = Path(self.local_model_path)
        if not local_model_path_obj.exists() or not local_model_path_obj.is_dir():
            logger.error(f"Configured similarity model path does not exist or is not a directory: {self.local_model_path}")
            raise FileNotFoundError(f"Local similarity model directory not found: {self.local_model_path}")

        # Device configuration - force CPU for stability with ONNX
        self.device = 'cpu'
        logger.info(f"Selected device for similarity operations: '{self.device}'")

        # Load Tokenizer
        try:
            logger.info(f"Attempting to load similarity tokenizer from local path: {local_model_path_obj}...")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path_obj)
            logger.info("Similarity tokenizer loaded successfully from local path.")
        except Exception as e_local_tok:
            logger.warning(f"Failed to load tokenizer from local path ({local_model_path_obj}): {e_local_tok}.")
            if self.tokenizer_id:
                logger.info(f"Falling back to loading tokenizer from ID: {self.tokenizer_id}...")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
                    logger.info(f"Similarity tokenizer loaded successfully from ID: {self.tokenizer_id}.")
                except Exception as e_id_tok:
                    logger.error(f"Failed to load tokenizer from ID ({self.tokenizer_id}): {e_id_tok}", exc_info=True)
                    raise ValueError("Could not load similarity tokenizer locally or from configured ID.")
            else:
                logger.error("Cannot load similarity tokenizer: Not found locally and no tokenizer_id configured.")
                raise ValueError("Cannot load similarity tokenizer: Not found locally and no tokenizer_id configured.")

        # Load ONNX Model
        try:
            logger.info(f"Attempting to load ONNX similarity model from local path: {local_model_path_obj}...")
            
            # Determine the ONNX model filename
            onnx_quant_file = local_model_path_obj / "model_dynamic_quantized.onnx"
            onnx_base_file = local_model_path_obj / "model.onnx"
            onnx_filename = None

            if onnx_quant_file.exists():
                onnx_filename = onnx_quant_file.name
                logger.info(f"Found quantized similarity model file: {onnx_filename}")
            elif onnx_base_file.exists():
                onnx_filename = onnx_base_file.name
                logger.info(f"Found base ONNX similarity model file: {onnx_filename}")
            else:
                potential_files = list(local_model_path_obj.glob("*.onnx"))
                if len(potential_files) == 1:
                    onnx_filename = potential_files[0].name
                    logger.warning(f"Using automatically detected ONNX file: {onnx_filename}")
                elif len(potential_files) > 1:
                    logger.error(f"Multiple .onnx files found in {local_model_path_obj}: {[f.name for f in potential_files]}. Please ensure only one model file exists.")
                    raise FileNotFoundError(f"Ambiguous ONNX model files in {local_model_path_obj}")
                else:
                    logger.error(f"Could not find 'model.onnx' or 'model_dynamic_quantized.onnx' in {local_model_path_obj}")
                    raise FileNotFoundError(f"No ONNX similarity model file found in {local_model_path_obj}")

            # Set up ONNX Runtime providers - force CPU for similarity model
            providers = ['CPUExecutionProvider']
            provider_options = {}
            logger.info(f"Using ONNX Runtime providers for similarity: {providers}")

            self.embedding_model = ORTModelForFeatureExtraction.from_pretrained(
                local_model_path_obj,
                file_name=onnx_filename,
                provider=providers[0],
                provider_options=provider_options,
            )

            logger.info(f"ONNX Similarity model loaded successfully using provider '{providers[0]}'.")

        except Exception as e:
            logger.error(f"Failed to load local ONNX similarity model from '{local_model_path_obj}': {e}", exc_info=True)
            self.embedding_model = None
            raise

    def generate_similarity_embeddings(self, texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
        """
        Generate embeddings specifically for similarity calculations with batch processing.
        
        Note: This does NOT add 'passage:' prefix like the E5 model, as XLM-R similarity
        models typically don't use such prefixes.

        Args:
            texts: A list of strings to embed.
            batch_size: Number of texts to process in each batch (default 32)

        Returns:
            A numpy array of embeddings (shape: [num_texts, embedding_dimension]),
            or None if embedding fails or the model is not loaded.
        """
        if not self.embedding_model or not self.tokenizer:
            logger.error("Similarity embedding model or tokenizer not available. Cannot generate embeddings.")
            return None
        if not texts:
            return np.array([])

        logger.info(f"Generating similarity embeddings for {len(texts)} texts in batches of {batch_size}...")

        try:
            all_embeddings = []
            
            # Import tqdm for progress tracking
            try:
                from tqdm import tqdm
                progress_bar = tqdm(range(0, len(texts), batch_size), desc="Similarity embeddings")
            except ImportError:
                logger.warning("tqdm not available, proceeding without progress bar")
                progress_bar = range(0, len(texts), batch_size)
            
            for i in progress_bar:
                batch_texts = texts[i:i + batch_size]
                
                # XLM-R similarity models typically don't use prefixes
                # Tokenize the batch texts directly
                encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')

                # Ensure inputs are on CPU for ONNX Runtime
                if any(t.device.type != 'cpu' for t in encoded_input.values()):
                    logger.debug("Moving tokenized input to CPU for ONNX Runtime.")
                    encoded_input = {k: v.to('cpu') for k, v in encoded_input.items()}

                # Run inference with the ONNX model
                with torch.no_grad():
                    model_output = self.embedding_model(**encoded_input)

                # Perform mean pooling using attention mask
                attention_mask = encoded_input['attention_mask']
                sentence_embeddings = _mean_pooling(model_output, attention_mask)

                # Normalize embeddings to unit length (L2 norm)
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

                # Move embeddings to CPU before converting to numpy
                if sentence_embeddings.device.type != 'cpu':
                    sentence_embeddings = sentence_embeddings.cpu()

                all_embeddings.append(sentence_embeddings.numpy())
                
                # Update progress description with batch info
                if hasattr(progress_bar, 'set_description'):
                    progress_bar.set_description(f"Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            # Concatenate all batch embeddings
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            logger.info(f"Successfully generated similarity embeddings with shape: {final_embeddings.shape}")
            return final_embeddings

        except Exception as e:
            logger.error(f"Error during similarity embedding generation: {e}", exc_info=True)
            return None

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            embeddings = self.generate_similarity_embeddings([text1, text2])
            if embeddings is None or len(embeddings) < 2:
                return 0.0

            emb1 = embeddings[0]
            emb2 = embeddings[1]
            
            # Calculate cosine similarity (embeddings are already normalized)
            similarity = np.dot(emb1, emb2)
            return float(similarity)

        except Exception as e:
            logger.warning(f"Error calculating similarity between texts: {e}")
            return 0.0

    def calculate_similarity_batch(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """
        Calculate similarity between batches of texts.

        Args:
            texts1: First batch of texts
            texts2: Second batch of texts

        Returns:
            Matrix of similarities with shape [len(texts1), len(texts2)]
        """
        try:
            # Generate embeddings for both batches
            emb1 = self.generate_similarity_embeddings(texts1)
            emb2 = self.generate_similarity_embeddings(texts2)
            
            if emb1 is None or emb2 is None:
                return np.zeros((len(texts1), len(texts2)))

            # Calculate cosine similarity matrix (embeddings are already normalized)
            similarities = np.dot(emb1, emb2.T)
            return similarities

        except Exception as e:
            logger.error(f"Error calculating batch similarities: {e}")
            return np.zeros((len(texts1), len(texts2)))

# Test functionality
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Testing SimilarityEmbeddingGenerator...")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    config_file = project_root / "config" / "config.yaml"

    if not config_file.exists():
        logger.error(f"Test mode: config.yaml not found at expected location: {config_file}")
        sys.exit(1)

    try:
        sim_gen = SimilarityEmbeddingGenerator(config_file)

        if sim_gen.embedding_model and sim_gen.tokenizer:
            logger.info("SimilarityEmbeddingGenerator initialized successfully.")

            test_texts = [
                "This is a sentence about cats.",
                "This is a sentence about dogs.",
                "Cats are furry animals.",
                "The weather is nice today.",
                "It's raining outside."
            ]
            
            embeddings = sim_gen.generate_similarity_embeddings(test_texts)

            if embeddings is not None:
                logger.info(f"Generated similarity embeddings shape: {embeddings.shape}")
                
                # Test pairwise similarities
                sim1 = sim_gen.calculate_similarity(test_texts[0], test_texts[2])  # cats <-> cats
                sim2 = sim_gen.calculate_similarity(test_texts[0], test_texts[3])  # cats <-> weather
                
                logger.info(f"Similarity (cats <-> cats): {sim1:.3f}")
                logger.info(f"Similarity (cats <-> weather): {sim2:.3f}")
                
                # Test batch similarities
                batch_sims = sim_gen.calculate_similarity_batch(test_texts[:2], test_texts[2:])
                logger.info(f"Batch similarities shape: {batch_sims.shape}")
                
            else:
                logger.error("Failed to generate similarity embeddings during test.")
        else:
            logger.error("Failed to initialize SimilarityEmbeddingGenerator model/tokenizer during test.")

    except Exception as main_e:
        logger.error(f"Error during SimilarityEmbeddingGenerator test: {main_e}", exc_info=True) 