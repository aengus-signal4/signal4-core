#!/usr/bin/env python3
import sys
from pathlib import Path
import logging
import yaml
import numpy as np
import torch
import torch.nn.functional as F # For normalization
from typing import Optional

# Configure logger for the utility
logger = logging.getLogger(__name__) # Use standard logging

# Attempt to import ONNX runtime and transformers
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
except ImportError:
    logger.error("Error: optimum, transformers, or torch is not installed.")
    logger.error("Please install them: pip install optimum[onnxruntime] transformers torch")
    # Or pip install optimum[onnxruntime-gpu] transformers torch for CUDA
    sys.exit(1)

# Helper function (can be outside the class or private)
def _mean_pooling(model_output, attention_mask):
    """Performs mean pooling on the last hidden state using the attention mask."""
    # Check if model_output has last_hidden_state attribute
    if not hasattr(model_output, 'last_hidden_state'):
        logger.error("Model output object does not have 'last_hidden_state'. Output keys: %s", list(getattr(model_output, 'keys', lambda: [])()))
        # Fallback or raise: Try accessing the first element if it's a dict/tuple
        if isinstance(model_output, (dict, tuple)) and len(model_output) > 0:
             try:
                  token_embeddings = model_output[0]
                  logger.warning("Accessing model output by index [0] as fallback.")
             except (KeyError, IndexError):
                  raise TypeError("Model output structure not recognized for mean pooling.")
        else:
             raise TypeError("Model output structure not recognized for mean pooling.")
    else:
        token_embeddings = model_output.last_hidden_state

    if not isinstance(token_embeddings, torch.Tensor):
        raise TypeError(f"Expected token_embeddings to be a torch.Tensor, got {type(token_embeddings)}")
    if not isinstance(attention_mask, torch.Tensor):
         raise TypeError(f"Expected attention_mask to be a torch.Tensor, got {type(attention_mask)}")

    # Ensure attention_mask is on the same device as token_embeddings
    if attention_mask.device != token_embeddings.device:
         attention_mask = attention_mask.to(token_embeddings.device)

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class EmbeddingGenerator:
    """Handles loading and using the ONNX embedding model."""

    def __init__(self, config_path: Path):
        """
        Initializes the EmbeddingGenerator by loading the configuration,
        determining the device, and loading the ONNX model and tokenizer.

        Args:
            config_path: Path to the main YAML configuration file.
        """
        self.config = self._load_config(config_path)
        embedding_config = self.config.get('embedding', {})

        self.local_model_path = embedding_config.get('local_model_path')
        self.tokenizer_id = embedding_config.get('tokenizer_id')
        self.embedding_dimension = embedding_config.get('dimension', 1024)

        self.tokenizer = None
        self.embedding_model = None
        self.device = None

        self._setup_model_and_tokenizer(embedding_config)

    def _load_config(self, config_path: Path) -> dict:
        """Loads the YAML configuration file."""
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

    def _setup_model_and_tokenizer(self, embedding_config: dict):
        """Loads the tokenizer and ONNX model based on the configuration."""
        if not self.local_model_path:
            logger.error("Missing 'local_model_path' in embedding config section.")
            raise ValueError("Embedding model path not configured.")

        local_model_path_obj = Path(self.local_model_path)
        if not local_model_path_obj.exists() or not local_model_path_obj.is_dir():
            logger.error(f"Configured local_model_path does not exist or is not a directory: {self.local_model_path}")
            logger.error("Ensure the directory exists, e.g., by running scripts/export_onnx_model.py.")
            raise FileNotFoundError(f"Local model directory not found or invalid: {self.local_model_path}")

        # --- Determine Device ---
        embedding_device_preference = embedding_config.get('device', 'auto')
        if embedding_device_preference == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("MPS device detected, but forcing CPU for ONNX Runtime stability/compatibility.")
                self.device = 'cpu' # Force CPU for stability with current optimum/onnxruntime
            else:
                self.device = 'cpu'
        elif embedding_device_preference == 'mps':
            logger.warning("MPS device requested in config, but forcing CPU for ONNX Runtime stability/compatibility.")
            self.device = 'cpu' # Force CPU if MPS is explicitly requested
        elif embedding_device_preference in ['cuda', 'cuda:0', 'cpu']:
            self.device = embedding_device_preference
            if self.device == 'cuda' and not torch.cuda.is_available():
                 logger.warning("CUDA requested but not available, falling back to CPU.")
                 self.device = 'cpu'
        else:
            logger.warning(f"Unsupported embedding device '{embedding_device_preference}' requested. Falling back to CPU.")
            self.device = 'cpu'
        logger.info(f"Selected device for embedding operations: '{self.device}'")

        # --- Load Tokenizer ---
        try:
            logger.info(f"Attempting to load tokenizer from local path: {local_model_path_obj}...")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path_obj)
            logger.info("Tokenizer loaded successfully from local path.")
        except Exception as e_local_tok:
            logger.warning(f"Failed to load tokenizer from local path ({local_model_path_obj}): {e_local_tok}.")
            if self.tokenizer_id:
                logger.info(f"Falling back to loading tokenizer from ID: {self.tokenizer_id}...")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
                    logger.info(f"Tokenizer loaded successfully from ID: {self.tokenizer_id}.")
                except Exception as e_id_tok:
                    logger.error(f"Failed to load tokenizer from ID ({self.tokenizer_id}): {e_id_tok}", exc_info=True)
                    raise ValueError("Could not load tokenizer locally or from configured ID.")
            else:
                logger.error("Cannot load tokenizer: Not found locally and no tokenizer_id configured.")
                raise ValueError("Cannot load tokenizer: Not found locally and no tokenizer_id configured.")

        # --- Load ONNX Model ---
        try:
            logger.info(f"Attempting to load ONNX embedding model from local path: {local_model_path_obj}...")
            # Determine the ONNX model filename
            onnx_quant_file = local_model_path_obj / "model_dynamic_quantized.onnx"
            onnx_base_file = local_model_path_obj / "model.onnx"
            onnx_filename = None

            if onnx_quant_file.exists():
                onnx_filename = onnx_quant_file.name
                logger.info(f"Found quantized model file: {onnx_filename}")
            elif onnx_base_file.exists():
                onnx_filename = onnx_base_file.name
                logger.info(f"Found base ONNX model file: {onnx_filename}")
            else:
                potential_files = list(local_model_path_obj.glob("*.onnx"))
                if len(potential_files) == 1:
                    onnx_filename = potential_files[0].name
                    logger.warning(f"Using automatically detected ONNX file: {onnx_filename}")
                elif len(potential_files) > 1:
                     logger.error(f"Multiple .onnx files found in {local_model_path_obj}: {[f.name for f in potential_files]}. Please ensure only one model file ('model.onnx' or 'model_dynamic_quantized.onnx') exists.")
                     raise FileNotFoundError(f"Ambiguous ONNX model files in {local_model_path_obj}")
                else:
                    logger.error(f"Could not find 'model.onnx' or 'model_dynamic_quantized.onnx' (or any .onnx file) in {local_model_path_obj}")
                    raise FileNotFoundError(f"No ONNX model file found in {local_model_path_obj}")

            # Set up ONNX Runtime providers based on the determined device
            providers = None
            provider_options = {}
            if self.device.startswith('cuda'):
                # Assume cuda:0 for simplicity if 'cuda' is specified
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                # Extract device ID if specified, e.g., cuda:1
                try:
                     device_id = int(self.device.split(':')[-1]) if ':' in self.device else 0
                except ValueError:
                     logger.warning(f"Invalid CUDA device format '{self.device}', defaulting to device 0.")
                     device_id = 0
                provider_options={'device_id': device_id}
                logger.info(f"Using ONNX Runtime providers: {providers} with options: {provider_options}")
            else: # Default to CPU
                providers = ['CPUExecutionProvider']
                logger.info(f"Using ONNX Runtime providers: {providers}")
                # Ensure self.device reflects CPU usage if it wasn't explicitly 'cpu'
                if self.device != 'cpu':
                     logger.warning(f"Device set to '{self.device}', but forcing CPUExecutionProvider.")
                     self.device = 'cpu'

            self.embedding_model = ORTModelForFeatureExtraction.from_pretrained(
                local_model_path_obj,
                file_name=onnx_filename,
                provider=providers[0], # Pass the primary provider explicitly
                provider_options=provider_options,
                # session_options=session_options # Optional: for advanced config like thread count
            )

            # Move model to the target device if using GPU (though ONNX handles this via provider)
            # For ORTModelForFeatureExtraction, placement is mainly via the provider.
            # If the underlying torch model parts needed moving, it would be here:
            # self.embedding_model.to(self.device)
            # However, this is usually not necessary/done for ORT models loaded this way.

            logger.info(f"ONNX Embedding model loaded successfully using provider '{providers[0]}'.")

        except Exception as e:
            logger.error(f"Failed to load local ONNX model from '{local_model_path_obj}': {e}", exc_info=True)
            self.embedding_model = None # Ensure it's None on failure
            raise # Re-raise the exception

    def generate_embeddings(self, texts: list[str]) -> Optional[np.ndarray]:
        """
        Generates embeddings for a list of text strings.
        Adds the 'passage: ' prefix suitable for E5 document embeddings.

        Args:
            texts: A list of strings to embed.

        Returns:
            A numpy array of embeddings (shape: [num_texts, embedding_dimension]),
            or None if embedding fails or the model is not loaded.
        """
        if not self.embedding_model or not self.tokenizer:
            logger.error("Embedding model or tokenizer not available. Cannot generate embeddings.")
            return None
        if not texts:
            return np.array([]) # Return empty array if no texts provided

        logger.debug(f"Generating embeddings for {len(texts)} texts using device '{self.device}' (with 'passage:' prefix)...")

        try:
            # --- Add 'passage: ' prefix --- 
            prefixed_texts = [f"passage: {text}" for text in texts]
            # ------------------------------

            # Tokenize the prefixed texts
            encoded_input = self.tokenizer(prefixed_texts, padding=True, truncation=True, return_tensors='pt')

            # --- Ensure inputs are on CPU for ONNX Runtime CPUExecutionProvider ---
            # ORT GPU provider might handle tensors on GPU, but CPU is safer baseline.
            # If self.device is cuda, ORT CUDA provider should handle it. Let's keep this check:
            if 'CPUExecutionProvider' in self.embedding_model.providers:
                 if any(t.device.type != 'cpu' for t in encoded_input.values()):
                      logger.debug("Moving tokenized input to CPU for ONNX Runtime.")
                      encoded_input = {k: v.to('cpu') for k, v in encoded_input.items()}

            # Run inference with the ONNX model
            # ORTModelForFeatureExtraction handles device placement based on the provider set during init.
            with torch.no_grad():
                model_output = self.embedding_model(**encoded_input)

            # Perform mean pooling using attention mask
            # Ensure attention_mask is on the same device as the output embeddings for pooling
            attention_mask = encoded_input['attention_mask']
            # Pooling happens on the device of the model_output tensors
            sentence_embeddings = _mean_pooling(model_output, attention_mask)

            # Normalize embeddings to unit length (L2 norm)
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            # Move embeddings to CPU before converting to numpy
            if sentence_embeddings.device.type != 'cpu':
                 sentence_embeddings = sentence_embeddings.cpu()

            logger.debug(f"Successfully generated embeddings with shape: {sentence_embeddings.shape}")
            return sentence_embeddings.numpy()

        except Exception as e:
            logger.error(f"Error during embedding generation: {e}", exc_info=True)
            return None

    def generate_turn_embeddings(self, turns: list[dict], min_turn_words: int) -> dict[int, list[float]]:
        """
        Generates embeddings for speaker turns that meet a minimum word count.

        Args:
            turns: A list of speaker turn dictionaries, each with 'text'.
            min_turn_words: The minimum number of words a turn must have to be embedded.

        Returns:
            A dictionary mapping the original turn index to its embedding vector (list of floats).
        """
        if not self.embedding_model or not self.tokenizer:
            logger.error("Embedding model or tokenizer not loaded. Cannot generate turn embeddings.")
            return {}

        texts_to_embed = []
        turn_indices_to_embed = [] # Store original indices corresponding to texts_to_embed

        for i, turn in enumerate(turns):
            text = turn.get('text', '')
            word_count = len(text.split())
            if word_count >= min_turn_words:
                texts_to_embed.append(text)
                turn_indices_to_embed.append(i)
            else:
                 logger.debug(f"Skipping embedding for turn {i} (speaker {turn.get('speaker', 'N/A')}, words: {word_count}, text: '{text[:50]}...')")

        embedding_map = {}
        if texts_to_embed:
            logger.info(f"Attempting to generate embeddings for {len(texts_to_embed)} eligible turns (min words: {min_turn_words})...")
            embeddings_array = self.generate_embeddings(texts_to_embed)

            if embeddings_array is not None and len(embeddings_array) == len(turn_indices_to_embed):
                logger.info(f"Successfully generated {len(embeddings_array)} embeddings.")
                # Convert numpy rows to lists and map back to original turn indices
                embedding_map = {
                    original_index: embedding_vector.tolist()
                    for original_index, embedding_vector
                    in zip(turn_indices_to_embed, embeddings_array)
                }
            elif embeddings_array is not None: # Handle length mismatch case
                 logger.error(f"Embedding generation returned unexpected number of vectors ({len(embeddings_array)}) for {len(texts_to_embed)} inputs.")
            else: # Handle None case (error already logged in generate_embeddings)
                 logger.error("Embedding generation failed.")
        else:
            logger.info("No turns met the minimum word count requirement for embedding.")

        return embedding_map

    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generates embedding for a single text string.
        This is a convenience method that wraps generate_embeddings for single text input.

        Args:
            text: A single string to embed.

        Returns:
            A 1D numpy array of embeddings (shape: [embedding_dimension]),
            or None if embedding fails or the model is not loaded.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            return None
            
        embeddings = self.generate_embeddings([text])
        if embeddings is not None and len(embeddings) > 0:
            return embeddings[0]  # Return first (and only) embedding as 1D array
        else:
            return None

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string using the tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        if not self.tokenizer:
            logger.error("Tokenizer not available. Cannot count tokens.")
            return 0
            
        if not text or not text.strip():
            return 0
            
        try:
            # Add passage prefix to match the actual embedding generation
            prefixed_text = f"passage: {text}"
            tokens = self.tokenizer.encode(prefixed_text, add_special_tokens=True)
            return len(tokens)
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return 0

# Example usage (optional, for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Testing EmbeddingGenerator...")

    # Assume config.yaml is in the parent directory relative to this script's location if run directly
    # Adjust this path as needed for your project structure
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent # Adjust if utils is nested differently
    config_file = project_root / "config" / "config.yaml"

    if not config_file.exists():
         logger.error(f"Test mode: config.yaml not found at expected location: {config_file}")
         sys.exit(1)

    try:
        embed_gen = EmbeddingGenerator(config_file)

        if embed_gen.embedding_model and embed_gen.tokenizer:
             logger.info("EmbeddingGenerator initialized successfully.")

             test_texts = [
                  "This is the first sentence.",
                  "Here is another sentence for testing embeddings.",
                  "Short one.",
                  "Це речення українською мовою для перевірки багатомовності.", # Ukrainian
                  "これは多言語性をテストするための日本語の文です。" # Japanese
             ]
             embeddings = embed_gen.generate_embeddings(test_texts)

             if embeddings is not None:
                  logger.info(f"Generated embeddings shape: {embeddings.shape}")
                  # logger.info(f"First embedding vector (first 5 dims): {embeddings[0][:5]}")
             else:
                  logger.error("Failed to generate embeddings during test.")

             # Test turn embedding generation
             test_turns = [
                {'text': 'This is turn one, it should be long enough.', 'speaker': 'A', 'start': 0.0, 'end': 5.0, 'turn_index': 0},
                {'text': 'Too short.', 'speaker': 'B', 'start': 5.5, 'end': 6.0, 'turn_index': 1},
                {'text': 'This is the second turn that meets the minimum word count.', 'speaker': 'A', 'start': 6.5, 'end': 12.0, 'turn_index': 2},
             ]
             min_words = 5
             turn_embeddings_map = embed_gen.generate_turn_embeddings(test_turns, min_words)

             logger.info(f"Generated embeddings for turns (min words {min_words}): {len(turn_embeddings_map)} turns.")
             logger.info(f"Map keys (original turn indices): {list(turn_embeddings_map.keys())}")
             if 0 in turn_embeddings_map:
                  # logger.info(f"Embedding for turn 0 (first 5 dims): {turn_embeddings_map[0][:5]}")
                  assert len(turn_embeddings_map[0]) == embed_gen.embedding_dimension
             if 2 in turn_embeddings_map:
                  # logger.info(f"Embedding for turn 2 (first 5 dims): {turn_embeddings_map[2][:5]}")
                  assert len(turn_embeddings_map[2]) == embed_gen.embedding_dimension
             if 1 in turn_embeddings_map:
                   logger.error("ERROR: Turn 1 (short) should not have been embedded.")

        else:
             logger.error("Failed to initialize EmbeddingGenerator model/tokenizer during test.")

    except Exception as main_e:
        logger.error(f"Error during EmbeddingGenerator test: {main_e}", exc_info=True) 