"""
LLM-based classifier for themes and subthemes.
"""

import asyncio
import logging
import re
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .data_structures import ClassificationResult

logger = logging.getLogger(__name__)

# MLX imports
try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class LLMClassifier:
    """LLM-based classifier for themes and sub-themes using MLX"""

    def __init__(
        self,
        themes_csv: str,
        model_name: str = "tier_1",
        use_gpu: bool = True,
        llm_endpoint: str = None,  # Deprecated
        project: Optional[str] = None,
        model_server_url: Optional[str] = None,
        use_model_server: bool = True
    ):
        self.project = project
        self.model_name = model_name
        self.model_server_url = model_server_url
        self.use_model_server = use_model_server and model_server_url is not None

        # Model name mapping
        self.model_configs = {
            "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit": {
                "aliases": ["qwen3:80b", "80b", "large", "tier_1", "tier1", "best"],
            },
            "mlx-community/Qwen3-4B-Instruct-2507-4bit": {
                "aliases": ["qwen3:4b-instruct", "qwen3:4b", "4b", "medium", "tier_2", "tier2", "balanced"],
            },
            "mlx-community/LFM2-8B-A1B-4bit": {
                "aliases": ["lfm2:8b", "8b", "small", "fast", "tier_3", "tier3", "fastest"],
            }
        }

        # If using model_server, skip local model loading
        if self.use_model_server:
            logger.info(f"Using model_server at {self.model_server_url} (model: {model_name})")
            self.model = None
            self.tokenizer = None
            import httpx
            self.http_client = httpx.AsyncClient(timeout=120.0)
        else:
            # Load local MLX model
            if not MLX_AVAILABLE:
                raise RuntimeError("MLX not available and no model_server configured")

            self.model_path = self._resolve_model_name(model_name)
            logger.info(f"Loading local MLX model: {self.model_path}")

            self.model, self.tokenizer = load(self.model_path)
            logger.info(f"Successfully loaded local MLX model: {self.model_path}")
            self.http_client = None

        self.themes, self.subthemes = self._load_themes(themes_csv)

    def _resolve_model_name(self, requested_model: str) -> str:
        """Resolve model alias to full model name."""
        if requested_model in self.model_configs:
            return requested_model

        for model_path, config in self.model_configs.items():
            if requested_model in config.get('aliases', []):
                return model_path

        logger.warning(f"Unknown model '{requested_model}', defaulting to 4B model")
        return "mlx-community/Qwen3-4B-Instruct-2507-4bit"

    def _load_themes(self, csv_path: str) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """Load themes and subthemes from CSV"""
        df = pd.read_csv(csv_path)

        themes = {}
        subthemes = {}

        for theme_id in df['theme_id'].unique():
            theme_df = df[df['theme_id'] == theme_id]
            theme_info = theme_df.iloc[0]

            themes[theme_id] = {
                'id': theme_id,
                'name': theme_info['theme_name'],
                'description': theme_info['theme_description']
            }

            # Collect subthemes
            theme_subthemes = []
            for _, row in theme_df.iterrows():
                if pd.notna(row.get('subtheme_id')):
                    subtheme = {
                        'id': row['subtheme_id'],
                        'name': row['subtheme_name'],
                        'description': row.get('subtheme_description_short', row['subtheme_description'])
                    }
                    theme_subthemes.append(subtheme)

            if theme_subthemes:
                subthemes[theme_id] = theme_subthemes

        return themes, subthemes

    def _call_llm_sync(self, prompt: str, priority: int = 2) -> str:
        """Synchronous wrapper for async _call_llm"""
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self._call_llm(prompt, priority))
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._call_llm(prompt, priority))

    async def _call_llm(self, prompt: str, priority: int = 2) -> str:
        """Call LLM - either via model_server or local MLX"""
        logger.debug("="*60)
        logger.debug("LLM PROMPT:")
        logger.debug(prompt)
        logger.debug("="*60)

        messages = [
            {"role": "system", "content": "You are a precise classification assistant. Respond only with the requested numbers."},
            {"role": "user", "content": prompt}
        ]

        # Try model_server first if configured
        if self.use_model_server and self.http_client:
            try:
                response = await self.http_client.post(
                    f"{self.model_server_url}/llm-request",
                    json={
                        "messages": [{"role": msg["role"], "content": msg["content"]} for msg in messages],
                        "model": self.model_name,
                        "priority": priority,
                        "temperature": 0.1,
                        "max_tokens": 50,
                        "top_p": 0.9
                    },
                    timeout=120.0
                )

                if response.status_code == 200:
                    data = response.json()
                    llm_response = data['response'].strip()
                    logger.debug(f"LLM RESPONSE (model_server): {llm_response}")
                    logger.debug("="*60)
                    return llm_response
                else:
                    logger.warning(f"model_server returned {response.status_code}, falling back to local MLX")

            except Exception as e:
                logger.warning(f"model_server request failed: {e}, falling back to local MLX")

        # Fallback to local MLX
        if self.model is None:
            raise RuntimeError("No model_server available and local MLX model not loaded")

        try:
            if self.tokenizer.chat_template is not None:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

            sampler = make_sampler(temp=0.1, top_p=0.9)

            response_text = generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=50,
                sampler=sampler,
                verbose=False
            )

            llm_response = response_text.strip()
            logger.debug(f"LLM RESPONSE (local): {llm_response}")
            logger.debug("="*60)

            return llm_response

        except Exception as e:
            logger.error(f"MLX generation error: {e}")
            raise

    async def _call_llm_batch(self, prompts: List[str], priority: int = 2, max_chunk_size: int = 50) -> List[str]:
        """Call LLM with multiple prompts concurrently"""
        if not prompts:
            return []

        if len(prompts) > max_chunk_size:
            logger.debug(f"Splitting {len(prompts)} prompts into chunks of {max_chunk_size}")
            all_results = []
            for i in range(0, len(prompts), max_chunk_size):
                chunk = prompts[i:i+max_chunk_size]
                chunk_results = await self._call_llm_batch(chunk, priority=priority, max_chunk_size=max_chunk_size)
                all_results.extend(chunk_results)
            return all_results

        tasks = [self._call_llm(prompt, priority=priority) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r if not isinstance(r, Exception) else "" for r in results]

    def classify_theme(self, text: str, context: Optional[Dict] = None, segment_id: Optional[int] = None) -> ClassificationResult:
        """Classify text into themes"""
        if not text or not text.strip():
            return ClassificationResult()

        themes_list = "\n".join([
            f"{tid}. {tinfo['description']}"
            for tid, tinfo in sorted(self.themes.items())
        ])

        prompt = f"""Classify the following text segment into relevant themes.
You may select multiple themes if applicable.
Respond with ONLY the theme numbers separated by commas (e.g., "1,3,5").
If no themes apply, respond with "0".

THEMES:
{themes_list}

TEXT TO CLASSIFY:
{text[:2000]}

THEME NUMBERS (comma-separated):"""

        try:
            response = self._call_llm_sync(prompt, priority=1)

            if response == "0":
                result = ClassificationResult(reasoning="No themes identified")
            else:
                numbers = re.findall(r'\d+', response)
                theme_ids = []
                seen = set()
                for num_str in numbers:
                    num = int(num_str)
                    if num in self.themes and num not in seen:
                        theme_ids.append(num)
                        seen.add(num)

                if not theme_ids:
                    result = ClassificationResult(reasoning="No valid themes identified")
                else:
                    theme_names = [self.themes[tid]['name'] for tid in theme_ids]

                    result = ClassificationResult(
                        theme_id=theme_ids[0],
                        theme_ids=theme_ids,
                        theme_name=theme_names[0],
                        theme_names=theme_names,
                        confidence=0.9 if len(theme_ids) == 1 else 0.7,
                        reasoning=f"Classified as {', '.join(theme_names)}"
                    )

            return result

        except Exception as e:
            logger.error(f"Theme classification error: {e}")
            return ClassificationResult(reasoning=f"Error: {str(e)}")

    def classify_subtheme(self, text: str, theme_id: int, context: Optional[Dict] = None, segment_id: Optional[int] = None) -> ClassificationResult:
        """Classify text into subthemes within a theme"""
        if not text or theme_id not in self.themes:
            return ClassificationResult()

        if theme_id not in self.subthemes:
            return ClassificationResult(reasoning=f"No subthemes for theme {theme_id}")

        # Create simple numbering for subthemes
        subtheme_mapping = {}
        subthemes_list = []
        for i, st in enumerate(self.subthemes[theme_id], 1):
            subtheme_mapping[i] = st['id']
            subthemes_list.append(f"{i}. {st['description']}")

        theme_info = self.themes[theme_id]
        prompt = f"""Classify the following text segment into relevant sub-themes for the theme "{theme_info['name']}".

INSTRUCTIONS:
- If the theme does not apply to this text at all, respond with "not applicable"
- If the theme applies but no specific sub-themes match, respond with "0" (theme only)
- If specific sub-themes apply, respond with their numbers separated by commas (e.g., "1,3,5")
- You may select multiple sub-themes if applicable

SUB-THEMES for {theme_info['name']}:
{chr(10).join(subthemes_list)}

TEXT TO CLASSIFY:
{text}

RESPONSE (sub-theme numbers, "0" for theme only, or "not applicable"):"""

        try:
            response = self._call_llm_sync(prompt, priority=1)
            response_clean = response.strip().lower()

            # Check for "not applicable" - theme does not apply at all
            if "not applicable" in response_clean or "not_applicable" in response_clean:
                return ClassificationResult(
                    theme_ids=[],  # Signal that theme should be removed
                    reasoning="Theme not applicable"
                )

            # Check for "0" - theme applies but no specific subthemes
            if response_clean in ["0", "none"]:
                return ClassificationResult(
                    theme_ids=[theme_id],  # Keep theme but no subthemes
                    reasoning="Theme applies but no specific subthemes identified"
                )

            # Parse subtheme numbers
            numbers = re.findall(r'\d+', response)
            subtheme_ids = []
            for num_str in numbers:
                num = int(num_str)
                if num in subtheme_mapping:
                    subtheme_ids.append(subtheme_mapping[num])

            if not subtheme_ids:
                result = ClassificationResult(reasoning="No valid subthemes identified")
            else:
                available_subthemes = {st['id']: st for st in self.subthemes[theme_id]}
                subtheme_names = [
                    available_subthemes[sid]['name']
                    for sid in subtheme_ids
                    if sid in available_subthemes
                ]

                result = ClassificationResult(
                    subtheme_ids=subtheme_ids,
                    subtheme_names=subtheme_names,
                    confidence=0.8 if len(subtheme_ids) <= 2 else 0.6,
                    reasoning=f"Identified {len(subtheme_ids)} subthemes"
                )

            return result

        except Exception as e:
            logger.error(f"Subtheme classification error: {e}")
            return ClassificationResult(reasoning=f"Error: {str(e)}")
