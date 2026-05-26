"""
Feature extraction for Vision-Language Models with Multi-turn Conversation Support.

Extracts vision representations from user prompts' attention to vision tokens.

Supports ShareGPT format multi-turn conversations.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


def _clean_text_for_tokens(text: str) -> str:
    return (
        text.replace("<image>", " ")
        .replace("<img>", " ")
        .replace("<|vision_start|>", " ")
        .replace("<|vision_end|>", " ")
        .replace("<|image_pad|>", " ")
    )


def _tokenize_text(text: str) -> List[str]:
    cleaned = _clean_text_for_tokens(text).lower()
    return TOKEN_PATTERN.findall(cleaned)


def _stable_hash_index(token: str, dim: int) -> Tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    index = int.from_bytes(digest[:4], "little") % dim
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return index, sign


def _hashed_vector(tokens: Sequence[str], dim: int) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    for token in tokens:
        if not token:
            continue
        index, sign = _stable_hash_index(token, dim)
        vec[index] += sign
    norm = torch.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _safe_mean_hidden(hidden: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if hidden.dim() == 3:
        hidden = hidden[0]
    if not indices:
        return hidden.mean(dim=0)
    valid_indices = [idx for idx in indices if 0 <= idx < hidden.shape[0]]
    if not valid_indices:
        return hidden.mean(dim=0)
    return hidden[valid_indices].mean(dim=0)


def _bucket_numeric(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if number <= 0:
        return "0"
    if number <= 4:
        return "1-4"
    if number <= 8:
        return "5-8"
    if number <= 16:
        return "9-16"
    if number <= 32:
        return "17-32"
    if number <= 64:
        return "33-64"
    return "65+"


def _flatten_message_text(messages: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages:
        content = str(message.get("content", ""))
        if content:
            parts.append(content)
    return " ".join(parts)


def _metadata_tokens(metadata: Optional[Dict[str, Any]], messages: Sequence[Dict[str, Any]]) -> List[str]:
    meta = metadata or {}
    tokens: List[str] = []

    scene = str(meta.get("scene", "unknown")).lower()
    tokens.append(f"scene:{scene}")

    for tag in sorted(set(meta.get("task_tags", []) or [])):
        tokens.append(f"task:{str(tag).lower()}")

    for tag in sorted(set(meta.get("rare_tags", []) or [])):
        tokens.append(f"rare:{str(tag).lower()}")

    for label in sorted(set(meta.get("positive_labels", []) or [])):
        tokens.append(f"label:{str(label).lower()}")

    qa_total = meta.get("qa_pairs_total")
    qa_kept = meta.get("qa_pairs_kept")
    image_count = len(meta.get("images", []) or [])
    message_count = len(messages)
    tokens.extend(
        [
            f"qa_total:{_bucket_numeric(qa_total)}",
            f"qa_kept:{_bucket_numeric(qa_kept)}",
            f"images:{image_count}",
            f"messages:{_bucket_numeric(message_count)}",
        ]
    )

    return tokens


class VLMFeatureExtractor:
    """Extract features from Vision-Language Models."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        trust_remote_code: bool = True,
        torch_dtype: str = "auto",
        max_length: int = 4096,
        model_type: str = "llava",
        feature_mode: str = "hybrid_meta",
        text_alpha: float = 0.30,
        meta_alpha: float = 0.15,
        qwen_image_min_pixels: Optional[int] = None,
        qwen_image_max_pixels: Optional[int] = None,
        use_base_model_hidden: bool = True,
        cleanup_cuda_cache: bool = False,
    ):
        """
        Initialize the VLM feature extractor.

        Args:
            model_name: HuggingFace model name or path
            device: Device to run model on
            trust_remote_code: Whether to trust remote code
            torch_dtype: Data type for model weights
            max_length: Maximum sequence length
            model_type: Model type - "llava" or "qwen"
            feature_mode: "hybrid_meta" for fast hidden/text/meta features,
                or "qwen_attention" for the original attention-guided baseline.
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model_type = model_type.lower()
        self.feature_mode = feature_mode.lower()
        self.text_alpha = text_alpha
        self.meta_alpha = meta_alpha
        self.qwen_image_min_pixels = qwen_image_min_pixels
        self.qwen_image_max_pixels = qwen_image_max_pixels
        self.use_base_model_hidden = use_base_model_hidden
        self.cleanup_cuda_cache = cleanup_cuda_cache
        self._base_model_fast_path_disabled = False
        if self.feature_mode not in {"hybrid_meta", "qwen_attention"}:
            raise ValueError("feature_mode must be one of: hybrid_meta, qwen_attention")

        # Map torch_dtype string to actual dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, "auto")

        print(f"Loading model: {model_name}")
        print(f"Model type: {self.model_type}")
        print(f"Feature mode: {self.feature_mode}")
        print(f"Device: {device}")
        print(f"Dtype: {torch_dtype}")
        print(f"Max length: {max_length}")
        print(f"Use base model hidden fast path: {self.use_base_model_hidden}")
        print(f"Cleanup CUDA cache per batch: {self.cleanup_cuda_cache}")

        # Load processor and model
        processor_kwargs = {
            'trust_remote_code': trust_remote_code,
        }
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

        # Qwen: 手动设置 image_processor.size 以控制视觉 token 数量
        # (max_pixels 参数在 from_pretrained 中不生效)
        if self.model_type == 'qwen':
            min_pixels = qwen_image_min_pixels or 256 * 32 * 32
            max_pixels = qwen_image_max_pixels or 576 * 32 * 32
            self.processor.image_processor.size = {
                "longest_edge": max_pixels,
                "shortest_edge": min_pixels,
            }
            print(f"Qwen image pixels: min={min_pixels}, max={max_pixels}")


        # Set padding side to left for decoder-only models
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": self.torch_dtype,
            "device_map": device if device == "auto" else None,
        }
        if self.feature_mode == "qwen_attention":
            model_kwargs["attn_implementation"] = "eager"  # Required for output_attentions
        else:
            model_kwargs["attn_implementation"] = "sdpa"

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            **model_kwargs,
        )

        if device != "auto":
            self.model = self.model.to(device)

        self.model.eval()

        # Disable KV cache
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = False

        # Check if processor supports apply_chat_template
        self.use_chat_template = hasattr(self.processor, 'apply_chat_template')
        if self.use_chat_template:
            print("✓ Using official apply_chat_template")
        else:
            print("⚠ Fallback to manual formatting")

        # Get Qwen vision token IDs
        self.qwen_vision_token_ids = set()
        if self.model_type == 'qwen':
            tokenizer = self.processor.tokenizer
            for token_name in ['<|image_pad|>', '<|vision_start|>', '<|vision_end|>']:
                token_id = tokenizer.convert_tokens_to_ids(token_name)
                if token_id != tokenizer.unk_token_id:
                    self.qwen_vision_token_ids.add(token_id)
            print(f"✓ Qwen vision token IDs: {self.qwen_vision_token_ids}")

        print("Model loaded successfully!")

    def convert_sharegpt_to_hf_format(
        self,
        messages: List[Dict],
        num_images: int = 0,
    ) -> List[Dict]:
        """
        Convert ShareGPT format to HuggingFace LLaVA format.

        ShareGPT: [{"role": "user", "content": "text"}, ...]
        HF: [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}, ...]

        Args:
            messages: ShareGPT format messages
            num_images: Number of images in this conversation (will add image placeholders)

        Returns:
            HF format messages
        """
        hf_messages = []
        images_added = False

        for msg in messages:
            role = msg.get('role')
            content = msg.get('content', '')
            hf_content = []

            # For the first user message, add image placeholders and remove existing <image> tags
            if role == 'user' and num_images > 0 and not images_added:
                hf_content.extend({"type": "image"} for _ in range(num_images))
                images_added = True
                # Remove <image> tags from content to avoid duplication
                content = content.replace('<image>', '').replace('<img>', '').strip()

            # Add text content
            if content:
                hf_content.append({"type": "text", "text": content})

            if hf_content:
                hf_messages.append({"role": role, "content": hf_content})

        return hf_messages

    def convert_sharegpt_to_qwen_format(
        self,
        messages: List[Dict],
        num_images: int = 0,
    ) -> List[Dict]:
        """
        Convert ShareGPT format to Qwen-VL format.

        Uses placeholder {"type": "image"} only - real images are passed via processor(images=...).
        This avoids "double image passing" mismatch between template and processor.

        Args:
            messages: ShareGPT format messages
            num_images: Number of images (for placeholders only)

        Returns:
            Qwen-VL format messages
        """
        qwen_messages = []
        images_added = False

        for msg in messages:
            role = msg.get('role')
            content = msg.get('content', '')

            if role == 'user' and num_images > 0 and not images_added:
                qwen_content = []
                # Add image placeholders (no real image objects)
                qwen_content.extend({"type": "image"} for _ in range(num_images))
                images_added = True
                # Remove <image> tags from content
                clean_content = content.replace('<image>', '').replace('<img>', '').strip()
                if clean_content:
                    qwen_content.append({"type": "text", "text": clean_content})
                qwen_messages.append({"role": role, "content": qwen_content})
            else:
                # Assistant or subsequent user messages: content as string
                qwen_messages.append({"role": role, "content": content})

        return qwen_messages

    def create_assistant_labels(
        self,
        input_ids: torch.Tensor,
        messages: List[Dict],
    ) -> torch.Tensor:
        """
        Create labels to identify assistant tokens (used for user_indices calculation).

        Strategy: Tokenize assistant responses and search for them in input_ids.
        This approach is robust to image tokens and special formatting.

        Args:
            input_ids: Input token IDs [seq_len] from batch processing
            messages: List of messages with 'role' and 'content' (ShareGPT format)

        Returns:
            labels: Same shape as input_ids, with -100 for non-assistant positions
        """
        labels = torch.full_like(input_ids, -100)

        try:
            # Handle padding: find actual content in input_ids
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                non_pad_mask = input_ids != pad_token_id
                if non_pad_mask.any():
                    first_non_pad = non_pad_mask.nonzero(as_tuple=True)[0][0].item()
                else:
                    first_non_pad = 0
            else:
                first_non_pad = 0

            # Search for each assistant response in input_ids
            search_start = first_non_pad

            for msg in messages:
                role = msg.get('role')
                content = msg.get('content', '').strip()

                if role == 'assistant' and content:
                    # Tokenize just the assistant's response content
                    # Use add_special_tokens=False to get pure content tokens
                    response_ids = self.processor.tokenizer(
                        content,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )['input_ids'][0]

                    if len(response_ids) == 0:
                        continue

                    # Search for this token sequence in input_ids
                    # Start from where we left off to maintain order
                    found_pos = self._find_token_sequence(
                        input_ids,
                        response_ids,
                        start_pos=search_start
                    )

                    if found_pos is not None:
                        # Mark these positions in labels
                        end_pos = found_pos + len(response_ids)
                        labels[found_pos:end_pos] = input_ids[found_pos:end_pos]
                        # Update search start for next assistant message
                        search_start = end_pos

        except Exception as e:
            print(f"Warning: Failed to create assistant labels: {e}")
            import traceback
            traceback.print_exc()

        return labels

    def _find_token_sequence(
        self,
        input_ids: torch.Tensor,
        pattern: torch.Tensor,
        start_pos: int = 0,
    ) -> int:
        """
        Find a token sequence in input_ids using sliding window.

        Args:
            input_ids: The full token sequence to search in
            pattern: The token pattern to find
            start_pos: Position to start searching from

        Returns:
            Starting position of the pattern, or None if not found
        """
        if len(pattern) == 0 or len(input_ids) == 0:
            return None

        pattern_len = len(pattern)
        max_start = len(input_ids) - pattern_len + 1

        for i in range(start_pos, max_start):
            if torch.equal(input_ids[i:i + pattern_len], pattern):
                return i

        return None

    def identify_token_types(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[List[int], List[int]]:
        """
        Identify vision and text tokens.

        Args:
            input_ids: Input token IDs [batch, seq_len] or [seq_len]

        Returns:
            (vision_indices, text_indices)
        """
        input_ids_flat = input_ids[0] if input_ids.dim() > 1 else input_ids

        vision_indices = []
        text_indices = []

        # Qwen: use specific vision token IDs
        if self.model_type == 'qwen' and self.qwen_vision_token_ids:
            for idx, token_id in enumerate(input_ids_flat):
                if token_id.item() in self.qwen_vision_token_ids:
                    vision_indices.append(idx)
                else:
                    text_indices.append(idx)
            return vision_indices, text_indices

        # LLaVA: use vocab_size threshold
        vocab_size = self.processor.tokenizer.vocab_size
        vision_token_id = getattr(self.processor.tokenizer, 'vision_token_id', None)
        image_token_id = getattr(self.processor.tokenizer, 'image_token_id', None)

        for idx, token_id in enumerate(input_ids_flat):
            token_id_val = token_id.item()
            is_vision = False

            if vision_token_id is not None and token_id_val == vision_token_id:
                is_vision = True
            elif image_token_id is not None and token_id_val == image_token_id:
                is_vision = True
            elif token_id_val >= vocab_size:
                is_vision = True

            if is_vision:
                vision_indices.append(idx)
            else:
                text_indices.append(idx)

        return vision_indices, text_indices


    def build_hybrid_representation(
        self,
        hidden: torch.Tensor,
        vision_indices: List[int],
        messages: List[Dict],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build the fast hybrid representation without extracting attentions."""
        if hidden.dim() == 3:
            hidden = hidden[0]

        vision_repr = _safe_mean_hidden(hidden.float().cpu(), vision_indices)
        vision_repr = F.normalize(vision_repr, dim=0)
        dim = int(vision_repr.numel())

        text_tokens = _tokenize_text(_flatten_message_text(messages))
        text_repr = _hashed_vector(text_tokens, dim)
        meta_repr = _hashed_vector(_metadata_tokens(metadata, messages), dim)

        combined = vision_repr + self.text_alpha * text_repr + self.meta_alpha * meta_repr
        combined = F.normalize(combined, dim=0)

        return {
            "vision_repr": vision_repr,
            "text_repr": text_repr,
            "meta_repr": meta_repr,
            "combined_repr": combined,
        }


    def _maybe_empty_cache(self) -> None:
        if self.cleanup_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()


    def _forward_hybrid_base_model(self, inputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Use the base VLM model to skip LM head logits and all-layer hidden states."""
        if (
            self.feature_mode != "hybrid_meta"
            or not self.use_base_model_hidden
            or self._base_model_fast_path_disabled
        ):
            return None

        base_model = getattr(self.model, "model", None)
        if base_model is None:
            self._base_model_fast_path_disabled = True
            return None

        try:
            outputs = base_model(
                **inputs,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
        except TypeError:
            try:
                outputs = base_model(
                    **inputs,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
            except Exception as exc:
                print(f"Warning: base model hidden fast path disabled: {exc}")
                self._base_model_fast_path_disabled = True
                return None
        except Exception as exc:
            print(f"Warning: base model hidden fast path disabled: {exc}")
            self._base_model_fast_path_disabled = True
            return None

        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None and isinstance(outputs, (tuple, list)) and outputs:
            last_hidden = outputs[0]
        return last_hidden


    def extract_features_batch(
        self,
        images_list: List[List[Image.Image]],
        messages_list: List[List[Dict]],
        sample_meta_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict]:
        """
        Extract features for multi-turn conversations.

        For each sample:
        - qwen_attention: extract the original user-attended vision baseline.
        - hybrid_meta: extract a fast hidden/text/metadata representation.

        Args:
            images_list: List of image lists
            messages_list: List of conversation messages (ShareGPT format)
            sample_meta_list: Optional MITS metadata per sample.

        Returns:
            List of dicts with features for each sample
        """
        batch_size = len(messages_list)
        if sample_meta_list is None:
            sample_meta_list = [{} for _ in range(batch_size)]
        elif len(sample_meta_list) != batch_size:
            raise ValueError("sample_meta_list must have the same length as messages_list")

        # Build conversation text based on model type
        text_inputs = []
        for idx, messages in enumerate(messages_list):
            if self.model_type == 'qwen':
                # Convert ShareGPT format to Qwen format (placeholders only, real images via processor)
                num_images = len(images_list[idx]) if idx < len(images_list) else 0
                qwen_messages = self.convert_sharegpt_to_qwen_format(messages, num_images)
                text_input = self.processor.apply_chat_template(
                    qwen_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

            else:
                # LLaVA: use apply_chat_template if supported
                if self.use_chat_template:
                    # Convert ShareGPT format to HF format
                    num_images = len(images_list[idx]) if idx < len(images_list) else 0
                    hf_messages = self.convert_sharegpt_to_hf_format(messages, num_images)

                    # Use official chat template
                    text_input = self.processor.apply_chat_template(
                        hf_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                else:
                    # Fallback: Manual format for LLaVA
                    conversation_parts = []
                    for msg in messages:
                        role = msg.get('role', '')
                        content = msg.get('content', '')

                        if role == 'user':
                            conversation_parts.append(f"USER: {content}")
                        elif role == 'assistant':
                            conversation_parts.append(f"ASSISTANT: {content}")

                    text_input = " ".join(conversation_parts)

            text_inputs.append(text_input)

        # Process inputs
        # NOTE: Pass images_list directly (list of lists) so processor knows
        # which images belong to which text in the batch
        # Check if there are any actual images in the batch
        has_images = any(len(imgs) > 0 for imgs in images_list)

        if has_images:
            inputs = self.processor(
                text=text_inputs,
                images=images_list,  # Keep as list of lists, not flattened
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        else:
            # Text-only processing (no images)
            inputs = self.processor(
                text=text_inputs,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )

        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}


        # Forward pass
        output_attentions = self.feature_mode == "qwen_attention"
        outputs = None
        with torch.inference_mode():
            batch_input_ids = inputs['input_ids'].cpu().clone()

            if self.feature_mode == "hybrid_meta":
                batch_hidden = self._forward_hybrid_base_model(inputs)
                batch_attention = None
                if batch_hidden is None:
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        output_attentions=False,
                        use_cache=False,
                        return_dict=True,
                    )
                    batch_hidden = outputs.hidden_states[-1] if outputs.hidden_states else None
            else:
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                    use_cache=False,
                    return_dict=True,
                )
                batch_hidden = outputs.hidden_states[1] if len(outputs.hidden_states) > 1 else None
                batch_attention = outputs.attentions[0] if outputs.attentions else None

        # Cleanup inputs
        for key in list(inputs.keys()):
            if isinstance(inputs[key], torch.Tensor):
                del inputs[key]
        del inputs

        # Process each sample. Attention tensors are only materialized for the baseline.
        # This reduces O(n²) memory to O(n) before cloning
        results = []
        processed_attentions = None

        if self.feature_mode == "qwen_attention":
            processed_attentions = []
            if batch_attention is not None:
                # Mean over batch and heads once: [batch, heads, seq, seq] -> [batch, seq, seq]
                attention_mean = batch_attention.mean(dim=1)

                for i in range(batch_size):
                    # Only keep the 2D attention matrix for this sample.
                    processed_attentions.append(attention_mean[i].cpu().float())

                # Free the huge 4D tensor immediately.
                batch_attention = None
                attention_mean = None
            else:
                processed_attentions = [None] * batch_size

            self._maybe_empty_cache()

            for i in range(batch_size):
                sample_input_ids = batch_input_ids[i:i+1].clone()
                sample_first_hidden = (
                    batch_hidden[i].detach().float().cpu()
                    if batch_hidden is not None
                    else None
                )
                sample_attention = processed_attentions[i]

                # Identify token types
                vision_indices, text_indices = self.identify_token_types(sample_input_ids)

                # Create labels to identify assistant tokens precisely.
                labels = self.create_assistant_labels(
                    sample_input_ids[0],
                    messages_list[i],
                )

                # User indices = text tokens that are NOT assistant tokens (labels == -100).
                # This ensures we only use USER tokens for vision attention.
                assistant_mask = labels != -100
                user_indices = [
                    idx for idx in text_indices
                    if idx not in vision_indices and not assistant_mask[idx].item()
                ]

                results.append({
                    'first_layer_hidden': sample_first_hidden,
                    'attention': sample_attention,
                    'input_ids': sample_input_ids,
                    'vision_indices': vision_indices,
                    'user_indices': user_indices,
                })
        else:
            self._maybe_empty_cache()

            for i in range(batch_size):
                sample_input_ids = batch_input_ids[i:i+1].clone()
                sample_hidden = (
                    batch_hidden[i].detach().float().cpu()
                    if batch_hidden is not None
                    else None
                )
                vision_indices, _text_indices = self.identify_token_types(sample_input_ids)

                if sample_hidden is None:
                    results.append({'combined_repr': None, 'success': False})
                    continue

                hybrid_features = self.build_hybrid_representation(
                    hidden=sample_hidden,
                    vision_indices=vision_indices,
                    messages=messages_list[i],
                    metadata=sample_meta_list[i],
                )

                results.append({
                    'combined_repr': hybrid_features['combined_repr'],
                    'vision_repr': hybrid_features['vision_repr'],
                    'text_repr': hybrid_features['text_repr'],
                    'meta_repr': hybrid_features['meta_repr'],
                    'vision_indices': vision_indices,
                    'success': True,
                })

        # Cleanup
        del batch_hidden, batch_input_ids
        if outputs is not None:
            del outputs
        if processed_attentions is not None:
            del processed_attentions

        self._maybe_empty_cache()

        return results

    def load_images(
        self,
        image_data: Union[str, List[str], np.ndarray, List[np.ndarray], Image.Image, List[Image.Image], dict, List[dict]],
    ) -> List[Image.Image]:
        """Load images from various formats."""
        if not isinstance(image_data, list):
            image_data = [image_data]

        images = []
        for item in image_data:
            try:
                if isinstance(item, Image.Image):
                    images.append(item)
                elif isinstance(item, str):
                    if item and item.strip():
                        images.append(Image.open(item).convert('RGB'))
                elif isinstance(item, np.ndarray):
                    if item.ndim == 2:
                        images.append(Image.fromarray(item, mode='L').convert('RGB'))
                    elif item.ndim == 3:
                        if item.dtype != np.uint8:
                            if item.max() <= 1.0:
                                item = (item * 255).astype(np.uint8)
                            else:
                                item = item.astype(np.uint8)
                        if item.shape[2] == 3:
                            images.append(Image.fromarray(item, mode='RGB'))
                        elif item.shape[2] == 4:
                            images.append(Image.fromarray(item, mode='RGBA').convert('RGB'))
            except Exception as e:
                print(f"Warning: Failed to load image: {e}")
                continue

        return images
