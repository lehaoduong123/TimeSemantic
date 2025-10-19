import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional, Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaTokenizer
from src.models.AXIS.Pretrain_ts_encoder import TimeSeriesPretrainModel

class MultiheadAttention(nn.Module):
    """Standard Multi-head Attention module, non-causal by default.
    
    This module implements the standard multi-head attention mechanism as described
    in "Attention Is All You Need" paper, but without causal masking.
    
    Attributes:
        embed_dim: Dimension of the embedding vectors.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
    """
    
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """Initialize MultiheadAttention module.
        
        Args:
            embed_dim: Dimension of the embedding vectors.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V, and output
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Project inputs to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi-head attention
        Q = Q.view(query.shape[0], query.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(value.shape[0], value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # Prepare attention mask for padding
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = None

        # Compute scaled dot-product attention (non-causal)
        y = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            is_causal=False  # Non-causal attention for encoder
        )

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], -1)
        y = self.out_proj(y)
        return y


class Perceiver(nn.Module):
    """Perceiver module for time series to text mapping.
    
    This module maps time series embeddings and fixed prompt embeddings to word embeddings
    using cross-attention mechanisms. It serves as a bridge between time series representations
    and text embeddings.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int,
                 d_proj: int,
                 num_prototype: int,
                 num_fixed_tokens: int,
                 num_heads: int):
        """Initialize Perceiver module.
        
        Args:
            vocab_size: Size of the vocabulary for word embeddings.
            hidden_size: Hidden dimension size of the LLM.
            d_proj: Projection dimension of time series embeddings.
            num_prototype: Number of prototype embeddings for mapping.
            num_fixed_tokens: Number of fixed prompt tokens.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.d_proj = d_proj
        self.num_prototype = num_prototype
        self.num_fixed_tokens = num_fixed_tokens
        
        # Mapping layer from vocab to prototype space
        self.mapping_layer = nn.Linear(vocab_size, num_prototype)
        
        # Fixed prompt embeddings
        self.fix_prompt_embeddings = nn.Parameter(
            torch.randn(1, num_fixed_tokens, hidden_size)
        )
        
        # Local time series projection
        self.local_word_proj = nn.Linear(d_proj, hidden_size)
        
        # Cross-attention for local embeddings
        self.local_attention = MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads
        )
        
        self._init_parameters()
    
    def _init_parameters(self) -> None:
        """Initialize parameters for the Perceiver module."""
        # Initialize linear layers
        linear_layers = [
            self.local_word_proj,
            self.mapping_layer
        ]
        for layer in linear_layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Initialize attention layers
        attention_layers = [
            self.local_attention.q_proj,
            self.local_attention.k_proj,
            self.local_attention.v_proj,
            self.local_attention.out_proj,
        ]
        for layer in attention_layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Initialize prompt embeddings
        nn.init.normal_(self.fix_prompt_embeddings, mean=0.0, std=0.02)
    
    def get_source_embeddings(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """Transform word embeddings to source embeddings for cross-attention.
        
        Args:
            word_embeddings: Word embeddings from the LLM.
            
        Returns:
            Source embeddings for cross-attention.
        """
        return self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0).unsqueeze(0)
    
    def process_local_embeddings(self, 
                                local_embeddings: torch.Tensor,
                                source_embeddings: torch.Tensor,
                                start_idx: int,
                                end_idx: int) -> torch.Tensor:
        local_ts_embeddings = local_embeddings[start_idx:end_idx, :].unsqueeze(0)
        local_ts_embeddings = self.local_word_proj(local_ts_embeddings)
        projected_local_embeddings = self.local_attention(
            local_ts_embeddings, 
            source_embeddings, 
            source_embeddings
        ).squeeze(0)
        
        return projected_local_embeddings
    
    def process_fixed_embeddings(self, 
                                source_embeddings: torch.Tensor,
                                num_tokens: int) -> torch.Tensor:
        """Process fixed prompt embeddings through cross-attention.
        
        Args:
            source_embeddings: Source embeddings for cross-attention.
            num_tokens: Number of fixed tokens needed.
            
        Returns:
            Processed fixed embeddings.
        """
        # Get fixed embeddings
        fixed_embeddings = self.fix_prompt_embeddings.squeeze(0)[:num_tokens].unsqueeze(0)
        
        # Apply cross-attention
        processed_fixed_embeddings = self.local_attention(
            fixed_embeddings,
            source_embeddings,
            source_embeddings
        ).squeeze(0)
        
        return processed_fixed_embeddings


class AXIS(nn.Module):
    def __init__(
        self,
        config: Any,
    ) -> None:
        """Initialize AXIS model.
        
        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        self.config = config
        
        # Load model without device_map for better multi-GPU compatibility
        self.model = AutoModelForCausalLM.from_pretrained(
            config.llm_config.model_name,
            torch_dtype=torch.float16 if config.mixed_precision else torch.float32,
            trust_remote_code=True,
            token=config.llm_config.token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_config.model_name, token=config.llm_config.token)

        # Freeze base model parameters
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Add special tokens
        special_tokens = ['<|local_hint|>', '<|fixed_hint|>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Set padding token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # Initialize model components
        num_prototype = config.llm_config.num_prototype
        self.num_fixed_tokens = config.llm_config.num_fixed_tokens
        self.vocab_size = self.model.config.vocab_size
        self.max_batch_size = config.llm_config.max_batch_size
        
        # Initialize Perceiver module
        self.perceiver = Perceiver(
            vocab_size=self.vocab_size,
            hidden_size=self.model.config.hidden_size,
            d_proj=config.ts_config.d_proj,
            num_prototype=num_prototype,
            num_fixed_tokens=self.num_fixed_tokens,
            num_heads=config.llm_config.num_heads
        )
        
        self.local_hint_token_id = self.tokenizer.convert_tokens_to_ids('<|local_hint|>')
        self.fixed_hint_token_id = self.tokenizer.convert_tokens_to_ids('<|fixed_hint|>')
    
    def get_device(self) -> torch.device:
        """Get the device of the model's parameters."""
        return next(self.parameters()).device



    def generate_input_ids_and_labels(self, 
                                      questions: List[str], 
                                      answers: List[str], 
                                      time_series: torch.Tensor,
                                      start_indices: List[int], 
                                      end_indices: List[int],
                                      ablation_mode: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(questions)
        question_prompts = []
        answer_prompts = []
        
        # Verify input lengths
        if len(answers) != batch_size:
            raise ValueError(f"Answers length ({len(answers)}) does not match batch size ({batch_size})")
        if len(start_indices) != batch_size:
            raise ValueError(f"Start indices length ({len(start_indices)}) does not match batch size ({batch_size})")
        if len(end_indices) != batch_size:
            raise ValueError(f"End indices length ({len(end_indices)}) does not match batch size ({batch_size})")
        if time_series.size(0) != batch_size:
            raise ValueError(f"Time series batch size ({time_series.size(0)}) does not match batch size ({batch_size})")
        
        for i in range(batch_size):
            num_local_hint_tokens = end_indices[i] - start_indices[i]
            num_fixed_hint_tokens = self.num_fixed_tokens

            # Create special token sequences for each hint type (may be removed by ablation)
            if ablation_mode == "wo_local_hint":
                local_hint_tokens = ""
            else:
                local_hint_tokens = "<|local_hint|>" * num_local_hint_tokens
            
            if ablation_mode == "wo_fixed_hint":
                fixed_hint_tokens = ""
            else:
                fixed_hint_tokens = "<|fixed_hint|>" * num_fixed_hint_tokens

            # Time series values text (may be removed by ablation)
            str_time_series = ', '.join(f'{(x * 100):.0f}' for x in time_series[i][start_indices[i]:end_indices[i]].tolist())
            if ablation_mode == "wo_windows":
                str_time_series_text = "(removed)"
            else:
                str_time_series_text = str_time_series

            # Build prompt with optional ablations (prompt stays in English)
            question_prompt = f"""
            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps {start_indices[i]} to {end_indices[i]}
            - **Values (scaled by 100):** {str_time_series_text}

            ### Contextual Hints
            - **Per-Step Analysis:** {local_hint_tokens}
            - **Overall Summary Hints:** {fixed_hint_tokens}

            ### Question
            {questions[i]}
            """
            question_prompts.append(question_prompt)
            answer_prompts.append(f"Answer: {answers[i]}")
        question_input = self.tokenizer(
            question_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )
        answer_input = self.tokenizer(
            answer_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )
        full_input_ids = torch.cat([question_input['input_ids'], 
                                    torch.tensor(self.tokenizer.eos_token_id).reshape(1, -1).repeat_interleave(batch_size, 0),
                                    answer_input['input_ids'], 
                                    torch.tensor(self.tokenizer.eos_token_id).reshape(1, -1).repeat_interleave(batch_size, 0)],
                                    dim=1)
        full_attention_mask = torch.cat([question_input['attention_mask'], 
                                         torch.ones(batch_size, 1),
                                         answer_input['attention_mask'], 
                                         torch.ones(batch_size, 1)],
                                         dim=1)
        full_labels = torch.cat([torch.full_like(question_input['input_ids'], -100), 
                                torch.tensor(self.tokenizer.eos_token_id).reshape(1, -1).repeat_interleave(batch_size, 0),
                                answer_input['input_ids'], 
                                torch.tensor(self.tokenizer.eos_token_id).reshape(1, -1).repeat_interleave(batch_size, 0)],
                                dim=1)

        full_labels[full_labels == self.tokenizer.pad_token_id] = -100
        question_length = question_input['input_ids'].shape[1]
        full_labels[:, -1] = self.tokenizer.eos_token_id

        return full_input_ids, full_attention_mask, full_labels, question_length
     
    def get_hint_embeddings(self, 
                            input_ids: torch.Tensor,
                            local_embeddings: torch.Tensor, 
                            start_indices: List[int], 
                            end_indices: List[int]) -> torch.Tensor:
        """Replace special tokens in input embeddings with computed hint embeddings.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            local_embeddings: Local time series embeddings of shape (batch_size, local_seq_len, d_proj).
            start_indices: Start indices for local embeddings extraction.
            end_indices: End indices for local embeddings extraction.
            
        Returns:
            Modified input embeddings with special tokens replaced by hint embeddings.
        """
        input_embeddings = self.model.get_input_embeddings()(input_ids)
        batch_size = local_embeddings.shape[0]
        
        # Get word embeddings dynamically to handle device placement
        word_embeddings = self.model.get_input_embeddings().weight
        source_embeddings = self.perceiver.get_source_embeddings(word_embeddings)
    
        for i in range(batch_size):
            # Process local embeddings using Perceiver
            projected_local_embeddings = self.perceiver.process_local_embeddings(
                local_embeddings[i], 
                source_embeddings, 
                start_indices[i], 
                end_indices[i]
            )
            
            current_input_ids = input_ids[i]
            
            # Replace local hint tokens
            local_hint_positions = (current_input_ids == self.local_hint_token_id).nonzero(as_tuple=True)[0]
            input_embeddings[i, local_hint_positions] = projected_local_embeddings[:len(local_hint_positions)].to(input_embeddings.dtype)

            # Process fixed embeddings using Perceiver
            fixed_hint_positions = (current_input_ids == self.fixed_hint_token_id).nonzero(as_tuple=True)[0]
            if len(fixed_hint_positions) > 0:
                processed_fixed_embeddings = self.perceiver.process_fixed_embeddings(
                    source_embeddings, 
                    len(fixed_hint_positions)
                )
                # Match dtype with input_embeddings
                processed_fixed_embeddings_typed = processed_fixed_embeddings.to(input_embeddings.dtype)
                input_embeddings[i, fixed_hint_positions] = processed_fixed_embeddings_typed

        return input_embeddings

    def forward(self, 
                local_embeddings: torch.Tensor, 
                time_series: torch.Tensor, 
                questions: List[str],
                answers: List[str],
                start_indices: List[int],
                end_indices: List[int],
                return_logits: Optional[bool] = False,
                ablation_mode: Optional[str] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for AXIS model.
        
        Args:
            local_embeddings: Local time series embeddings of shape (batch_size, local_seq_len, d_proj).
            time_series: Raw time series data.
            questions: List of questions to be answered.
            answers: List of ground truth answers.
            start_indices: List of start indices of the time series windows.
            end_indices: List of end indices of the time series windows.
            return_logits: If True, returns logits and loss.
            
        Returns:
            A tuple of (loss, logits) where:
                - loss: Training loss if return_logits is False, else None
                - logits: Model logits if return_logits is True, else None
        """
        input_ids, attention_mask, labels, question_length = self.generate_input_ids_and_labels(
            questions=questions,
            answers=answers,
            time_series=time_series,
            start_indices=start_indices,
            end_indices=end_indices,
            ablation_mode=ablation_mode
        )
        
        # Move tensors to model device for multi-GPU compatibility
        device = self.get_device()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        input_embeddings = self.get_hint_embeddings(
            input_ids=input_ids, 
            local_embeddings=local_embeddings, 
            start_indices=start_indices, 
            end_indices=end_indices
        )
        
        # Model forward pass
        outputs = self.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )
        
        if return_logits:
            return outputs.loss, outputs.logits
        else:
            return outputs.loss
    
    def generate(self, 
                 local_embeddings: torch.Tensor, 
                 time_series: torch.Tensor, 
                 questions: List[str],
                 answers: List[str],
                 start_indices: List[int],
                 end_indices: List[int],
                 ablation_mode: Optional[str] = None) -> List[str]:
        input_ids, attention_mask, labels, question_length = self.generate_input_ids_and_labels(
            questions=questions,
            answers=answers,
            time_series=time_series,
            start_indices=start_indices,
            end_indices=end_indices,
            ablation_mode=ablation_mode
        )
        
        # Move tensors to model device for multi-GPU compatibility
        device = self.get_device()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        input_embeddings = self.get_hint_embeddings(
            input_ids=input_ids, 
            local_embeddings=local_embeddings, 
            start_indices=start_indices, 
            end_indices=end_indices
        )
        generate_kwargs = dict(
            do_sample=False,
            num_beams=5,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            length_penalty=1,
        )
        outputs = self.model.generate(
            inputs_embeds=input_embeddings[:, :question_length, :],
            attention_mask=attention_mask[:, :question_length],
            max_new_tokens=1000,
            # bad_words_ids=[[think_token_id]],
            **generate_kwargs
        )
        outputs_word = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return outputs_word

    def _get_most_likely_text(
        self,
        outputs: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """Get the most likely text from model outputs.
        
        Args:
            outputs: Model outputs containing logits.
            tokenizer: Tokenizer for decoding.
            skip_special_tokens: Whether to skip special tokens in decoding.
        
        Returns:
            Decoded text or list of decoded texts.
        """
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs[0]
            
        predicted_token_ids = logits.argmax(dim=-1)
        tokenizer = self.tokenizer
        if predicted_token_ids.dim() > 1:
            texts = []
            for ids in predicted_token_ids:
                text = tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
                texts.append(text)
            return texts
        else:
            return tokenizer.decode(predicted_token_ids, skip_special_tokens=skip_special_tokens)


class AXISCombinedModel(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.axis = AXIS(config)
        self.ts_pretrain_model = TimeSeriesPretrainModel(config)
    
    def forward(self, 
                padded_sequences: torch.Tensor,
                attention_masks: torch.Tensor,
                questions: List[str],
                answers: List[str],
                start_indices: List[int],
                end_indices: List[int],
                return_logits: Optional[bool] = False,
                ablation_mode: Optional[str] = None) -> torch.Tensor:
        """Forward pass for the combined model.
        
        Args:
            padded_sequences: Padded time series sequences
            attention_masks: Attention masks for the sequences
            questions: List of questions
            answers: List of answers
            start_indices: Start indices for local embeddings
            end_indices: End indices for local embeddings
            return_logits: Whether to return logits along with loss
            
        Returns:
            Loss tensor (and optionally logits)
        """
        # Generate embeddings using pretrain model
        # Check if ts training is enabled

        if not self.config.enable_ts_train:
            with torch.no_grad():
                local_embeddings = self.ts_pretrain_model(
                    padded_sequences, 
                    mask=attention_masks
                )
        else:
            local_embeddings = self.ts_pretrain_model(
                padded_sequences, 
                mask=attention_masks
            )
        
        # Generate predictions using axis model
        return self.axis(
            local_embeddings=local_embeddings,
            time_series=padded_sequences,
            questions=questions,
            answers=answers,
            start_indices=start_indices,
            end_indices=end_indices,
            return_logits=return_logits,
            ablation_mode=ablation_mode
        )
    
    def generate(self,
                 padded_sequences: torch.Tensor,
                 attention_masks: torch.Tensor,
                 questions: List[str],
                 answers: List[str],
                 start_indices: List[int],
                 end_indices: List[int],
                 return_logits: Optional[bool] = False,
                 ablation_mode: Optional[str] = None) -> Tuple[List[str], Optional[List[torch.Tensor]]]:
        """Generate responses using the combined model.
        
        Args:
            padded_sequences: Padded time series sequences
            attention_masks: Attention masks for the sequences
            questions: List of questions
            answers: List of answers (used for prompt formatting)
            start_indices: Start indices for local embeddings
            end_indices: End indices for local embeddings
            
        Returns:
            Generated text responses
        """
        # Generate embeddings using pretrain model
        with torch.no_grad():
            global_embeddings, local_embeddings = self.ts_pretrain_model(
                padded_sequences, 
                mask=attention_masks
            )
        
        # Generate responses using axis model
        answer = self.axis.generate(
                local_embeddings=local_embeddings,
                time_series=padded_sequences,
                questions=questions,
                answers=answers,
                start_indices=start_indices,
                end_indices=end_indices,
                ablation_mode=ablation_mode
            )
        if return_logits:
            return answer
        else:
            logits = self.ts_pretrain_model.anomaly_head(local_embeddings)
            anomaly_scores = [logits[i, start_indices[i]:end_indices[i], :] for i in range(len(start_indices))]
            return answer, anomaly_scores
        
