import datetime
import itertools
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import numpy as np
from typing import Tuple, List, Dict, Any, Union, Optional
from dataclasses import dataclass

from src.models.AXIS.ts_encoder_bi_bias import TimeSeriesEncoder
from experiments.configs.axis_config import AXISConfig, default_config

import warnings
warnings.filterwarnings("ignore")

@dataclass
class PretrainBatch:
    """Batch structure for pretraining tasks."""
    time_series: torch.Tensor
    labels: torch.Tensor
    masked_time_series: torch.Tensor
    mask_indices: torch.Tensor
    

class TimeSeriesPretrainModel(nn.Module):
    """Model for time series pretraining with masked reconstruction and anomaly detection."""
    
    def __init__(self, config: AXISConfig):
        super().__init__()
        self.config = config
        
        # Extract TimeSeriesEncoder parameters from config
        ts_config = config.ts_config
        self.ts_encoder = TimeSeriesEncoder(
            d_model=ts_config.d_model,
            d_proj=ts_config.d_proj,
            patch_size=ts_config.patch_size,
            num_layers=ts_config.num_layers,
            num_heads=ts_config.num_heads,
            d_ff_dropout=ts_config.d_ff_dropout,
            use_rope=ts_config.use_rope,
            num_features=ts_config.num_features,
            activation=ts_config.activation
        )
        
        # Masked reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.ts_config.d_proj, config.ts_config.d_proj * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ts_config.d_proj * 4, config.ts_config.d_proj * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ts_config.d_proj * 4, 1)  # (B, seq_len, num_features, 1)
        )
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(config.ts_config.d_proj, config.ts_config.d_proj // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ts_config.d_proj // 2, 2)  # (B, seq_len, num_features, 2) for binary classification
        )
        
    def forward(self, time_series: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Forward pass through the encoder."""
        local_embeddings = self.ts_encoder(time_series, mask)
        return local_embeddings

    def masked_reconstruction_loss(self, 
                                   local_embeddings: torch.Tensor,  # (B, seq_len, num_features, d_proj)
                                   original_time_series: torch.Tensor,  # (B, seq_len, num_features),
                                   mask: torch.Tensor  # (B, seq_len)
                                   ) -> torch.Tensor:
        """Compute masked reconstruction loss."""
        batch_size, seq_len, num_features = original_time_series.shape
        patch_size = self.config.ts_config.patch_size
        
        # Ensure data type consistency
        mask = mask.bool()
        
        # Only compute loss for masked positions
        # local_embeddings: [B, seq_len, num_features, d_proj]
        # Predict original values through reconstruction head
        reconstructed = self.reconstruction_head(local_embeddings)  # (B, seq_len, num_features, 1)
        reconstructed = reconstructed.view(batch_size, seq_len, num_features)  
        
        # Only compute loss for masked positions
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_features)  # (B, seq_len, num_features)
        reconstruction_loss = F.mse_loss(
            reconstructed[mask_expanded],
            original_time_series[mask_expanded]
        )
        return reconstruction_loss
    
    def anomaly_detection_loss(self, 
                               local_embeddings: torch.Tensor,  # (B, seq_len, num_features, d_proj)
                               labels: torch.Tensor) -> torch.Tensor:  # (B, seq_len)
        """Compute anomaly detection loss for each timestep."""
        # Project local embeddings to anomaly scores
        logits = self.anomaly_head(local_embeddings)  # (B, seq_len, num_features, 2)
        logits = torch.mean(logits, dim=-2)  # Average over num_features to get (B, seq_len, 2)
                            
        
        # Reshape for loss computation
        batch_size, seq_len, _ = logits.shape
        logits = logits.view(-1, 2)  # (B*seq_len, 2)
        labels = labels.view(-1)  # (B*seq_len)
        labels = (labels > 0.5).long()
        # Create mask for valid labels (not padding)
        valid_mask = (labels != -1)
        
        # Compute loss only on valid timesteps
        if valid_mask.sum() > 0:
            anomaly_loss = F.cross_entropy(
                logits[valid_mask],
                labels[valid_mask]
            )
        else:
            anomaly_loss = torch.tensor(0.0, device=logits.device)
            
        return anomaly_loss
