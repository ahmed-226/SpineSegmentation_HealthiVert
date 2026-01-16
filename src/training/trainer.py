"""
Training pipeline for VerSe2019 Vertebrae Segmentation
PyTorch Implementation
"""
import os
import time
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from ..models.networks import (
    SimpleUNet, SpatialConfigurationNet, SegmentationUNet
)
from ..data.dataset import (
    SpineLocalizationDataset, VertebraeLocalizationDataset,
    VertebraeSegmentationDataset, load_landmarks_csv, load_id_list
)
from ..utils.heatmap_utils import (
    generate_heatmap_target, LearnableSigmas,
    extract_all_landmarks, compute_landmark_distance_error
)
from ..config import PipelineConfig


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base trainer class with common functionality"""
    
    def __init__(
        self,
        config: PipelineConfig,
        stage: str,
        output_dir: str,
        cv_fold: int = 0,
        resume_checkpoint: Optional[str] = None
    ):
        self.config = config
        self.stage = stage
        self.output_dir = Path(output_dir)
        self.cv_fold = cv_fold
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        
        # Training state
        self.current_iter = 0
        self.best_val_loss = float('inf')
        
        # Resume from checkpoint
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
    
    def init_tensorboard(self):
        """Initialize TensorBoard writer"""
        log_dir = self.output_dir / 'logs' / f'fold_{self.cv_fold}'
        self.writer = SummaryWriter(log_dir=str(log_dir))
        logger.info(f"TensorBoard logs: {log_dir}")
    
    def save_checkpoint(self, filename: str = 'checkpoint.pth'):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': self.current_iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.output_dir / 'checkpoints' / f'fold_{self.cv_fold}' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.current_iter = checkpoint['iteration']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from iteration {self.current_iter}")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = 'train'):
        """Log metrics to TensorBoard"""
        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, self.current_iter)
    
    def train(self):
        """Main training loop - to be implemented by subclasses"""
        raise NotImplementedError
    
    def train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Single training step - to be implemented by subclasses"""
        raise NotImplementedError
    
    def validate(self) -> Dict[str, float]:
        """Validation loop - to be implemented by subclasses"""
        raise NotImplementedError


class SpineLocalizationTrainer(BaseTrainer):
    """Trainer for Stage 1: Spine Localization"""
    
    def __init__(
        self,
        config: PipelineConfig,
        output_dir: str,
        cv_fold: int = 0,
        resume_checkpoint: Optional[str] = None,
        model: Optional[nn.Module] = None,
        train_dataset: Optional[SpineLocalizationDataset] = None,
        val_dataset: Optional[SpineLocalizationDataset] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__(config, 'spine_localization', output_dir, cv_fold, resume_checkpoint)
        self.stage_config = config.spine_localization
        
        # Override device if provided
        if device is not None:
            self.device = device
        
        # Initialize components
        if model is not None:
            self.model = model.to(self.device)
            logger.info(f"Using provided model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        else:
            self._init_model()
        
        self._init_optimizer()
        
        if train_dataset is not None and val_dataset is not None:
            self._init_data_loaders_from_datasets(train_dataset, val_dataset)
        else:
            self._init_data_loaders()
        
        self.init_tensorboard()
    
    def _init_data_loaders_from_datasets(self, train_dataset, val_dataset):
        """Initialize data loaders from provided datasets"""
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.stage_config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 4,
            pin_memory=self.config.pin_memory if hasattr(self.config, 'pin_memory') else True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 4
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    def _init_model(self):
        """Initialize the network"""
        self.model = SimpleUNet(
            in_channels=1,
            num_labels=self.stage_config.num_labels,
            num_filters_base=self.stage_config.num_filters_base,
            num_levels=self.stage_config.num_levels,
            dropout_ratio=self.stage_config.dropout_ratio,
            heatmap_initialization=self.stage_config.heatmap_initialization
        ).to(self.device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.stage_config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Exponential decay scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.1 ** (1 / self.stage_config.max_iterations)
        )
    
    def _init_data_loaders(self):
        """Initialize data loaders"""
        data_folder = self.config.data.data_folder
        
        # Load ID lists
        train_ids = load_id_list(os.path.join(data_folder, self.config.data.train_list))
        val_ids = load_id_list(os.path.join(data_folder, self.config.data.val_list))
        
        # Load landmarks
        landmarks_dict = load_landmarks_csv(
            os.path.join(data_folder, self.config.data.landmarks_file)
        )
        
        # Create datasets
        train_dataset = SpineLocalizationDataset(
            data_folder=data_folder,
            id_list=train_ids,
            landmarks_dict=landmarks_dict,
            image_size=self.stage_config.image_size,
            image_spacing=self.stage_config.image_spacing,
            heatmap_sigma=self.stage_config.heatmap_sigma,
            is_training=True
        )
        
        val_dataset = SpineLocalizationDataset(
            data_folder=data_folder,
            id_list=val_ids,
            landmarks_dict=landmarks_dict,
            image_size=self.stage_config.image_size,
            image_spacing=self.stage_config.image_spacing,
            heatmap_sigma=self.stage_config.heatmap_sigma,
            is_training=False
        )
        
        # Create loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.stage_config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    def train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Single training step"""
        image = batch['image'].to(self.device)
        target = batch['target'].to(self.device)
        
        # Forward pass
        prediction = self.model(image)
        
        # L2 loss
        loss = F.mse_loss(prediction, target, reduction='sum') / (2 * image.size(0))
        
        return {'loss': loss}
    
    def validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                image = batch['image'].to(self.device)
                target = batch['target'].to(self.device)
                
                prediction = self.model(image)
                loss = F.mse_loss(prediction, target, reduction='sum') / 2
                
                total_loss += loss.item()
                num_samples += image.size(0)
        
        self.model.train()
        
        avg_loss = total_loss / num_samples
        return {'loss': avg_loss}
    
    def train(self):
        """Main training loop"""
        logger.info("Starting spine localization training...")
        
        train_iter = iter(self.train_loader)
        self.model.train()
        
        start_time = time.time()
        
        while self.current_iter < self.stage_config.max_iterations:
            # Get next batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Training step
            self.optimizer.zero_grad()
            losses = self.train_step(batch)
            losses['loss'].backward()
            self.optimizer.step()
            self.scheduler.step()
            
            self.current_iter += 1
            
            # Logging
            if self.current_iter % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Iter {self.current_iter}/{self.stage_config.max_iterations} | "
                    f"Loss: {losses['loss'].item():.6f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
                self.log_metrics({'loss': losses['loss'].item()}, prefix='train')
            
            # Validation
            if self.current_iter % self.stage_config.test_interval == 0:
                val_metrics = self.validate()
                logger.info(f"Validation loss: {val_metrics['loss']:.6f}")
                self.log_metrics(val_metrics, prefix='val')
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pth')
            
            # Regular checkpoint
            if self.current_iter % self.stage_config.snapshot_interval == 0:
                self.save_checkpoint(f'checkpoint_{self.current_iter}.pth')
        
        logger.info("Training completed!")
        self.writer.close()


class VertebraeLocalizationTrainer(BaseTrainer):
    """Trainer for Stage 2: Vertebrae Localization with SpatialConfigurationNet"""
    
    def __init__(
        self,
        config: PipelineConfig,
        output_dir: str,
        cv_fold: int = 0,
        resume_checkpoint: Optional[str] = None,
        model: Optional[nn.Module] = None,
        train_dataset: Optional[VertebraeLocalizationDataset] = None,
        val_dataset: Optional[VertebraeLocalizationDataset] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__(config, 'vertebrae_localization', output_dir, cv_fold, resume_checkpoint)
        self.stage_config = config.vertebrae_localization
        
        # Override device if provided
        if device is not None:
            self.device = device
        
        # Initialize components
        if model is not None:
            self.model = model.to(self.device)
            logger.info(f"Using provided model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        else:
            self._init_model()
        
        self._init_optimizer()
        
        if train_dataset is not None and val_dataset is not None:
            self._init_data_loaders_from_datasets(train_dataset, val_dataset)
        else:
            self._init_data_loaders()
        
        self.init_tensorboard()
    
    def _init_data_loaders_from_datasets(self, train_dataset, val_dataset):
        """Initialize data loaders from provided datasets"""
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.stage_config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 4,
            pin_memory=self.config.pin_memory if hasattr(self.config, 'pin_memory') else True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 4
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    def _init_model(self):
        """Initialize the network"""
        self.model = SpatialConfigurationNet(
            in_channels=1,
            num_labels=self.stage_config.num_landmarks,
            num_filters_base=self.stage_config.num_filters_base,
            num_levels=self.stage_config.num_levels,
            spatial_downsample=self.stage_config.spatial_downsample,
            dropout_ratio=self.stage_config.dropout_ratio,
            activation=self.stage_config.activation.replace('_', ''),
            local_activation=self.stage_config.local_activation,
            spatial_activation=self.stage_config.spatial_activation
        ).to(self.device)
        
        # Learnable sigmas
        if self.stage_config.learnable_sigma:
            self.sigmas = LearnableSigmas(
                num_landmarks=self.stage_config.num_landmarks,
                initial_sigma=self.stage_config.heatmap_sigma
            ).to(self.device)
        else:
            self.sigmas = torch.full(
                (self.stage_config.num_landmarks,),
                self.stage_config.heatmap_sigma,
                device=self.device
            )
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler"""
        params = list(self.model.parameters())
        
        if self.stage_config.learnable_sigma:
            params += list(self.sigmas.parameters())
        
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.stage_config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.1 ** (1 / self.stage_config.max_iterations)
        )
    
    def _init_data_loaders(self):
        """Initialize data loaders"""
        data_folder = self.config.data.data_folder
        
        train_ids = load_id_list(os.path.join(data_folder, self.config.data.train_list))
        val_ids = load_id_list(os.path.join(data_folder, self.config.data.val_list))
        
        landmarks_dict = load_landmarks_csv(
            os.path.join(data_folder, self.config.data.landmarks_file)
        )
        
        train_dataset = VertebraeLocalizationDataset(
            data_folder=data_folder,
            id_list=train_ids,
            landmarks_dict=landmarks_dict,
            image_size=self.stage_config.image_size,
            image_spacing=self.stage_config.image_spacing,
            num_landmarks=self.stage_config.num_landmarks,
            heatmap_sigma=self.stage_config.heatmap_sigma,
            is_training=True
        )
        
        val_dataset = VertebraeLocalizationDataset(
            data_folder=data_folder,
            id_list=val_ids,
            landmarks_dict=landmarks_dict,
            image_size=self.stage_config.image_size,
            image_spacing=self.stage_config.image_spacing,
            num_landmarks=self.stage_config.num_landmarks,
            heatmap_sigma=self.stage_config.heatmap_sigma,
            is_training=False
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.stage_config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    def train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Single training step"""
        image = batch['image'].to(self.device)
        landmarks = batch['landmarks'].to(self.device)
        
        # Forward pass
        heatmaps, local_heatmaps, spatial_heatmaps = self.model(image)
        
        # Get sigmas
        if self.stage_config.learnable_sigma:
            sigmas = self.sigmas()
        else:
            sigmas = self.sigmas
        
        # Generate target heatmaps
        target_heatmaps = generate_heatmap_target(
            shape=image.shape[2:],
            landmarks=landmarks,
            sigmas=sigmas,
            device=self.device
        )
        
        # L2 loss
        loss = F.mse_loss(heatmaps, target_heatmaps, reduction='sum') / (2 * image.size(0))
        
        # Sigma regularization
        if self.stage_config.learnable_sigma:
            valid_mask = landmarks[:, :, 0].sum(dim=0)  # [num_landmarks]
            sigma_reg = self.sigmas.regularization_loss(
                valid_mask=valid_mask,
                weight=self.stage_config.sigma_regularization
            )
            loss = loss + sigma_reg
        
        return {
            'loss': loss,
            'heatmap_loss': loss.detach()
        }
    
    def validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        total_loss = 0.0
        total_distance = 0.0
        num_samples = 0
        num_valid_landmarks = 0
        
        if self.stage_config.learnable_sigma:
            sigmas = self.sigmas()
        else:
            sigmas = self.sigmas
        
        with torch.no_grad():
            for batch in self.val_loader:
                image = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                heatmaps, _, _ = self.model(image)
                
                target_heatmaps = generate_heatmap_target(
                    shape=image.shape[2:],
                    landmarks=landmarks,
                    sigmas=sigmas,
                    device=self.device
                )
                
                loss = F.mse_loss(heatmaps, target_heatmaps, reduction='sum') / 2
                total_loss += loss.item()
                
                # Extract landmarks and compute distance
                pred_landmarks, _ = extract_all_landmarks(heatmaps[0])
                pred_landmarks = pred_landmarks.unsqueeze(0)  # Add batch dim
                
                distances = compute_landmark_distance_error(
                    pred_landmarks, landmarks,
                    spacing=self.stage_config.image_spacing
                )
                
                valid_mask = (landmarks[:, :, 0] > 0.5) & (pred_landmarks[:, :, 0] > 0.5)
                if valid_mask.sum() > 0:
                    total_distance += distances[valid_mask].sum().item()
                    num_valid_landmarks += valid_mask.sum().item()
                
                num_samples += image.size(0)
        
        self.model.train()
        
        avg_loss = total_loss / num_samples
        avg_distance = total_distance / num_valid_landmarks if num_valid_landmarks > 0 else 0
        
        return {
            'loss': avg_loss,
            'mean_distance_mm': avg_distance
        }
    
    def train(self):
        """Main training loop"""
        logger.info("Starting vertebrae localization training...")
        
        train_iter = iter(self.train_loader)
        self.model.train()
        
        start_time = time.time()
        
        while self.current_iter < self.stage_config.max_iterations:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            self.optimizer.zero_grad()
            losses = self.train_step(batch)
            losses['loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.stage_config.gradient_clip_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            self.current_iter += 1
            
            # Logging
            if self.current_iter % 100 == 0:
                elapsed = time.time() - start_time
                
                sigma_str = ""
                if self.stage_config.learnable_sigma:
                    sigmas = self.sigmas()
                    sigma_str = f" | Sigma: [{sigmas.min():.2f}, {sigmas.max():.2f}]"
                
                logger.info(
                    f"Iter {self.current_iter}/{self.stage_config.max_iterations} | "
                    f"Loss: {losses['loss'].item():.6f}{sigma_str} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
                self.log_metrics({'loss': losses['loss'].item()}, prefix='train')
            
            # Validation
            if self.current_iter % self.stage_config.test_interval == 0:
                val_metrics = self.validate()
                logger.info(
                    f"Validation - Loss: {val_metrics['loss']:.6f}, "
                    f"Mean distance: {val_metrics['mean_distance_mm']:.2f} mm"
                )
                self.log_metrics(val_metrics, prefix='val')
                
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pth')
            
            # Regular checkpoint
            if self.current_iter % self.stage_config.snapshot_interval == 0:
                self.save_checkpoint(f'checkpoint_{self.current_iter}.pth')
        
        logger.info("Training completed!")
        self.writer.close()


class VertebraeSegmentationTrainer(BaseTrainer):
    """Trainer for Stage 3: Vertebrae Segmentation"""
    
    def __init__(
        self,
        config: PipelineConfig,
        output_dir: str,
        labels_folder: str = None,
        cv_fold: int = 0,
        resume_checkpoint: Optional[str] = None,
        model: Optional[nn.Module] = None,
        train_dataset: Optional[VertebraeSegmentationDataset] = None,
        val_dataset: Optional[VertebraeSegmentationDataset] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__(config, 'vertebrae_segmentation', output_dir, cv_fold, resume_checkpoint)
        self.stage_config = config.vertebrae_segmentation
        self.labels_folder = labels_folder
        
        # Override device if provided
        if device is not None:
            self.device = device
        
        # Initialize components
        if model is not None:
            self.model = model.to(self.device)
            logger.info(f"Using provided model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        else:
            self._init_model()
        
        self._init_optimizer()
        
        if train_dataset is not None and val_dataset is not None:
            self._init_data_loaders_from_datasets(train_dataset, val_dataset)
        else:
            self._init_data_loaders()
        
        self.init_tensorboard()
    
    def _init_data_loaders_from_datasets(self, train_dataset, val_dataset):
        """Initialize data loaders from provided datasets"""
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.stage_config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 4,
            pin_memory=self.config.pin_memory if hasattr(self.config, 'pin_memory') else True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 4
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        self._init_optimizer()
        self._init_data_loaders()
        self.init_tensorboard()
    
    def _init_model(self):
        """Initialize the network"""
        self.model = SegmentationUNet(
            in_channels=1,
            num_classes=self.stage_config.num_labels,
            num_filters_base=self.stage_config.num_filters_base,
            num_levels=self.stage_config.num_levels,
            dropout_ratio=self.stage_config.dropout_ratio
        ).to(self.device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.stage_config.learning_rate
        )
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.1 ** (1 / self.stage_config.max_iterations)
        )
    
    def _init_data_loaders(self):
        """Initialize data loaders"""
        data_folder = self.config.data.data_folder
        
        train_ids = load_id_list(os.path.join(data_folder, self.config.data.train_list))
        val_ids = load_id_list(os.path.join(data_folder, self.config.data.val_list))
        
        landmarks_dict = load_landmarks_csv(
            os.path.join(data_folder, self.config.data.landmarks_file)
        )
        
        train_dataset = VertebraeSegmentationDataset(
            data_folder=data_folder,
            id_list=train_ids,
            landmarks_dict=landmarks_dict,
            labels_folder=self.labels_folder,
            image_size=self.stage_config.image_size,
            image_spacing=self.stage_config.image_spacing,
            num_classes=self.stage_config.num_labels,
            is_training=True
        )
        
        val_dataset = VertebraeSegmentationDataset(
            data_folder=data_folder,
            id_list=val_ids,
            landmarks_dict=landmarks_dict,
            labels_folder=self.labels_folder,
            image_size=self.stage_config.image_size,
            image_spacing=self.stage_config.image_spacing,
            num_classes=self.stage_config.num_labels,
            is_training=False
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.stage_config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    def train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Single training step"""
        image = batch['image'].to(self.device)
        label = batch['label'].to(self.device)
        
        # Forward pass
        logits = self.model(image)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, label)
        
        return {'loss': loss}
    
    def compute_dice(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """Compute Dice coefficient for each class"""
        dice_scores = torch.zeros(num_classes, device=prediction.device)
        
        for c in range(num_classes):
            pred_c = (prediction == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            if union > 0:
                dice_scores[c] = 2 * intersection / union
            else:
                dice_scores[c] = 1.0  # Both empty
        
        return dice_scores
    
    def validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        total_loss = 0.0
        total_dice = torch.zeros(self.stage_config.num_labels, device=self.device)
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                image = batch['image'].to(self.device)
                label = batch['label'].to(self.device)
                
                logits = self.model(image)
                loss = F.cross_entropy(logits, label)
                
                prediction = torch.argmax(logits, dim=1)
                dice = self.compute_dice(prediction, label, self.stage_config.num_labels)
                
                total_loss += loss.item()
                total_dice += dice
                num_samples += 1
        
        self.model.train()
        
        avg_loss = total_loss / num_samples
        avg_dice = total_dice / num_samples
        mean_dice = avg_dice[1:].mean().item()  # Exclude background
        
        return {
            'loss': avg_loss,
            'mean_dice': mean_dice
        }
    
    def train(self):
        """Main training loop"""
        logger.info("Starting vertebrae segmentation training...")
        
        train_iter = iter(self.train_loader)
        self.model.train()
        
        start_time = time.time()
        
        while self.current_iter < self.stage_config.max_iterations:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            self.optimizer.zero_grad()
            losses = self.train_step(batch)
            losses['loss'].backward()
            self.optimizer.step()
            self.scheduler.step()
            
            self.current_iter += 1
            
            if self.current_iter % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Iter {self.current_iter}/{self.stage_config.max_iterations} | "
                    f"Loss: {losses['loss'].item():.6f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
                self.log_metrics({'loss': losses['loss'].item()}, prefix='train')
            
            if self.current_iter % self.stage_config.test_interval == 0:
                val_metrics = self.validate()
                logger.info(
                    f"Validation - Loss: {val_metrics['loss']:.6f}, "
                    f"Mean Dice: {val_metrics['mean_dice']:.4f}"
                )
                self.log_metrics(val_metrics, prefix='val')
                
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pth')
            
            if self.current_iter % self.stage_config.snapshot_interval == 0:
                self.save_checkpoint(f'checkpoint_{self.current_iter}.pth')
        
        logger.info("Training completed!")
        self.writer.close()


def train_pipeline(
    config: PipelineConfig,
    stage: str,
    output_dir: str,
    labels_folder: Optional[str] = None,
    cv_fold: int = 0,
    resume_checkpoint: Optional[str] = None
):
    """
    Train a specific stage of the pipeline.
    
    Args:
        config: Pipeline configuration
        stage: 'spine_localization', 'vertebrae_localization', or 'vertebrae_segmentation'
        output_dir: Output directory for checkpoints and logs
        labels_folder: Folder with segmentation labels (required for segmentation)
        cv_fold: Cross-validation fold
        resume_checkpoint: Path to checkpoint to resume from
    """
    if stage == 'spine_localization':
        trainer = SpineLocalizationTrainer(
            config, output_dir, cv_fold, resume_checkpoint
        )
    elif stage == 'vertebrae_localization':
        trainer = VertebraeLocalizationTrainer(
            config, output_dir, cv_fold, resume_checkpoint
        )
    elif stage == 'vertebrae_segmentation':
        if labels_folder is None:
            raise ValueError("labels_folder is required for segmentation training")
        trainer = VertebraeSegmentationTrainer(
            config, output_dir, labels_folder, cv_fold, resume_checkpoint
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    trainer.train()
