"""
Training utilities for brain tumor segmentation models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
from typing import Dict, Optional
import numpy as np

from evaluation.metrics import dice_coefficient, evaluate_model


class Trainer:
    """Generic trainer for segmentation models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        config,
        model_name: str = "model"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            config: Configuration object
            model_name: Name for saving checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.model_name = model_name
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=True
        )
        
        # Mixed precision training
        self.use_amp = config.USE_MIXED_PRECISION and hasattr(torch.cuda, 'amp')
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # TensorBoard writer
        log_dir = config.LOGS_DIR / model_name
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.train_losses = []
        self.val_dice_scores = []
        
        # Early stopping
        self.patience_counter = 0
        self.early_stopping_patience = config.EARLY_STOPPING_PATIENCE
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)
                    loss = self.criterion(predictions, masks)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                loss = self.criterion(predictions, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.LOG_INTERVAL == 0:
                print(f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}')
                
                # Log to TensorBoard
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
        
        avg_loss = running_loss / num_batches
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        # Evaluate on validation set
        metrics = evaluate_model(self.model, self.val_loader, self.device)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_dice': self.best_val_dice,
            'train_losses': self.train_losses,
            'val_dice_scores': self.val_dice_scores
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.config.SAVED_MODELS_DIR / f"{self.model_name}_latest.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.config.SAVED_MODELS_DIR / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with DSC: {self.best_val_dice:.4f}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_dice = checkpoint['best_val_dice']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_dice_scores = checkpoint.get('val_dice_scores', [])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int, resume_from: Optional[Path] = None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            val_dice = val_metrics['dice']
            self.val_dice_scores.append(val_dice)
            
            # Update learning rate
            self.scheduler.step(val_dice)
            
            # Check if best model
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Log epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val DSC: {val_dice:.4f}")
            print(f"Val IoU: {val_metrics['iou']:.4f}")
            print(f"Best Val DSC: {self.best_val_dice:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to TensorBoard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Dice', val_dice, epoch)
            self.writer.add_scalar('Epoch/Val_IoU', val_metrics['iou'], epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        print(f"Best validation DSC: {self.best_val_dice:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(is_best=False)
        self.writer.close()

