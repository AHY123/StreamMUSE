import torch
import logging
from typing import Any, Optional, Dict
import gc
import psutil
import os


class ErrorRecoveryMixin:
    """
    Mixin providing error handling and recovery mechanisms for reward models.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_count = 0
        self.max_errors = 10
        self.original_batch_size = None
        
    def handle_cuda_oom(self, error: Exception, batch: Any = None) -> bool:
        """
        Handle CUDA out of memory errors by reducing batch size and clearing cache.
        
        Args:
            error: CUDA OOM exception
            batch: Current batch being processed
            
        Returns:
            bool: True if recovery successful, False otherwise
        """
        self.logger.warning(f"CUDA OOM detected: {error}")
        
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            # Log memory status
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                self.logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            return True
            
        except Exception as recovery_error:
            self.logger.error(f"Failed to recover from CUDA OOM: {recovery_error}")
            return False
    
    def handle_nan_gradient(self, error: Exception, model: torch.nn.Module) -> bool:
        """
        Handle NaN gradients by clipping and resetting optimizer state.
        
        Args:
            error: NaN/Inf gradient exception
            model: Model with problematic gradients
            
        Returns:
            bool: True if recovery successful, False otherwise
        """
        self.logger.warning(f"NaN/Inf gradient detected: {error}")
        
        try:
            # Zero out problematic gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data = torch.where(
                        torch.isnan(param.grad.data) | torch.isinf(param.grad.data),
                        torch.zeros_like(param.grad.data),
                        param.grad.data
                    )
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            self.logger.info("Successfully cleaned NaN/Inf gradients")
            return True
            
        except Exception as recovery_error:
            self.logger.error(f"Failed to recover from NaN gradients: {recovery_error}")
            return False
    
    def validate_input_tensors(self, **tensors) -> bool:
        """
        Validate input tensors for common issues.
        
        Args:
            **tensors: Named tensors to validate
            
        Returns:
            bool: True if all tensors are valid, False otherwise
        """
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                self.logger.error(f"Input '{name}' is not a tensor: {type(tensor)}")
                return False
            
            if tensor.numel() == 0:
                self.logger.error(f"Input '{name}' is empty tensor")
                return False
            
            if torch.isnan(tensor).any():
                self.logger.error(f"Input '{name}' contains NaN values")
                return False
            
            if torch.isinf(tensor).any():
                self.logger.error(f"Input '{name}' contains Inf values")
                return False
            
            # Check vocabulary bounds for token sequences
            if 'token' in name.lower() or 'sequence' in name.lower():
                if tensor.dtype in [torch.long, torch.int]:
                    if (tensor < 0).any() or (tensor >= 3205).any():  # N_TOKENS = 3205
                        self.logger.warning(f"Input '{name}' contains out-of-vocabulary tokens")
        
        return True
    
    def validate_sequence_lengths(self, melody_tokens: torch.Tensor, accompaniment_tokens: torch.Tensor) -> bool:
        """
        Validate sequence lengths and compatibility.
        
        Args:
            melody_tokens: Melody token sequence
            accompaniment_tokens: Accompaniment token sequence
            
        Returns:
            bool: True if sequences are compatible, False otherwise
        """
        if len(melody_tokens.shape) != 1 or len(accompaniment_tokens.shape) != 1:
            self.logger.error("Input sequences must be 1D tensors")
            return False
        
        if len(melody_tokens) == 0 or len(accompaniment_tokens) == 0:
            self.logger.error("Input sequences cannot be empty")
            return False
        
        # Check if sequences are reasonable length (not too short or too long)
        min_length, max_length = 10, 2048
        for name, seq in [("melody", melody_tokens), ("accompaniment", accompaniment_tokens)]:
            if len(seq) < min_length:
                self.logger.warning(f"{name} sequence is very short: {len(seq)} tokens")
            elif len(seq) > max_length:
                self.logger.warning(f"{name} sequence is very long: {len(seq)} tokens")
        
        return True
    
    def safe_forward(self, forward_func, *args, **kwargs):
        """
        Safely execute forward pass with error handling.
        
        Args:
            forward_func: Forward function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of forward function or None if failed
        """
        try:
            return forward_func(*args, **kwargs)
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                if self.handle_cuda_oom(e):
                    # Retry once after recovery
                    try:
                        return forward_func(*args, **kwargs)
                    except Exception as retry_error:
                        self.logger.error(f"Failed retry after CUDA OOM recovery: {retry_error}")
                        return None
            elif "NaN" in str(e) or "Inf" in str(e):
                self.logger.error(f"NaN/Inf detected in forward pass: {e}")
                return None
            else:
                self.logger.error(f"Runtime error in forward pass: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Unexpected error in forward pass: {e}")
            return None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = {}
        
        # System memory
        process = psutil.Process(os.getpid())
        memory_info['system_memory_mb'] = process.memory_info().rss / 1024**2
        
        # GPU memory if available
        if torch.cuda.is_available():
            memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            memory_info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            memory_info['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        
        return memory_info
    
    def should_continue_training(self) -> bool:
        """Check if training should continue based on error count."""
        return self.error_count < self.max_errors
    
    def log_error_and_increment(self, error: Exception, context: str = ""):
        """Log error and increment error counter."""
        self.error_count += 1
        self.logger.error(f"Error {self.error_count}/{self.max_errors} in {context}: {error}")
        
        if self.error_count >= self.max_errors:
            self.logger.critical(f"Maximum error count ({self.max_errors}) reached. Training should be stopped.")
    
    def reset_error_count(self):
        """Reset error counter (call after successful operations)."""
        if self.error_count > 0:
            self.logger.info(f"Resetting error count from {self.error_count} to 0")
            self.error_count = 0