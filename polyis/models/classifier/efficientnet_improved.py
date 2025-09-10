import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetS(torch.nn.Module):
    """Improved EfficientNet-V2-S for binary classification.
    
    Key improvements:
    - Returns logits instead of probabilities (for use with BCEWithLogitsLoss)
    - Optional dropout layer before final classification
    - Better initialization of the final layer
    """
    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT, progress=False)
        
        # Get the original classifier
        original_classifier = self.model.classifier
        last_layer = original_classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        
        # Create new classifier with optional dropout
        new_classifier = []
        
        # Add all layers except the last one
        for layer in original_classifier[:-1]:
            new_classifier.append(layer)
        
        # Add dropout if specified
        if dropout_rate > 0:
            new_classifier.append(nn.Dropout(dropout_rate))
        
        # Add final linear layer for binary classification
        final_layer = torch.nn.Linear(
            last_layer.in_features, 
            1,
            bias=last_layer.bias is not None,
            device=last_layer.weight.device
        )
        
        # Better initialization for binary classification
        # Initialize with smaller weights to prevent saturation
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        if final_layer.bias is not None:
            nn.init.constant_(final_layer.bias, 0.0)
        
        new_classifier.append(final_layer)
        
        # Replace the classifier
        self.model.classifier = nn.Sequential(*new_classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for use with BCEWithLogitsLoss."""
        return self.model(x).squeeze(-1)  # Remove last dimension to get shape (batch_size,)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probabilities for inference."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions."""
        probabilities = self.predict_proba(x)
        return (probabilities > threshold).float()


class EfficientNetL(torch.nn.Module):
    """Improved EfficientNet-V2-L for binary classification.
    
    Key improvements:
    - Returns logits instead of probabilities (for use with BCEWithLogitsLoss)
    - Optional dropout layer before final classification
    - Better initialization of the final layer
    """
    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        self.model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT, progress=False)
        
        # Get the original classifier
        original_classifier = self.model.classifier
        last_layer = original_classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        
        # Create new classifier with optional dropout
        new_classifier = []
        
        # Add all layers except the last one
        for layer in original_classifier[:-1]:
            new_classifier.append(layer)
        
        # Add dropout if specified
        if dropout_rate > 0:
            new_classifier.append(nn.Dropout(dropout_rate))
        
        # Add final linear layer for binary classification
        final_layer = torch.nn.Linear(
            last_layer.in_features, 
            1,
            bias=last_layer.bias is not None,
            device=last_layer.weight.device
        )
        
        # Better initialization for binary classification
        # Initialize with smaller weights to prevent saturation
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        if final_layer.bias is not None:
            nn.init.constant_(final_layer.bias, 0.0)
        
        new_classifier.append(final_layer)
        
        # Replace the classifier
        self.model.classifier = nn.Sequential(*new_classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for use with BCEWithLogitsLoss."""
        return self.model(x).squeeze(-1)  # Remove last dimension to get shape (batch_size,)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probabilities for inference."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions."""
        probabilities = self.predict_proba(x)
        return (probabilities > threshold).float()


class EfficientNetWithClassWeights(EfficientNetS):
    """EfficientNet with support for class-weighted loss during training.
    
    This is useful when you have imbalanced datasets where one class
    is much more frequent than the other.
    """
    def __init__(self, dropout_rate: float = 0.2, pos_weight: float = 1.0):
        super().__init__(dropout_rate=dropout_rate)
        self.register_buffer('pos_weight', torch.tensor(pos_weight))
    
    def get_loss_fn(self):
        """Get the appropriate loss function with class weights."""
        return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)


class EfficientNetWithFocalLoss(EfficientNetS):
    """EfficientNet with focal loss for handling class imbalance."""
    
    def __init__(self, dropout_rate: float = 0.2, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__(dropout_rate=dropout_rate)
        self.alpha = alpha
        self.gamma = gamma
    
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for binary classification."""
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
