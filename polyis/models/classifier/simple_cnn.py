import torch


class SimpleCNN(torch.nn.Module):
    def __init__(self, img_size: int, width=60, dropout=0.2):
        super().__init__()
        self.encoder = self._build_encoder(img_size, width)

        # Calculate the flattened size after encoder (4x4 x width)
        encoder_output_size = 4 * 4 * width
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_output_size, encoder_output_size),
            torch.nn.BatchNorm1d(encoder_output_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(encoder_output_size, encoder_output_size),
            torch.nn.BatchNorm1d(encoder_output_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(encoder_output_size, 1),
        )

    def _calculate_conv_output_size(self, input_size, kernel_size, stride, padding=0):
        """Calculate output size after Conv2d operation."""
        return (input_size + 2 * padding - kernel_size) // stride + 1
    
    def _build_encoder(self, img_size, width):
        """Build encoder that reduces image size to 4x4."""
        prev_features = 3
        sequential = []
        current_size = img_size
        
        # Calculate the exact path to reach 4x4
        target_size = 4
        
        while current_size > target_size:
            # Calculate what stride we need to get closer to target
            # For stride=2: output = (current - 4) // 2 + 1
            # For stride=1: output = current - 3
            
            # Try stride=2 first
            stride2_output = (current_size - 4) // 2 + 1
            
            # If stride=2 gets us too close to target, use stride=1
            if stride2_output <= target_size:
                stride = 1
                output_size = current_size - 3
                
                # If stride=1 would overshoot, we need to stop here
                if output_size < target_size:
                    break
            else:
                stride = 2
                output_size = stride2_output
            
            sequential.append(torch.nn.Conv2d(
                in_channels=prev_features,
                out_channels=width,
                kernel_size=4,
                stride=stride,
                padding=0
            ))
            sequential.append(torch.nn.BatchNorm2d(width))
            sequential.append(torch.nn.ReLU())
            
            prev_features = width
            current_size = output_size
        
        # If we didn't reach exactly 4x4, add a final layer with appropriate kernel size
        if current_size != target_size:
            # Calculate what kernel size we need to go from current_size to 4
            # output = current_size - (kernel_size - 1)
            # 4 = current_size - (kernel_size - 1)
            # kernel_size = current_size - 4 + 1
            final_kernel = current_size - target_size + 1
            
            sequential.append(torch.nn.Conv2d(
                in_channels=prev_features,
                out_channels=width,
                kernel_size=final_kernel,
                stride=1,
                padding=0
            ))
            sequential.append(torch.nn.BatchNorm2d(width))
            sequential.append(torch.nn.ReLU())
            current_size = target_size
        return torch.nn.Sequential(*sequential)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SimpleCNN classifier.
        Args:
            imgs: Input image tensor of shape (batch_size, 3, height, width)
        Returns:
            Logits for binary classification (car vs no car). Shape (batch_size, 1)
        """
        x = self.encoder(imgs)
        x = x.flatten(1)
        x = self.decoder(x)
        return torch.sigmoid(x)
