import torch


class ConvNotFoundError(RuntimeError):
    """Error raised when first convolutional layer is not found."""


class ClassifyImageWithPosition(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, pos_encode_size: int = 16, allow_adapter: bool = False):
        super().__init__()
        self.base_model = base_model
        num_features = base_model.num_features
        assert isinstance(num_features, int)
        num_features = int(num_features)

        try:
            self._to_6_channels()
            self.adapter = torch.nn.Identity()
        except ConvNotFoundError as e:
            if not allow_adapter:
                raise e
            self.adapter = torch.nn.Conv2d(in_channels=6, out_channels=3,
                                        kernel_size=1, stride=1, padding=0)
            adapter_weight = torch.zeros_like(self.adapter.weight)
            for i in range(3):
                adapter_weight[i, i, 0, 0] = 1.0
            blend_value = 1.0 / 3.0
            adapter_weight[:, 3:, 0, 0] = blend_value
            self.adapter.weight.data = adapter_weight
            assert self.adapter.bias is not None
            torch.nn.init.zeros_(self.adapter.bias)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features + pos_encode_size, num_features + pos_encode_size),
            torch.nn.BatchNorm1d(num_features + pos_encode_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(num_features + pos_encode_size, 1),
        )

        self.pos_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, pos_encode_size),
            torch.nn.BatchNorm1d(pos_encode_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )
        
        # Reset all gradients to ensure clean initial state
        self.zero_grad(set_to_none=True)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        x = self.base_model(x)
        pos = self.pos_encoder(pos)
        x = torch.cat((x, pos), dim=1)
        x = self.classifier(x)
        return x

    def freeze_base_model(self, keep_first_layer_trainable: bool = False):
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        if keep_first_layer_trainable:
            try:
                first_conv = self._get_first_conv()
                for param in first_conv.parameters():
                    param.requires_grad = True
            except ConvNotFoundError: pass

    def unfreeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def _to_6_channels(self):
        # Get first convolutional layer (raises exception if not found)
        conv_first = self._get_first_conv()
        
        # Validate it has 3 input channels (required for fusion with 6->3 adapter)
        if conv_first.in_channels != 3:
            raise RuntimeError(
                f"Cannot fuse adapter: first Conv2d must have 3 input channels (pretrained on RGB), "
                f"but found {conv_first.in_channels} input channels"
            )
        
        # Expand first conv layer from 3 to 6 input channels
        # Original: (N, 3, K, K) for RGB input
        # Expanded: (N, 6, K, K) for [R, G, B, Diff_R, Diff_G, Diff_B] input
        #
        # The adapter conceptually does: output_RGB = input_RGB + avg(input_Diff_RGB)
        # When fused with first conv: we need to replicate this behavior
        #
        # Fused weights structure:
        #   Channels 0-2: Copy original weights (process original RGB)
        #   Channels 3-5: Average of original weights (process averaged diff and add to each RGB)
        first_weight = conv_first.weight  # (N, 3, K, K)
        
        # Create expanded weight tensor: (N, 6, K, K)
        N = first_weight.shape[0]
        K_h, K_w = first_weight.shape[2], first_weight.shape[3]
        fused_weight = torch.zeros(N, 6, K_h, K_w, 
                                   device=first_weight.device, 
                                   dtype=first_weight.dtype)
        
        # Channels 0-2: Copy original weights (identity mapping for original RGB)
        fused_weight[:, 0:3, :, :] = first_weight
        
        # Channels 3-5: Average of the 3 original channel weights, replicated 3 times
        # This computes (R+G+B)/3 from diff channels and adds it to each output channel
        avg_weight = first_weight.sum(dim=1, keepdim=True) / 3.0  # (N, 1, K, K)
        fused_weight[:, 3:6, :, :] = avg_weight.expand(-1, 3, -1, -1)  # Broadcast to 3 channels
        
        # Update first layer in-place: change in_channels from 3 to 6
        conv_first.in_channels = 6
        conv_first.weight = torch.nn.Parameter(fused_weight)
    
    def _get_first_conv(self) -> torch.nn.Conv2d:
        """
        Locate and extract the first convolutional layer from the base model.
        
        This method handles different model architectures by checking for common patterns:
        - ResNet, WideResNet: base_model.model.conv1
        - MobileNet, EfficientNet, ShuffleNet: base_model.model.features[0]
        - YOLO: base_model.convs[0]
        
        Returns:
            The first Conv2d layer that will be modified to accept 6 input channels
            
        Raises:
            ConvNotFoundError: If first conv layer cannot be found or extracted
        """
        first_layer_module: torch.nn.Module | None = None
        
        # Branch 1: Models with .model attribute (ResNet, MobileNet, EfficientNet, ShuffleNet, WideResNet)
        if hasattr(self.base_model, 'model'):
            model = self.base_model.model
            assert isinstance(model, torch.nn.Module), \
                f"base_model.model is not a torch.nn.Module, got {type(model).__name__}"
            
            # Sub-branch 1a: ResNet-style models with .conv1 attribute
            # Used by: ResNet18, ResNet101, ResNet152, WideResNet50, WideResNet101
            # Structure: model.conv1 -> Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            if hasattr(model, 'conv1'):
                assert isinstance(model.conv1, torch.nn.Module), \
                    f"model.conv1 is not a torch.nn.Module, got {type(model.conv1).__name__}"
                first_layer_module = model.conv1
            
            # Sub-branch 1b: Feature-based models with .features Sequential container
            # Used by: MobileNetV3 (Small/Large), EfficientNetV2 (Small/Large), ShuffleNetV2
            # Structure: model.features[0] -> First conv layer in Sequential
            #   MobileNet: Conv2dNormActivation (wraps Conv2d)
            #   EfficientNet: Conv2dNormActivation (wraps Conv2d)
            #   ShuffleNet: Conv2d(3, 24, kernel_size=3, stride=2, padding=1)
            elif hasattr(model, 'features'):
                features = model.features
                assert isinstance(features, torch.nn.Module), \
                    f"model.features is not a torch.nn.Module, got {type(features).__name__}"
                features_children = list(features.children())
                if len(features_children) > 0:
                    first_layer_module = features_children[0]
                    assert isinstance(first_layer_module, torch.nn.Module), \
                        f"model.features[0] is not a torch.nn.Module, got {type(first_layer_module).__name__}"
        
        # Branch 2: YOLO models with .convs attribute
        # Used by: YoloN, YoloS, YoloM, YoloL, YoloX (YOLO11 classification models)
        # Structure: base_model.convs[0] -> Conv wrapper containing Conv2d(3, 32, kernel_size=3, stride=2)
        elif hasattr(self.base_model, 'convs'):
            convs = self.base_model.convs
            assert isinstance(convs, torch.nn.Module), \
                f"base_model.convs is not a torch.nn.Module, got {type(convs).__name__}"
            convs_children = list(convs.children())
            if len(convs_children) > 0:
                first_layer_module = convs_children[0]
                assert isinstance(first_layer_module, torch.nn.Module), \
                    f"base_model.convs[0] is not a torch.nn.Module, got {type(first_layer_module).__name__}"
        
        if first_layer_module is None:
            raise ConvNotFoundError(
                "Could not locate first layer in base model. "
                "Expected base_model to have 'model.conv1', 'model.features[0]', or 'convs[0]'"
            )
        
        # Extract the actual Conv2d from potential wrapper
        conv_layer = self._extract_conv_from_module(first_layer_module)
        if conv_layer is None:
            raise ConvNotFoundError(
                f"Could not extract Conv2d from first layer module of type {type(first_layer_module).__name__}. "
                "Expected Conv2d directly or wrapped in a module with .conv attribute"
            )
        
        # Validate it's a Conv2d
        assert isinstance(conv_layer, torch.nn.Conv2d), \
            f"Extracted layer is not a Conv2d, got {type(conv_layer).__name__}"
        
        return conv_layer
    
    def _extract_conv_from_module(self, module: torch.nn.Module) -> torch.nn.Conv2d | None:
        # Direct Conv2d
        if isinstance(module, torch.nn.Conv2d):
            return module
        
        # Wrapped in a module with .conv attribute (e.g., YOLO's Conv wrapper)
        if hasattr(module, 'conv') and isinstance(module.conv, torch.nn.Conv2d):
            return module.conv
        
        # Check first child
        children = list(module.children())
        if len(children) > 0:
            first_child = children[0]
            if isinstance(first_child, torch.nn.Conv2d):
                return first_child
            elif hasattr(first_child, 'conv') and isinstance(first_child.conv, torch.nn.Conv2d):
                return first_child.conv
        
        return None
    

