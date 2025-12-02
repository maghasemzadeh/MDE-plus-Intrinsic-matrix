# Depth Anything V2 Models

This directory contains the model interface system for Depth Anything V2. It provides a unified interface for loading and using different depth estimation models.

## Structure

- `base.py` - Base abstract class (`BaseDepthModel`) that all models must inherit from
- `registry.py` - Model registry system for managing and loading different models
- `depth_anything_v2_basic.py` - Basic Depth Anything V2 model (with scale ambiguity)
- `depth_anything_v2_metric.py` - Metric Depth Anything V2 model (metric depth in meters)

## Usage

### Loading Models

```python
from depth_anything_v2.model_loader import load_model

# Load basic model
model = load_model('basic', encoder='vitl')

# Load metric model
model = load_model('metric', encoder='vitl', max_depth=20.0)

# Load with custom checkpoint
model = load_model('basic', checkpoint_path='path/to/checkpoint.pth')
```

### Using Models

```python
import cv2

# Load model
model = load_model('basic', encoder='vitl')

# Run inference
image = cv2.imread('image.jpg')
depth = model.infer_image(image, input_size=518)

# Check model info
info = model.get_model_info()
print(f"Model: {info['model_name']}")
print(f"Is metric: {model.is_metric()}")
```

### Adding New Models

To add a new model, create a new file and inherit from `BaseDepthModel`:

```python
from depth_anything_v2.models.base import BaseDepthModel
from depth_anything_v2.models import register_model

class MyCustomModel(BaseDepthModel):
    def __init__(self, **kwargs):
        super().__init__(model_name='my_custom_model', **kwargs)
        # Initialize your model
    
    def _build_model(self, **kwargs):
        # Build your model architecture
        pass
    
    def _load_checkpoint(self, checkpoint_path, **kwargs):
        # Load checkpoint weights
        pass
    
    def infer_image(self, image, input_size=518, **kwargs):
        # Run inference
        pass

# Register the model
register_model('my_custom_model', MyCustomModel)
```

Then use it:

```python
model = load_model('my_custom_model', encoder='vitl')
```

## Model Registry

The model registry allows you to:

- List all available models: `list_models()`
- Check if a model exists: `has_model('model_name')`
- Get a model class: `get_model('model_name')`
- Register a new model: `register_model('name', ModelClass)`

## Benefits

1. **Unified Interface**: All models follow the same interface, making it easy to switch between them
2. **Easy Extension**: Adding new models is straightforward - just inherit from `BaseDepthModel` and register
3. **Type Safety**: Models are type-checked to ensure they implement the required interface
4. **Backward Compatible**: Existing code continues to work with the new system

