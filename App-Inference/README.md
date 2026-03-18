# App-Inference

Interactive inference UI for Landscape-Classifier built with Gradio.

## Features

- 🎯 **Smart Model Selection** - Browse and select from trained models in `App-Inference/models/`
- 📸 **Flexible Input** - Upload image files or paste directly from clipboard
- 📊 **Detailed Predictions** - View top-k class predictions with confidence scores
- 📱 **Mobile-Friendly** - Fully responsive design for desktop, tablet, and mobile devices
- 🔄 **Model Discovery** - Auto-discover models and display metadata (size, classes)
- ⚡ **Optimized Performance** - Efficient caching and batch inference
- 🌄 **Landscape Classification** - 5-class classifier: normal, fire, flooded_areas, collapsed_building, traffic_incident

## Run

From the repository root using `.venv-ml`:

```bash
source .venv-ml/bin/activate
pip install -r requirements.txt
python App-Inference/app.py
```

Then open the local URL printed by Gradio (usually `http://127.0.0.1:7860`).

Or run in one step:

```bash
./App-Inference/run.sh
```

## Features Deep Dive

### Model Browser
- See all available models with file sizes
- Quick model switching with auto-refresh
- Model metadata display (architecture, classes)

### Image Input
- **Upload**: Drag & drop or click to select files
- **Clipboard**: Paste images directly from your clipboard
- Automatic format conversion and resizing

### Inference Controls
- **Top-K Slider**: Adjust number of predictions (1-10)
- **Run Button**: Execute inference with real-time feedback
- **Clear Button**: Reset inputs for next image
- **Refresh Button**: Update model list

### Mobile Optimization
- Responsive grid layout (1 column on mobile, 2+ on desktop)
- Touch-friendly button sizing
- Optimized spacing and text readability
- Fast loading with cached transforms

## Configuration

Key constants are at the top of `app.py`:

```python
MODELS_DIR = "App-Inference/models"  # Where trained models are stored
DEFAULT_TOP_K = 3                     # Default number of predictions
MAX_TOP_K = 10                        # Maximum allowed predictions
IMAGE_SIZE = 224                      # ResNet50 input size
```

## Model Format

Models must be PyTorch `.pth` files created by the training pipeline with:
- `model_state_dict` - trained weights
- `class_names` - list of class labels
- `num_classes` - number of classes
- `model_arch` - architecture name (e.g., "resnet50")

## Troubleshooting

### No models found
- Ensure you have `.pth` files in `App-Inference/models/`
- Check the startup output shows the model directory
- Use the "Refresh" button to update the model list

### Slow inference
- First inference is slower (model loading + compilation)
- Subsequent inferences use cached model
- GPU inference is faster if CUDA is available

### Image errors
- Ensure image is a valid format (PNG, JPEG, etc.)
- Try clipboard paste instead of upload if file upload fails
