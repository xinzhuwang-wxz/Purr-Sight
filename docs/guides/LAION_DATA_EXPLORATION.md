# LAION Subset Data Exploration Report

## Dataset Overview

**Location**: `data_formal_alin/laion-subset/data/`

**Files**: 9 parquet files (train-00000-of-00009.parquet through train-00008-of-00009.parquet)

**Total Samples**: ~24,840 samples (approximately 2,760 per file)

## Data Structure

### Columns

Each parquet file contains **2 columns**:

1. **`image`**: Dictionary containing image data
   - Type: `dict`
   - Keys: `['bytes', 'path']`
   - `bytes`: Raw image bytes (JPEG format)
   - `path`: Original file path (e.g., "0.jpg", "1.jpg")

2. **`caption`**: Text description of the image
   - Type: `str`
   - Mean length: ~70 characters
   - Range: 6-945 characters

### Image Format Details

**Storage Format**:
- Images are stored as **dictionaries** in the parquet file
- Each dictionary contains:
  - `bytes`: Raw JPEG image bytes
  - `path`: Original filename/path

**Image Characteristics** (from sample analysis):
- **Format**: JPEG
- **Color Mode**: RGB
- **Sizes**: Variable (NOT standardized to 512×512)
  - Width range: 400-1600 pixels
  - Height range: 297-1437 pixels
  - Mean width: ~737 pixels
  - Mean height: ~629 pixels
- **File sizes**: 30-500 KB per image

**Note**: Only ~0.1% of images are exactly 512×512. Most images have varying dimensions.

## Usage Example

### Loading Data

```python
import pandas as pd
from PIL import Image
import io

# Read parquet file
df = pd.read_parquet('data_formal_alin/laion-subset/data/train-00000-of-00009.parquet', engine='pyarrow')

# Extract image and caption from first row
row = df.iloc[0]
image_dict = row['image']
caption = row['caption']

# Convert image dictionary to PIL Image
img_bytes = image_dict['bytes']
pil_image = Image.open(io.BytesIO(img_bytes))

print(f"Caption: {caption}")
print(f"Image size: {pil_image.size}")
print(f"Image mode: {pil_image.mode}")
```

### Converting to PIL Image Objects

To convert all images to PIL Image objects (and optionally resize to 512×512):

```python
def extract_pil_image(image_dict, resize_to_512=False):
    """Extract PIL Image from dictionary.
    
    Args:
        image_dict: Dictionary with 'bytes' key
        resize_to_512: If True, resize image to 512×512
        
    Returns:
        PIL Image object
    """
    img_bytes = image_dict['bytes']
    img = Image.open(io.BytesIO(img_bytes))
    
    if resize_to_512:
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
    
    return img
```

## Statistics Summary

### Caption Statistics
- **Mean length**: 70.2 characters
- **Median length**: 55.0 characters
- **Min length**: 6 characters
- **Max length**: 945 characters
- **Non-null**: 100% (all samples have captions)

### Image Statistics
- **Non-null**: 100% (all samples have images)
- **Format**: JPEG
- **Mode**: RGB
- **512×512 images**: ~0.1% of total
- **Average file size**: ~154 KB

## Data Processing Recommendations

1. **Image Resizing**: If you need standardized 512×512 images, you'll need to resize them during preprocessing
2. **Memory Management**: Images are stored as bytes in parquet, which is efficient for storage but requires conversion to PIL Images for processing
3. **Batch Processing**: Consider processing images in batches to manage memory efficiently
4. **Data Augmentation**: The varying image sizes provide opportunities for data augmentation (cropping, resizing, etc.)
