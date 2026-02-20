"""Utility functions for working with LAION dataset parquet files."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from PIL import Image
import io


def load_laion_parquet(file_path: Path) -> pd.DataFrame:
    """Load a LAION parquet file.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        DataFrame with 'image' and 'caption' columns
    """
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
    except Exception:
        df = pd.read_parquet(file_path, engine='fastparquet')
    
    return df


def extract_pil_image(image_dict: Dict[str, Any], resize_to: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Extract PIL Image from LAION image dictionary.
    
    Args:
        image_dict: Dictionary containing 'bytes' key with image data
        resize_to: Optional tuple (width, height) to resize image. If None, returns original size.
        
    Returns:
        PIL Image object
        
    Raises:
        ValueError: If image_dict doesn't contain 'bytes' key
        IOError: If image bytes cannot be decoded
    """
    if 'bytes' not in image_dict:
        raise ValueError("image_dict must contain 'bytes' key")
    
    img_bytes = image_dict['bytes']
    if not isinstance(img_bytes, bytes):
        raise ValueError(f"Expected bytes, got {type(img_bytes)}")
    
    img = Image.open(io.BytesIO(img_bytes))
    
    if resize_to is not None:
        img = img.resize(resize_to, Image.Resampling.LANCZOS)
    
    return img


def extract_pil_image_512x512(image_dict: Dict[str, Any]) -> Image.Image:
    """Extract PIL Image and resize to 512×512.
    
    Convenience function for extracting and resizing images to 512×512.
    
    Args:
        image_dict: Dictionary containing 'bytes' key with image data
        
    Returns:
        PIL Image object resized to 512×512
    """
    return extract_pil_image(image_dict, resize_to=(512, 512))


def get_sample_from_parquet(
    file_path: Path,
    index: int = 0,
    resize_to: Optional[Tuple[int, int]] = None
) -> Tuple[Image.Image, str]:
    """Get a single sample (image, caption) from a parquet file.
    
    Args:
        file_path: Path to the parquet file
        index: Index of the sample to retrieve (default: 0)
        resize_to: Optional tuple (width, height) to resize image
        
    Returns:
        Tuple of (PIL Image, caption string)
        
    Raises:
        IndexError: If index is out of range
    """
    df = load_laion_parquet(file_path)
    
    if index >= len(df):
        raise IndexError(f"Index {index} out of range for dataset with {len(df)} samples")
    
    row = df.iloc[index]
    image_dict = row['image']
    caption = row['caption']
    
    pil_image = extract_pil_image(image_dict, resize_to=resize_to)
    
    return pil_image, caption


def load_all_laion_files(data_dir: Path) -> pd.DataFrame:
    """Load all parquet files from LAION data directory.
    
    Args:
        data_dir: Directory containing parquet files
        
    Returns:
        Combined DataFrame with all samples
    """
    parquet_files = sorted(list(data_dir.glob("*.parquet")))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    dfs = []
    for parquet_file in parquet_files:
        df = load_laion_parquet(parquet_file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def convert_to_pil_images(
    df: pd.DataFrame,
    resize_to: Optional[Tuple[int, int]] = None,
    max_samples: Optional[int] = None
) -> List[Image.Image]:
    """Convert image dictionaries in DataFrame to PIL Image objects.
    
    Args:
        df: DataFrame with 'image' column containing dictionaries
        resize_to: Optional tuple (width, height) to resize images
        max_samples: Optional limit on number of images to convert
        
    Returns:
        List of PIL Image objects
    """
    images = []
    limit = min(max_samples, len(df)) if max_samples else len(df)
    
    for idx in range(limit):
        image_dict = df.iloc[idx]['image']
        pil_image = extract_pil_image(image_dict, resize_to=resize_to)
        images.append(pil_image)
    
    return images
