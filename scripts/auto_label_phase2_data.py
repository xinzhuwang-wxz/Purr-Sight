"""
Auto-label Phase 2 data using Volcengine (Doubao) Multimodal LLM.

Reads an input JSONL file containing 'image' paths, sends them to the LLM 
with the V3 System Prompt, and saves the labeled data to a new file.

Usage:
    export ARK_API_KEY="your_api_key"
    python scripts/auto_label_phase2_data.py --input_file data/raw/phase2.jsonl --output_dir data/labeled
"""

import os
import json
import argparse
import base64
import time
import subprocess
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

from purrsight.LLM.prompts import SYSTEM_PROMPT_V3, JSON_SCHEMA_V3

# Configuration
MODEL_NAME = "doubao-seed-1-8-251228" # Or your specific endpoint ID
API_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

def encode_image(image_path_or_bytes):
    """Encodes a local image path or bytes to base64."""
    if isinstance(image_path_or_bytes, (str, Path)):
        with open(image_path_or_bytes, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image_path_or_bytes, bytes):
        return base64.b64encode(image_path_or_bytes).decode('utf-8')
    elif isinstance(image_path_or_bytes, Image.Image):
        buffered = BytesIO()
        image_path_or_bytes.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        raise ValueError("Unsupported image type")

def extract_frames_from_video(video_path: str, num_frames: int = 4) -> list[Image.Image]:
    """Extracts key frames from a video using ffmpeg.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract.

    Returns:
        A list of PIL Images extracted from the video.
    """
    try:
        # Check if video exists and is valid
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return []

        # Use ffprobe to get duration and stream info
        try:
            probe_cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            duration_out = subprocess.check_output(probe_cmd, stderr=subprocess.STDOUT).decode('utf-8').strip()
            if not duration_out or duration_out == 'N/A':
                 # Fallback for streams without duration metadata
                 duration = 5.0 
            else:
                duration = float(duration_out)
        except subprocess.CalledProcessError:
            print(f"Warning: Could not probe video {video_path}. Skipping.")
            return []
        
        # Calculate timestamps for evenly spaced frames
        # Ensure we don't go beyond duration
        if duration < 0.1:
             timestamps = [0.0]
        else:
             timestamps = [duration * (i + 0.5) / num_frames for i in range(num_frames)]
        
        frames = []
        for ts in timestamps:
            # Extract frame at timestamp
            cmd = [
                'ffmpeg', '-ss', str(ts), '-i', video_path, 
                '-vframes', '1', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-'
            ]
            
            # Run ffmpeg and capture output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0 and len(stdout) > 0:
                try:
                    img = Image.open(BytesIO(stdout))
                    img.load() # Verify image integrity
                    frames.append(img)
                except Exception:
                    pass
            # Don't spam warnings for every frame failure, just continue
                
        return frames
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return []

def extract_audio_from_video(video_path, output_dir):
    """
    Extracts audio track from video if present.
    Returns path to extracted audio file or None.
    """
    try:
        # Check for audio stream first
        probe_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a', 
            '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path
        ]
        output = subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL).decode('utf-8').strip()
        if not output:
            return None # No audio stream

        # Construct output filename
        video_name = Path(video_path).stem
        audio_filename = f"{video_name}_audio.wav"
        audio_path = os.path.join(output_dir, audio_filename)
        
        if os.path.exists(audio_path):
            return audio_path

        # Extract audio
        cmd = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', audio_path, '-y'
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return audio_path
    except Exception as e:
        # Silent fail for audio is acceptable
        return None

def label_sample(
    client: OpenAI,
    sample: dict,
    system_prompt: str,
    output_dir: str
) -> dict | None:
    """Sends a single sample to the LLM for labeling.

    Args:
        client: OpenAI client instance.
        sample: Dict containing 'image' or 'video' path.
        system_prompt: The System Prompt V3.
        output_dir: Directory to save extracted assets (audio).

    Returns:
        Dict: Labeled sample with 'instruction' and 'response', or None if failed.
    """
    image_path = sample.get("image")
    video_path = sample.get("video")
    audio_path = sample.get("audio") # Input explicit audio
    
    user_content = []
    final_audio_path = audio_path
    
    # Handle Video
    if video_path and os.path.exists(video_path):
        # 1. Extract Frames
        frames = extract_frames_from_video(video_path, num_frames=4)
        if not frames:
            print(f"Warning: Could not extract frames from {video_path}")
            return None
            
        # 2. Extract Audio (if not already provided)
        if not final_audio_path:
            extracted_audio = extract_audio_from_video(video_path, output_dir)
            if extracted_audio:
                final_audio_path = extracted_audio
        
        prompt_text = "Analyze the cat's behavior in these video frames according to the Ethogram. Consider the sequence of actions."
        if final_audio_path:
            prompt_text += " Note: The video contains an audio track (e.g., meowing, purring, hissing) which you should consider implied by the visual context or if you have audio processing capabilities."
        prompt_text += " Output valid JSON only."

        user_content.append({
            "type": "text",
            "text": prompt_text
        })
        
        for i, frame in enumerate(frames):
            base64_frame = encode_image(frame)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}
            })
            
    # Handle Image
    elif image_path and os.path.exists(image_path):
        base64_image = encode_image(image_path)
        
        prompt_text = "Analyze the cat's behavior in this image according to the Ethogram."
        if final_audio_path:
             prompt_text += " Note: There is an accompanying audio file."
        prompt_text += " Output valid JSON only."

        user_content.append({
            "type": "text",
            "text": prompt_text
        })
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    else:
        # Audio only or missing files
        if final_audio_path:
             print(f"Skipping audio-only sample: {final_audio_path}. Multimodal model requires visual input.")
        else:
             print(f"Warning: No valid visual input found for sample.")
        return None
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
        )
        
        llm_output = response.choices[0].message.content
        
        # Try to parse JSON to validate (simple check)
        # We strip potential markdown code blocks ```json ... ```
        clean_output = llm_output.strip()
        if clean_output.startswith("```json"):
            clean_output = clean_output[7:]
        if clean_output.endswith("```"):
            clean_output = clean_output[:-3]
        
        try:
            parsed_json = json.loads(clean_output)
            # Re-serialize to ensure compact format
            final_response = json.dumps(parsed_json, ensure_ascii=False)
        except json.JSONDecodeError:
            print(f"Warning: Model output invalid JSON. Saving raw output.")
            final_response = clean_output

        # Construct final labeled sample
        labeled_sample = {
            "image": image_path, # Keep original path
            "video": video_path, # Keep original path
            "audio": final_audio_path, # Use extracted or provided audio
            "instruction": "Analyze the cat's behavior based on the visual and auditory inputs. Provide a structured JSON report.",
            "response": final_response,
            "original_llm_output": llm_output # Keep for debugging
        }
        
        # Clean up keys with None values
        return {k: v for k, v in labeled_sample.items() if v is not None}

    except Exception as e:
        print(f"Error calling API: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Auto-label Purr-Sight data using Doubao LLM")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL with 'image' paths")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for labeled data")
    parser.add_argument("--api_key", type=str, default=None, help="Volcengine API Key (or set ARK_API_KEY env)")
    
    args = parser.parse_args()
    
    # Setup API Key
    api_key = args.api_key or os.getenv('ARK_API_KEY')
    if not api_key:
        print("Error: ARK_API_KEY not found. Please set it in .env or pass --api_key")
        return

    # Initialize Client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=api_key,
    )
    
    # Setup Paths
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a subdir for extracted audio
    audio_output_dir = output_dir / "extracted_audio"
    audio_output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"labeled_{input_path.name}"
    
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_file}")
    print(f"Model: {MODEL_NAME}")
    
    # Read Input Data
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Found {len(samples)} samples.")
    
    # Process
    success_count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for sample in tqdm(samples, desc="Labeling"):
            labeled = label_sample(client, sample, SYSTEM_PROMPT_V3, str(audio_output_dir))
            
            if labeled:
                f_out.write(json.dumps(labeled, ensure_ascii=False) + '\n')
                f_out.flush() # Write immediately
                success_count += 1
            
            # Simple rate limit avoidance
            time.sleep(0.5)
            
    print(f"Done! labeled {success_count}/{len(samples)} samples.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
