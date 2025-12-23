import numpy as np
import librosa
import soundfile as sf
from scipy import signal

def extract_channel(audio_path, channel=1, sample_rate=16000):
    """
    Extract specific channel from audio file.
    
    Args:
        audio_path: Path to audio file
        channel: Channel number (0-indexed, so channel=1 means second channel)
        sample_rate: Target sample rate for processing
        
    Returns:
        Audio data from specified channel and sample rate
    """
    # Load audio with soundfile to preserve channels
    audio, sr = sf.read(audio_path)
    
    # If mono, return as is
    if len(audio.shape) == 1:
        audio_mono = audio
    else:
        # Extract the specified channel
        if channel < audio.shape[1]:
            audio_mono = audio[:, channel]
        else:
            raise ValueError(f"Channel {channel} not found. Audio has {audio.shape[1]} channels")
    
    # Resample if needed
    if sr != sample_rate:
        audio_mono = librosa.resample(audio_mono, orig_sr=sr, target_sr=sample_rate)
        
    return audio_mono, sample_rate


def detect_hold_music(call_recording_path, hold_music_path, channel=1, 
                      threshold=0.3, hop_length=0.5, sample_rate=16000):
    """
    Detect hold music in a call recording.
    
    Args:
        call_recording_path: Path to call recording file (.opus)
        hold_music_path: Path to hold music reference file (.ogg)
        channel: Channel number to analyze (0-indexed, so 1 = second channel)
        threshold: Correlation threshold for detection (0-1, typical: 0.2-0.5)
        hop_length: Time in seconds to move window for detection
        sample_rate: Sample rate for processing
        
    Returns:
        Dictionary with detection results
    """
    print(f"Loading hold music reference: {hold_music_path}")
    hold_music, _ = librosa.load(hold_music_path, sr=sample_rate, mono=True)
    hold_music_duration = len(hold_music) / sample_rate
    print(f"Hold music duration: {hold_music_duration:.2f} seconds\n")
    
    print(f"Processing call recording: {call_recording_path}")
    print(f"Extracting channel {channel}...")
    
    # Extract specified channel from call recording
    audio, sr = extract_channel(call_recording_path, channel, sample_rate)
    audio_duration = len(audio) / sample_rate
    print(f"Call recording duration: {audio_duration:.2f} seconds\n")
    
    # Window parameters
    window_samples = len(hold_music)
    hop_samples = int(hop_length * sample_rate)
    
    print(f"Detecting hold music (threshold={threshold})...")
    
    # Detect segments
    detections = []
    max_correlation = 0
    
    num_windows = (len(audio) - window_samples) // hop_samples
    for i, start_idx in enumerate(range(0, len(audio) - window_samples, hop_samples)):
        if i % 100 == 0:
            print(f"Progress: {i}/{num_windows} windows processed", end='\r')
            
        end_idx = start_idx + window_samples
        segment = audio[start_idx:end_idx]
        
        # Compute normalized cross-correlation
        correlation = signal.correlate(segment, hold_music, mode='valid')
        norm_audio = np.sqrt(np.sum(segment**2))
        norm_ref = np.sqrt(np.sum(hold_music**2))
        
        if norm_audio > 0 and norm_ref > 0:
            correlation = correlation / (norm_audio * norm_ref)
        
        max_corr = np.max(correlation) if len(correlation) > 0 else 0
        max_correlation = max(max_correlation, max_corr)
        
        # Check if hold music is detected
        if max_corr > threshold:
            start_time = start_idx / sample_rate
            end_time = end_idx / sample_rate
            detections.append({
                'start': start_time,
                'end': end_time,
                'correlation': float(max_corr)
            })
    
    print(f"\nFound {len(detections)} initial detection windows")
    
    # Merge overlapping/nearby detections
    merged_detections = merge_detections(detections, gap_tolerance=1.0)
    print(f"Merged into {len(merged_detections)} continuous segments\n")
    
    # Find maximum duration
    max_duration = 0
    longest_segment = None
    
    for det in merged_detections:
        duration = det['end'] - det['start']
        if duration > max_duration:
            max_duration = duration
            longest_segment = det
    
    # Print results
    print("="*70)
    print("DETECTION RESULTS")
    print("="*70)
    
    if merged_detections:
        print(f"\n✓ Hold music DETECTED")
        print(f"  Number of segments: {len(merged_detections)}")
        print(f"  Maximum correlation: {max_correlation:.3f}")
        print(f"\n  MAXIMUM CONTINUOUS DURATION: {max_duration:.2f} seconds")
        
        if longest_segment:
            print(f"  Longest segment time: {longest_segment['start']:.2f}s - {longest_segment['end']:.2f}s")
            print(f"  Correlation: {longest_segment['correlation']:.3f}")
        
        print(f"\n  All detected segments:")
        for i, seg in enumerate(merged_detections, 1):
            duration = seg['end'] - seg['start']
            print(f"    {i}. {seg['start']:.2f}s - {seg['end']:.2f}s (duration: {duration:.2f}s, corr: {seg['correlation']:.3f})")
    else:
        print(f"\n✗ No hold music detected")
        print(f"  Maximum correlation found: {max_correlation:.3f}")
        print(f"  (Threshold was {threshold})")
    
    print("\n" + "="*70)
    
    return {
        'call_recording': call_recording_path,
        'hold_music': hold_music_path,
        'channel': channel,
        'audio_duration': audio_duration,
        'hold_music_detected': len(merged_detections) > 0,
        'num_segments': len(merged_detections),
        'segments': merged_detections,
        'max_duration': max_duration,
        'longest_segment': longest_segment,
        'max_correlation': float(max_correlation)
    }


def merge_detections(detections, gap_tolerance=1.0):
    """
    Merge overlapping or nearby detections.
    
    Args:
        detections: List of detection dictionaries
        gap_tolerance: Maximum gap in seconds to merge segments
    """
    if not detections:
        return []
    
    # Sort by start time
    sorted_dets = sorted(detections, key=lambda x: x['start'])
    
    merged = [sorted_dets[0].copy()]
    
    for det in sorted_dets[1:]:
        last_merged = merged[-1]
        
        # Check if current detection overlaps or is close to last merged
        if det['start'] <= last_merged['end'] + gap_tolerance:
            # Merge
            last_merged['end'] = max(last_merged['end'], det['end'])
            last_merged['correlation'] = max(last_merged['correlation'], det['correlation'])
        else:
            # Add as new segment
            merged.append(det.copy())
    
    return merged


# Example usage
if __name__ == "__main__":
    # Specify your file paths
    call_recording = "path/to/your/call_recording.opus"
    hold_music = "path/to/your/hold_music.ogg"
    
    # Detect hold music in channel 1 (second channel, 0-indexed)
    result = detect_hold_music(
        call_recording_path=call_recording,
        hold_music_path=hold_music,
        channel=1,  # Channel 1 = second channel (0-indexed)
        threshold=0.3,  # Adjust if needed: lower = more sensitive, higher = more strict
        hop_length=0.5,  # Check every 0.5 seconds
        sample_rate=16000  # Processing sample rate
    )
    
    # Optionally save results to JSON
    import json
    with open("detection_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nResults saved to detection_result.json")
