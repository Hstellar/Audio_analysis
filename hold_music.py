import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.fft import fft

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


def extract_features(audio, sr, n_mfcc=13):
    """Extract MFCC features for robust comparison."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    return np.vstack([mfcc, mfcc_delta])


def compute_spectral_similarity(segment, reference, sr):
    """
    Compute similarity using multiple robust methods.
    Returns a combined similarity score.
    """
    # Method 1: MFCC-based similarity
    mfcc_seg = extract_features(segment, sr)
    mfcc_ref = extract_features(reference, sr)
    
    # Use dynamic time warping distance
    from scipy.spatial.distance import euclidean
    
    # Normalize features
    mfcc_seg = (mfcc_seg - np.mean(mfcc_seg, axis=1, keepdims=True)) / (np.std(mfcc_seg, axis=1, keepdims=True) + 1e-8)
    mfcc_ref = (mfcc_ref - np.mean(mfcc_ref, axis=1, keepdims=True)) / (np.std(mfcc_ref, axis=1, keepdims=True) + 1e-8)
    
    # Compute correlation between MFCC features
    min_frames = min(mfcc_seg.shape[1], mfcc_ref.shape[1])
    if min_frames > 0:
        mfcc_seg_truncated = mfcc_seg[:, :min_frames]
        mfcc_ref_truncated = mfcc_ref[:, :min_frames]
        
        mfcc_similarity = np.mean([
            np.corrcoef(mfcc_seg_truncated[i], mfcc_ref_truncated[i])[0, 1] 
            for i in range(mfcc_seg_truncated.shape[0])
        ])
    else:
        mfcc_similarity = 0
    
    # Method 2: Spectral correlation
    spec_seg = np.abs(librosa.stft(segment))
    spec_ref = np.abs(librosa.stft(reference))
    
    min_frames_spec = min(spec_seg.shape[1], spec_ref.shape[1])
    if min_frames_spec > 0:
        spec_seg_truncated = spec_seg[:, :min_frames_spec]
        spec_ref_truncated = spec_ref[:, :min_frames_spec]
        
        # Flatten and compute correlation
        spec_seg_flat = spec_seg_truncated.flatten()
        spec_ref_flat = spec_ref_truncated.flatten()
        
        if len(spec_seg_flat) > 0 and len(spec_ref_flat) > 0:
            spec_correlation = np.corrcoef(spec_seg_flat, spec_ref_flat)[0, 1]
        else:
            spec_correlation = 0
    else:
        spec_correlation = 0
    
    # Method 3: Energy-normalized waveform correlation
    segment_norm = segment / (np.sqrt(np.sum(segment**2)) + 1e-8)
    reference_norm = reference / (np.sqrt(np.sum(reference**2)) + 1e-8)
    
    min_len = min(len(segment_norm), len(reference_norm))
    if min_len > 0:
        wave_correlation = np.corrcoef(segment_norm[:min_len], reference_norm[:min_len])[0, 1]
    else:
        wave_correlation = 0
    
    # Combine scores (weighted average)
    combined_score = (
        0.5 * mfcc_similarity +
        0.3 * spec_correlation +
        0.2 * wave_correlation
    )
    
    return combined_score, {
        'mfcc': mfcc_similarity,
        'spectral': spec_correlation,
        'waveform': wave_correlation
    }


def detect_hold_music(call_recording_path, hold_music_path, channel=1, 
                      threshold=0.35, hop_length=1.0, sample_rate=16000,
                      window_duration=None):
    """
    Detect hold music in a call recording using robust spectral analysis.
    
    Args:
        call_recording_path: Path to call recording file (.opus)
        hold_music_path: Path to hold music reference file (.ogg)
        channel: Channel number to analyze (0-indexed, so 1 = second channel)
        threshold: Similarity threshold for detection (0-1, typical: 0.3-0.5)
        hop_length: Time in seconds to move window for detection
        sample_rate: Sample rate for processing
        window_duration: Duration of analysis window (None = use hold music duration)
        
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
    
    # Use shorter window if hold music is very long (for efficiency)
    if window_duration is None:
        window_duration = min(hold_music_duration, 10.0)  # Max 10 seconds
    
    window_samples = int(window_duration * sample_rate)
    hop_samples = int(hop_length * sample_rate)
    
    # Also trim hold music reference to match window
    hold_music_reference = hold_music[:window_samples]
    
    print(f"Using {window_duration:.2f}s analysis window")
    print(f"Detecting hold music (threshold={threshold})...")
    
    # Detect segments
    detections = []
    max_similarity = 0
    
    num_windows = (len(audio) - window_samples) // hop_samples
    for i, start_idx in enumerate(range(0, len(audio) - window_samples, hop_samples)):
        if i % 50 == 0:
            print(f"Progress: {i}/{num_windows} windows processed (max similarity so far: {max_similarity:.3f})", end='\r')
            
        end_idx = start_idx + window_samples
        segment = audio[start_idx:end_idx]
        
        # Compute similarity using multiple methods
        try:
            similarity, breakdown = compute_spectral_similarity(segment, hold_music_reference, sr)
            max_similarity = max(max_similarity, similarity)
            
            # Check if hold music is detected
            if similarity > threshold:
                start_time = start_idx / sample_rate
                end_time = end_idx / sample_rate
                detections.append({
                    'start': start_time,
                    'end': end_time,
                    'similarity': float(similarity),
                    'breakdown': {k: float(v) for k, v in breakdown.items()}
                })
        except Exception as e:
            # Skip problematic segments
            continue
    
    print(f"\nFound {len(detections)} initial detection windows")
    
    # Merge overlapping/nearby detections
    merged_detections = merge_detections(detections, gap_tolerance=2.0)
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
        print(f"  Maximum similarity: {max_similarity:.3f}")
        print(f"\n  MAXIMUM CONTINUOUS DURATION: {max_duration:.2f} seconds")
        
        if longest_segment:
            print(f"  Longest segment time: {longest_segment['start']:.2f}s - {longest_segment['end']:.2f}s")
            print(f"  Similarity: {longest_segment['similarity']:.3f}")
            if 'breakdown' in longest_segment:
                print(f"  Score breakdown - MFCC: {longest_segment['breakdown']['mfcc']:.3f}, "
                      f"Spectral: {longest_segment['breakdown']['spectral']:.3f}, "
                      f"Waveform: {longest_segment['breakdown']['waveform']:.3f}")
        
        print(f"\n  All detected segments:")
        for i, seg in enumerate(merged_detections, 1):
            duration = seg['end'] - seg['start']
            print(f"    {i}. {seg['start']:.2f}s - {seg['end']:.2f}s "
                  f"(duration: {duration:.2f}s, similarity: {seg['similarity']:.3f})")
    else:
        print(f"\n✗ No hold music detected")
        print(f"  Maximum similarity found: {max_similarity:.3f}")
        print(f"  (Threshold was {threshold})")
        print(f"\n  Tips to improve detection:")
        print(f"    - Try lowering threshold to {max(0.2, threshold - 0.1):.2f}")
        print(f"    - Check if hold music is in the correct channel")
        print(f"    - Verify hold music reference file is correct")
    
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
        'max_similarity': float(max_similarity),
        'threshold_used': threshold
    }


def merge_detections(detections, gap_tolerance=2.0):
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
            # Merge - extend end time and keep higher similarity
            last_merged['end'] = max(last_merged['end'], det['end'])
            if det['similarity'] > last_merged['similarity']:
                last_merged['similarity'] = det['similarity']
                last_merged['breakdown'] = det.get('breakdown', {})
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
        threshold=0.35,  # Start with 0.35, adjust based on results
        hop_length=1.0,  # Check every 1 second
        sample_rate=16000  # Processing sample rate
    )
    
    # Optionally save results to JSON
    import json
    with open("detection_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nResults saved to detection_result.json")
