import aac_datasets.datasets.base
import ffmpeg
import numpy as np
import os
import librosa
import torch

from torch import Tensor


def custom_loading(dataset: aac_datasets.datasets.base.AACDataset, normalize_audios=False) ->  aac_datasets.datasets.base.AACDataset:
    """
    Uses custom data loading to a dataset:
    - Loads a 30s snippet from a longer audio file efficiently.
    - Pads shorter audios to 30s automatically.
    - Resamples audios to 32kHz.

    Args:
        dataset (aac_datasets.datasets.base.AACDataset): The dataset to which transformations will be applied.
        normalize_audios (bool): Whether to normalize the audios to the range [-1, 1]. Default is False.

    Returns:
         aac_datasets.datasets.base.AACDataset: The transformed dataset with added online columns for audio and metadata.
    """
    dataset.add_online_column("audio", _custom_load_audio_mp3, True)
    dataset.add_online_column("audio_metadata", _custom_load_metadata, True)
    dataset.transform = custom_transform
    return dataset


def custom_transform(sample: dict):
    """
    Custom audio padding logic.
    """
    sample['duration'] = sample['audio'].shape[-1] / sample['sr']
    sample['audio'] = _pad_or_subsample_audio(sample['audio'], 32000 * 30)
    return sample

def _custom_load_audio_mp3(args) -> Tensor:
    """
    Custom audio loading logic.
    """
    fname, fpath = args
    base, extension = os.path.splitext(fpath)
    extension = extension[1:]
    # replace WavCaps default folder with WavCaps_mp3
    base =  base.replace("WavCaps", "WavCaps_mp3")
    if extension == "flac": # load mp3 version of WavCaps
        fpath = ".".join([base, 'mp3'])
        extension = 'mp3'
    # load segment; truncate to 30 if longer than 30s

    if extension == "mp3":
        # load audiocaps and wavcaps with ffmpeg
        # audio, sr = _load_random_segment_ffmpeg(fpath, sample_rate=32000)  # type: ignore
        fname, audio = load_minimp3(fname, fpath)
    elif extension == "wav":
        # load clotho files with librosa, no subsampling required because all clotho files <= 30s
        audio, sr = librosa.load(fpath, sr=32000, mono=True)
        audio = audio[None,:]

    # Sanity check
    if audio.size == 0:
        raise RuntimeError(
            f"Invalid audio number of elements in {fpath}. (expected audio.nelement()={audio.nelement()} > 0)"
        )
    return fname, audio



def _normalize_waveform_tensor(waveform):
    """
    Normalize a waveform (PyTorch tensor) to the range [-1, 1] safely.

    Parameters:
    - waveform (torch.Tensor): Input waveform (integer or float tensor)

    Returns:
    - torch.Tensor: Normalized waveform in the range [-1, 1]
    """
    if not isinstance(waveform, torch.Tensor):
        raise ValueError("Input waveform must be a PyTorch tensor.")

    max_val = torch.max(torch.abs(waveform))

    if max_val == 0:
        return waveform  # Avoid division by zero, return unchanged (silent signal)

    return waveform / max_val


def _custom_load_metadata(self, index: int) -> dict:
    """
    Custom metadata loading logic.
    """
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    # just a placeholder...
    return dotdict(dict(
        duration = -1,
        sample_rate = int(32000),
        channels = int(1),
        num_frames = int(-1)
    ))


def _load_random_segment_ffmpeg(
        file_path: str,
        segment_duration: int = 30,
        sample_rate: int = 32000
) -> tuple[np.array, int]:
    """
    Efficiently extracts a random 30-second segment from an audio file using ffmpeg without loading the full file.

    :param file_path: Path to the audio file
    :param segment_duration: Segment duration in seconds (default: 30s)
    :param sample_rate: Sample rate for extracted audio (default: 32kHz)
    :return: PyTorch tensor of extracted audio and sample rate
    """
    try:
        # Get the total duration of the audio file using ffprobe
        probe = ffmpeg.probe(file_path)
        if 'format' not in probe or 'duration' not in probe['format']:
            raise ValueError(f"File appears to be corrupted or unreadable: {file_path}")

        duration = float(probe['format']['duration'])

        if duration < segment_duration:
            segment_duration = duration  # Adjust to full length
            start_time = 0  # Start from the beginning
        else:
            # avoid very small numbers to not run into ffmpeg issues
            start_time = max(0, round(torch.rand(1).item() * (duration - segment_duration), 3)) #

        # Use ffmpeg to extract only the required segment
        out, err = (
            ffmpeg.input(file_path, ss=start_time, t=segment_duration)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=sample_rate)
            .run(capture_stdout=True, capture_stderr=True)
        )

    except ffmpeg.Error as e:
        print(f"FFmpeg error when processing file: {file_path}")
        print("Standard Output:", e.stdout.decode() if e.stdout else "None")
        print("Standard Error:", e.stderr.decode() if e.stderr else "None")
        raise  # Re-raise the exception for debugging

    except Exception as e:
        print(f"Unexpected error when processing file: {file_path}")
        print(str(e))
        raise  # Re-raise the general exception

    audio = np.frombuffer(out, dtype=np.float32)

    # Convert to PyTorch tensor with shape (1, num_samples)
    return audio[None], sample_rate



def _pad_or_subsample_audio(audio: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Adjusts the audio tensor to a fixed length by randomly selecting a snippet if too long,
    or padding with zeros if too short.

    Args:
        audio (torch.Tensor): Input audio tensor of shape (channels, audio_length)
        max_length (int): Desired maximum length of the audio snippet

    Returns:
        torch.Tensor: Processed audio tensor of shape (channels, max_length)
    """
    channels, audio_length = audio.shape

    if audio_length > max_length:
        # Randomly select a start index
        start_idx = torch.randint(0, audio_length - max_length + 1, (1,)).item()
        audio = audio[:, start_idx:start_idx + max_length]
    elif audio_length < max_length:
        # Pad with zeros to the right
        pad = torch.zeros((channels, max_length - audio_length), dtype=audio.dtype, device=audio.device)
        audio = torch.cat([audio, pad], dim=1)

    return audio


def load_minimp3(fname, fpath, max_length=10.0, sample_rate=32000, random_sample_crop=True):
    # load only if necessary
    import minimp3py

    with open(fpath, 'rb') as f:
        data = f.read()

    duration, ch, sr = minimp3py.probe(data)
    if isinstance(max_length, float):
        max_length = int(max_length * sr)
    else:
        max_length = int(max_length * sr // sample_rate)
    offset = 0
    if random_sample_crop and duration > max_length:
        max_offset = max(int(duration - max_length), 0) + 1
        offset = torch.randint(max_offset, (1,)).item()
    waveform, _ = minimp3py.read(data, start=offset, length=max_length)
    waveform = waveform[:, 0]  # 0 for the first channel only
    if waveform.dtype != "float32":
        raise RuntimeError("Unexpected wave type")

    return fname, waveform[None, :]