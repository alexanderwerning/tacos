import torch.nn as nn
import torch

class CutInputIntoSegmentsWrapper(nn.Module):
    def __init__(self, model, max_input_length, segment_length, hop_size):
        """
        Args:
            model (nn.Module): The PyTorch model to wrap.
            max_input_length (int): Maximum length of input the model can handle.
            segment_length (int): Length of each segment if input exceeds max_input_length.
            hop_size (int): Hop size for overlapping segmentation.
        """
        super().__init__()
        self.model = model
        self.max_input_length = max_input_length
        self.segment_length = segment_length
        self.hop_size = hop_size

    def forward(self, x):
        """Processes the input audio through the model, handling segmentation if needed."""
        batch_size, input_length = x.shape

        if input_length <= self.max_input_length:
            return self.model(x)  # Add segment dimension

        # Split into overlapping segments
        segments = []
        indices = list(range(0, input_length - self.segment_length + 1, self.hop_size))
        for i in indices:
            segments.append(x[:, i:i + self.segment_length])

        segments = torch.stack(segments)  # Shape: (num_segments, batch_size, segment_length)
        outputs = self.model(segments.reshape(-1, self.segment_length))  # Process each segment; output shape is (batch_size, d, l)
        _, d, l = outputs.shape
        outputs = outputs.view(len(indices), batch_size, d, l)  # (num_segments, batch_size, d, l)
        outputs = outputs.permute(1, 2, 0, 3)  # (batch_size, d, num_segments, l)
        outputs = outputs.contiguous().view(batch_size, d, -1)  # Final: (batch_size, d, total_sequence_length)

        # Return segments separately
        return outputs.permute(0, 2, 1)

import torch
import torch.nn as nn

def process_spectrogram_with_segments(x, model, max_input_length, segment_length, hop_size):
    """
    Args:
        x (torch.Tensor): Input tensor of shape (batch, channel, mel, time)
        model (nn.Module): PyTorch model that processes (batch, channel, mel, segment_length)
        max_input_length (int): Max time length the model can process directly.
        segment_length (int): Length of each segment along time dimension.
        hop_size (int): Hop size for overlapping segments.

    Returns:
        torch.Tensor: Output of shape (batch, total_time_out, features)
    """
    batch_size, channel, mel, time = x.shape

    if time <= max_input_length:
        out = model(x)  # Assume output shape: (batch, features, time_out)
        return out  # (batch, time_out, features)

    # Segment input along time axis
    segments = []
    indices = list(range(0, time - segment_length + 1, hop_size))
    for i in indices:
        segments.append(x[:, :, :, i:i + segment_length])  # (batch, channel, mel, segment_length)

    segments = torch.stack(segments)  # (num_segments, batch, channel, mel, segment_length)
    segments = segments.permute(1, 0, 2, 3, 4).reshape(-1, channel, mel, segment_length)  # (batch * num_segments, channel, mel, segment_length)

    # Pass segments through the model
    outputs = model(segments).permute(0, 2, 1)  # Expected output: (batch * num_segments, features, out_length)
    _, features, out_length = outputs.shape

    num_segments = len(indices)
    outputs = outputs.view(batch_size, num_segments, features, out_length)  # (batch, num_segments, features, out_length)
    outputs = outputs.permute(0, 2, 1, 3).contiguous().view(batch_size, features, -1)  # (batch, features, total_time_out)

    return outputs.permute(0, 2, 1)  # (batch, total_time_out, features)
