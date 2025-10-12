import torch

import torchaudio.functional as F
import torchaudio
from src.models.prediction_wrapper import PredictionsWrapper
from src.models.asit.ASIT_wrapper import ASiTWrapper
from src.models.wrapper import CutInputIntoSegmentsWrapper

class ASITSEDWrapper(torch.nn.Module):

    def __init__(self, checkpoint='ASIT_strong_1', n_classes_strong=768, rnn_layers=1, seq_len=1491):
        super().__init__()
        """
        Args:
            s_patchout_t (int): Temporal patchout size.
            s_patchout_f (int): Frequency patchout size.
        """
        super().__init__()
        self.base_model = ASiTWrapper()
        self.model = PredictionsWrapper(
            base_model=self.base_model,
            checkpoint=checkpoint,
            rnn_layers=rnn_layers,
            n_classes_strong=n_classes_strong,
            window_length=999,
            hop_size=999,
            seq_len=seq_len
        )

        self.resample = torchaudio.transforms.Resample(
                            32000,
                            16000,
                            resampling_method="sinc_interp_kaiser"
                        )

    def forward(self, x, duration=None):

        with torch.cuda.amp.autocast(False):
            x = self.resample(x)

        mel = self.model.mel_forward(x)

        tokens = self.model(mel, duration=duration)[0] # get embedding, not token
        return tokens



if __name__ == "__main__":
    wrapper = ASITSEDWrapper()

    input = torch.zeros(1, 30*32000)
    output = wrapper(input, duration=torch.tensor([1]))
    print(output)