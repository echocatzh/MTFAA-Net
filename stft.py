"""
A simple wrapper for torch built-in STFT.
shmzhang@aslp-npu.org, 2022
"""

import torch as th
import torch.nn as nn
import einops


class STFT(nn.Module):
    def __init__(self, win_len, hop_len, fft_len, win_type):
        super(STFT, self).__init__()
        self.win, self.hop = win_len, hop_len
        self.nfft = fft_len
        window = {
            "hann": th.hann_window(win_len),
            "hamm": th.hamming_window(win_len),
        }
        assert win_type in window.keys()
        self.window = window[win_type]

    def transform(self, inp):
        """
        inp: B N
        """
        cspec = th.stft(inp, self.nfft, self.hop, self.win,
                        self.window.to(inp.device), return_complex=False)
        cspec = einops.rearrange(cspec, "b f t c -> b c f t")
        return cspec

    def inverse(self, real, imag):
        """
        real, imag: B F T
        """
        inp = th.stack([real, imag], dim=-1)
        return th.istft(inp, self.nfft, self.hop, self.win, self.window.to(real.device))
