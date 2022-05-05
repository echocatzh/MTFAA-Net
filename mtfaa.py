"""
Multi-scale temporal frequency axial attention neural network (MTFAA).

shmzhang@aslp-npu.org, 2022
"""

import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as tf
from typing import List

from tfcm import TFCM
from asa import ASA
from phase_encoder import PhaseEncoder
from f_sampling import FD, FU
from erb import Banks
from stft import STFT


def parse_1dstr(sstr: str) -> List[int]:
    return list(map(int, sstr.split(",")))


def parse_2dstr(sstr: str) -> List[List[int]]:
    return [parse_1dstr(tok) for tok in sstr.split(";")]


eps = 1e-10


class MTFAANet(nn.Module):

    def __init__(self,
                 n_sig=1,
                 PEc=4,
                 Co="48,96,192",
                 O="1,1,1",
                 causal=True,
                 bottleneck_layer=2,
                 tfcm_layer=6,
                 mag_f_dim=3,
                 win_len=32*48,
                 win_hop=8*48,
                 nerb=256,
                 sr=48000,
                 win_type="hann",
                 ):
        super(MTFAANet, self).__init__()
        self.PE = PhaseEncoder(PEc, n_sig)
        # 32ms @ 48kHz
        self.stft = STFT(win_len, win_hop, win_len, win_type)
        self.ERB = Banks(nerb, win_len, sr)
        self.encoder_fd = nn.ModuleList()
        self.encoder_bn = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.decoder_fu = nn.ModuleList()
        self.decoder_bn = nn.ModuleList()
        C_en = [PEc//2*n_sig] + parse_1dstr(Co)
        C_de = [4] + parse_1dstr(Co)
        O = parse_1dstr(O)
        for idx in range(len(C_en)-1):
            self.encoder_fd.append(
                FD(C_en[idx], C_en[idx+1]),
            )
            self.encoder_bn.append(
                nn.Sequential(
                    TFCM(C_en[idx+1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    ASA(C_en[idx+1], causal=causal),
                )
            )

        for idx in range(bottleneck_layer):
            self.bottleneck.append(
                nn.Sequential(
                    TFCM(C_en[-1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    ASA(C_en[-1], causal=causal),
                )
            )

        for idx in range(len(C_de)-1, 0, -1):
            self.decoder_fu.append(
                FU(C_de[idx], C_de[idx-1], O=(O[idx-1], 0)),
            )
            self.decoder_bn.append(
                nn.Sequential(
                    TFCM(C_de[idx-1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    ASA(C_de[idx-1], causal=causal),
                )
            )
        # MEA is causal, so mag_t_dim = 1.
        self.mag_mask = nn.Conv2d(
            4, mag_f_dim, kernel_size=(3, 1), padding=(1, 0))
        self.real_mask = nn.Conv2d(4, 1, kernel_size=(3, 1), padding=(1, 0))
        self.imag_mask = nn.Conv2d(4, 1, kernel_size=(3, 1), padding=(1, 0))
        kernel = th.eye(mag_f_dim)
        kernel = kernel.reshape(mag_f_dim, 1, mag_f_dim, 1)
        self.register_buffer('kernel', kernel)
        self.mag_f_dim = mag_f_dim

    def forward(self, sigs):
        """
        sigs: list [B N] of len(sigs)
        """
        cspecs = []
        for sig in sigs:
            cspecs.append(self.stft.transform(sig))
        # D / E ?
        D_cspec = cspecs[0]
        mag = th.norm(D_cspec, dim=1)
        pha = torch.atan2(D_cspec[:, -1, ...], D_cspec[:, 0, ...])
        out = self.ERB.amp2bank(self.PE(cspecs))
        encoder_out = []
        for idx in range(len(self.encoder_fd)):
            out = self.encoder_fd[idx](out)
            encoder_out.append(out)
            out = self.encoder_bn[idx](out)

        for idx in range(len(self.bottleneck)):
            out = self.bottleneck[idx](out)

        for idx in range(len(self.decoder_fu)):
            out = self.decoder_fu[idx](out, encoder_out[-1-idx])
            out = self.decoder_bn[idx](out)
        out = self.ERB.bank2amp(out)
        # stage 1
        mag_mask = self.mag_mask(out)
        mag_pad = tf.pad(
            mag[:, None], [0, 0, (self.mag_f_dim-1)//2, (self.mag_f_dim-1)//2])
        mag = tf.conv2d(mag_pad, self.kernel)
        mag = mag * mag_mask.sigmoid()
        mag = mag.sum(dim=1)
        # stage 2
        real_mask = self.real_mask(out).squeeze(1)
        imag_mask = self.imag_mask(out).squeeze(1)

        mag_mask = th.sqrt(th.clamp(real_mask**2+imag_mask**2, eps))
        pha_mask = th.atan2(imag_mask+eps, real_mask+eps)
        real = mag * mag_mask.tanh() * th.cos(pha+pha_mask)
        imag = mag * mag_mask.tanh() * th.sin(pha+pha_mask)
        return mag, th.stack([real, imag], dim=1), self.stft.inverse(real, imag)


def test_nnet():
    # noise supression (microphone, )
    nnet = MTFAANet(n_sig=1)
    inp = th.randn(3, 48000)
    mag, cspec, wav = nnet([inp])
    print(mag.shape, cspec.shape, wav.shape)
    # echo cancellation (microphone, error, reference,)
    nnet = MTFAANet(n_sig=3)
    mag, cspec, wav = nnet([inp, inp, inp])
    print(mag.shape, cspec.shape, wav.shape)


def test_mac():
    from thop import profile, clever_format
    import torch as th
    nnet = MTFAANet(n_sig=3)
    # hop=8ms, win=32ms@48KHz, process 1s.
    inp = th.randn(1, 48000)
    # inp = th.randn(1, 2, 769, 126)
    macs, params = profile(nnet, inputs=([inp, inp, inp],), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('macs: ', macs)
    print('params: ', params)


if __name__ == "__main__":
    # test_nnet()
    test_mac()
