import pkg_resources

__author__ = "John Wiseman jjwiseman@gmail.com; Modified by Tanapat Kahabodeekanokkul pahntanapat@gmail.com"
__copyright__ = "Copyright (C) 2016 John Wiseman; Copyright (C) 2021 Tanapat Kahabodeekanokkul"
__license__ = "MIT"
__version__ = pkg_resources.get_distribution('webrtcvad').version

from typing import Iterable, Tuple
import numpy as np
import _webrtcvad

VAD_MODE = 0
VAD_WINDOWS_ms = 10
VAD_MIN_SPEECH_DURATION_ms = 200


class VAD:
    """VAD for deployment
    """
    @staticmethod
    def to_16bit_PCM(wav: Iterable[float]):
        return np.around(
            np.clip((((wav + 1) * 65535) / 2) - 32768, -32768,
                    32767)).astype(np.int16).astype('<u2').tobytes()

    def __init__(self,
                 mode=VAD_MODE,
                 sampling_rate: int = 16000,
                 windows_ms: int = VAD_WINDOWS_ms,
                 min_speech_ms: float = VAD_MIN_SPEECH_DURATION_ms):

        self._vad = _webrtcvad.create()
        _webrtcvad.init(self._vad)
        if mode is not None:
            self.set_mode(mode)

        assert sampling_rate in {
            8000, 16000, 32000, 48000
        }, 'The sample_rate must be in 8000, 16000, 32000, or 48000 Hz.'
        assert windows_ms in {10, 20, 30
                              }, 'The windows_ms must be in 10, 20, or 30 ms.'

        #assert valid_rate_and_frame_length(self.sr, self.frame_width) # 2 bytes per sound member
        self.sr = sampling_rate
        self.frame_width = int((self.sr * windows_ms) / 1000)

        self.window = int((windows_ms * self.sr) / 1000)
        self.bin_window = self.window * 2

        self.min_vad_window = int(min_speech_ms / windows_ms)

    def set_mode(self, mode):
        _webrtcvad.set_mode(self._vad, mode)

    def __call__(
        self,
        wav: Iterable[float],
        keep_remain: bool = None,
    ) -> Tuple[Iterable[bool], int]:
        """VAD voice
        Args:
            wav: Iterable[float] - wave signal -1 to 1
            keep_remain: bool - return lenght of remained wav which is not fitted to window, whether padding and VAD

        Returns:
            Tuple[VAD, remain index]: VAD result of windows, and remain index from lastest [-rem:]
        """

        l = len(wav)
        edge = l * 2
        rem = l % self.window

        #print(rem)
        if rem:
            if keep_remain:
                #rem =  #wav[-rem:]
                pcm = self.to_16bit_PCM(wav[:-rem])
            else:
                ## Pad by rem part

                pcm = self.to_16bit_PCM(wav)  #+ (b'\0' * (2 * (std - rem)))
                pcm = pcm + (pcm[-(rem * 2):] * (
                    (self.window // rem) * 2)) + pcm[-(2 *
                                                       (self.window % rem)):]
                #rem =  std%rem  # equivalent to ((window - rem)%rem)  > 0

                rem = 0
        else:
            rem = 0
            pcm = self.to_16bit_PCM(wav)

        voice = [
            _webrtcvad.process(self._vad, self.sr,
                               pcm[p:(p + self.bin_window)], self.window)
            for p in range(0, edge, self.bin_window)
        ]

        del pcm

        return [np.asarray(voice, dtype=bool), rem]


def valid_rate_and_frame_length(rate, frame_length):
    return _webrtcvad.valid_rate_and_frame_length(rate, frame_length)
