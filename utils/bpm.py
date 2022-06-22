import time

import numpy as np


class get_pulse(object):

    def __init__(self, bpm_limits=[50, 160]):
        self.bpm_limits = bpm_limits
        self.fps = 0
        self.buffer_size = 250
        self.data_buffer = []
        self.times = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.t0 = time.time()
        self.bpm = 0
        self.text = ''

    def get_subface_coord(self, crop_bgr, fhs):
        fh_x, fh_y, fh_w, fh_h = fhs
        h, w, _ = crop_bgr.shape
        return [int(w * fh_x - (w * fh_w / 2.0)),
                int(h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_subface_means(self, crop_bgr, coord):
        x, y, w, h = coord
        subframe = crop_bgr[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])

        return (v1 + v2 + v3) / 3.

    def run(self, crop_bgr):
        self.times.append(time.time() - self.t0)

        forehead1 = self.get_subface_coord(crop_bgr, (0.5, 0.18, 0.25, 0.15))
        vals = self.get_subface_means(crop_bgr, forehead1)

        self.data_buffer.append(vals)
        L = len(self.data_buffer)
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        processed = np.array(self.data_buffer)
        self.samples = processed
        if L > 10:
            self.output_dim = processed.shape[0]

            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)

            freqs = 60. * self.freqs
            idx = np.where((freqs > self.bpm_limits[0]) & (freqs < self.bpm_limits[1]))[0]
            idx = [i for i in idx if i < len(self.fft)]

            pruned = self.fft[idx]
            phase = phase[idx]

            self.freqs = freqs[idx]
            self.fft = pruned
            idx2 = np.argmax(pruned)

            t = (np.sin(phase[idx2]) + 1.) / 2.
            t = 0.9 * t + 0.1

            self.bpm = self.freqs[idx2]

            gap = (self.buffer_size - L) / self.fps
            self.text = "(heart rate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap) if gap else \
                        "(heart rate: %0.1f bpm)" % (self.bpm)
            
        return self.text
