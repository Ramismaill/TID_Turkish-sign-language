import numpy as np

POSE_D = 99
LH_START, LH_END = 99, 162
RH_START, RH_END = 162, 225


def _x_indices(d=225):
    return np.arange(0, d, 3)


def temporal_resample(x, target_len):
    t, d = x.shape
    if t == target_len:
        return x
    old_idx = np.linspace(0, 1, t)
    new_idx = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, d), dtype=np.float32)
    for i in range(d):
        out[:, i] = np.interp(new_idx, old_idx, x[:, i])
    return out


class LandmarkAugment:
    def __init__(self, target_len=64, p_flip=0.5, p_jitter=0.7, p_scale=0.7,
                 p_rotate=0.6, p_time=0.6, p_drop=0.4):
        self.target_len = target_len
        self.p_flip = p_flip
        self.p_jitter = p_jitter
        self.p_scale = p_scale
        self.p_rotate = p_rotate
        self.p_time = p_time
        self.p_drop = p_drop

    def __call__(self, x):
        x = x.copy().astype(np.float32)

        if np.random.rand() < self.p_jitter:
            sigma = np.random.uniform(0.004, 0.012)
            x += np.random.normal(0, sigma, size=x.shape).astype(np.float32)

        if np.random.rand() < self.p_scale:
            s = np.random.uniform(0.85, 1.15)
            x *= s

        if np.random.rand() < self.p_rotate:
            theta = np.deg2rad(np.random.uniform(-15, 15))
            c, s = np.cos(theta), np.sin(theta)
            xyz = x.reshape(x.shape[0], -1, 3)
            xx = xyz[:, :, 0].copy()
            yy = xyz[:, :, 1].copy()
            xyz[:, :, 0] = c * xx - s * yy
            xyz[:, :, 1] = s * xx + c * yy
            x = xyz.reshape(x.shape[0], -1)

        if np.random.rand() < self.p_flip:
            x[:, _x_indices(x.shape[1])] *= -1.0
            lh = x[:, LH_START:LH_END].copy()
            rh = x[:, RH_START:RH_END].copy()
            x[:, LH_START:LH_END] = rh
            x[:, RH_START:RH_END] = lh

        if np.random.rand() < self.p_time:
            t_new = np.random.randint(48, 81)
            x = temporal_resample(x, t_new)
            x = temporal_resample(x, self.target_len)

        if np.random.rand() < self.p_drop:
            k = np.random.randint(1, 6)
            idx = np.random.choice(x.shape[0], size=k, replace=False)
            x[idx] = 0.0

        return x