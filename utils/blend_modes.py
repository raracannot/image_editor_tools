import numpy as np


def apply_blend_mode(bg, fg, mode):
    eps = 1e-5
    B, L = bg, fg
    if mode == 'MIX':
        return L
    if mode == 'DARKEN':
        return np.minimum(B, L)
    if mode == 'MULTIPLY':
        return B * L
    if mode == 'BURN':
        return np.clip(1.0 - (1.0 - B) / (L + eps), 0, 1)
    if mode == 'LIGHTEN':
        return np.maximum(B, L)
    if mode == 'SCREEN':
        return 1.0 - (1.0 - B) * (1.0 - L)
    if mode == 'DODGE':
        return np.clip(B / (1.0 - L + eps), 0, 1)
    if mode == 'ADD':
        return np.clip(B + L, 0, 1)
    if mode == 'OVERLAY':
        return np.where(B < 0.5, 2.0 * B * L, 1.0 - 2.0 * (1.0 - B) * (1.0 - L))
    if mode == 'SOFT_LIGHT':
        return np.where(L < 0.5,
                        2.0 * B * L + B * B * (1.0 - 2.0 * L),
                        np.sqrt(B) * (2.0 * L - 1.0) + 2.0 * B * (1.0 - L))
    if mode == 'LINEAR_LIGHT':
        return np.clip(B + 2.0 * L - 1.0, 0, 1)
    if mode == 'DIFFERENCE':
        return np.abs(B - L)
    if mode == 'EXCLUSION':
        return B + L - 2.0 * B * L
    if mode == 'SUBTRACT':
        return np.clip(B - L, 0, 1)
    if mode == 'DIVIDE':
        return np.clip(B / (L + eps), 0, 1)
    return L
