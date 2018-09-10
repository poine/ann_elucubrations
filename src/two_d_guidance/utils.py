import numpy as np

def normalize_headings2(h):
    # normalize angles to [-pi; pi[
    return ( h + np.pi) % (2 * np.pi ) - np.pi

def normalize_headings(h):
    # normalize angles to ]-pi; pi]
    return -((np.pi - h) % (2 * np.pi ) - np.pi)

def pt_on_circle(c, r, th):
    return c + np.stack([r*np.cos(th), r*np.sin(th)], axis=-1)
