#!/usr/bin/env python3
""" Convolutional Backpropagation """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same",
                  stride=(1, 1)):
    """ func performs back propagation over convolutional layer """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        pad_h = int(np.ceil(((h_prev * (sh - 1)) - sh + kh) / 2))
        pad_w = int(np.ceil(((w_prev * (sw - 1)) - sw + kw) / 2))
    else:
        pad_h = 0
        pad_w = 0
    img_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h),
                              (pad_w, pad_w), (0, 0)),
                     'constant', constant_values=(0))
    dA_prev = np.zeros(img_pad.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for z in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    dA_prev[z, i * sh:i * sh + kh,
                            j * sw:j * sw + kw, :] += (W[:, :, :, k] *
                                                       dZ[z, i, j, k])
                    dW[:, :, :, k] += (img_pad[z, i * sh:i * sh + kh,
                                               j * sw:j * sw + kw, :] *
                                       dZ[z, i, j, k])
    dA_h = dA_prev.shape[1]
    dA_w = dA_prev.shape[2]
    dA_prev = dA_prev[:, pad_h:dA_h - pad_h, pad_w:dA_w - pad_w, :]
    return dA_prev, dW, db
