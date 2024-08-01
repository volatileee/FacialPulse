import numpy as np
import cv2
def shape_to_face_v1(pts, w, h, s=1.2):
    mnx, mny = np.min(pts, axis=0)
    mxx, mxy = np.max(pts, axis=0)
    cx, cy = (mnx + mxx) // 2, (mny + mxy) // 2
    sz = int(max(mxx - mnx, mxy - mny) * s)
    sz = sz // 2 * 2
    x1, y1 = max(cx - sz // 2, 0), max(cy - sz // 2, 0)
    sz = min(w - x1, sz)
    sz = min(h - y1, sz)
    x2, y2 = x1 + sz, y1 + sz
    return [int(x1), int(y1), int(x2), int(y2)], sz

def merge(loc, fwd, fbk, Pp, s_fw=None, s_fb=None):
    n = 68
    chk = [True] * n
    trg = loc[1]
    fwd_p = fwd[1]
    fwd_b = fwd[0]
    fbk_p = fbk[0]
    fbk_d = fbk_p - fwd_b
    fbk_dist = np.linalg.norm(fbk_d, axis=1, keepdims=True)
    det_d = loc[1] - loc[0]
    det_dist = np.linalg.norm(det_d, axis=1, keepdims=True)
    prd_d = fwd[1] - fwd[0]
    prd_dist = np.linalg.norm(prd_d, axis=1, keepdims=True)
    prd_dist[np.where(prd_dist == 0)] = 1
    Pd = (det_dist / prd_dist).reshape(n)

    for i in range(n):
        if fbk_dist[i] > 2:
            chk[i] = False

    if s_fw is not None and np.sum(s_fw) != n:
        for i in range(n):
            if s_fw[i][0] == 0:
                chk[i] = False
    if s_fw is not None and np.sum(s_fb) != n:
        for i in range(n):
            if s_fb[i][0] == 0:
                chk[i] = False
    loc_m = trg.copy()
    Q = 0.3

    for i in range(n):
        if chk[i]:
            Pp[i] += Q
            K = Pp[i] / (Pp[i] + Pd[i])
            loc_m[i] = fwd_p[i] + K * (trg[i] - fwd_p[i])
            Pp[i] = (1 - K) * Pp[i]

    return loc_m, chk, Pp
