import cv2
import numpy as np
import csv

def landmark_calib_v1(fr, shp_seq):
    fr_num = len(fr)
    shp_num = len(shp_seq)
    assert fr_num == shp_num
    h, w = fr[0].shape[:2]
    sz_norm = 400
    fcs = []
    locs = []
    shp_prm = []

    for i in range(fr_num):
        fr_i = fr[i]
        shp_i = shp_seq[i]
        fc, sz = shape_to_face(shp_i, w, h, 1.2)
        if sz == 0:
            continue
        fc_fr = fr_i[fc[1]: fc[3], fc[0]:fc[2]]
        if sz < sz_norm:
            inter_p = cv2.INTER_CUBIC
        else:
            inter_p = cv2.INTER_AREA
        fc_norm = cv2.resize(fc_fr, (sz_norm, sz_norm), interpolation=inter_p)
        sc_shp = sz_norm / sz
        shp_norm = np.rint((shp_i - np.array([fc[0], fc[1]])) * sc_shp).astype(int)
        fcs.append(fc_norm)
        shp_prm.append([fc[0], fc[1], sc_shp])
        locs.append(shp_norm)

    seg_len = 2
    loc_sum = len(locs)
    if loc_sum == 0:
        return []
    loc_track = [locs[0]]
    n_pts = 68
    Pp = np.array([0] * n_pts).reshape(n_pts).astype(float)

    for i in range(loc_sum - 1):
        fc_seg = fcs[i:i + seg_len]
        loc_seg = locs[i:i + seg_len]
        lk_prm = dict(winSize=(15, 15), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        start_p = loc_track[i].astype(np.float32)
        trg_p = loc_seg[1].astype(np.float32)

        fwd_p, s_fw, e_fw = cv2.calcOpticalFlowPyrLK(fc_seg[0], fc_seg[1], start_p, trg_p, **lk_prm,
                                                     flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        fbk_p, s_fb, e_fb = cv2.calcOpticalFlowPyrLK(fc_seg[1], fc_seg[0], fwd_p, start_p, **lk_prm,
                                                     flags=cv2.OPTFLOW_USE_INITIAL_FLOW)

        fwd_pts = [loc_track[i].copy(), fwd_p]
        fbk_pts = [fbk_p, fwd_p.copy()]

        fwd_pts = np.rint(fwd_pts).astype(int)
        fbk_pts = np.rint(fbk_pts).astype(int)

        merge_p, chk, Pp = merge(loc_seg, fwd_pts, fbk_pts, Pp, s_fw, s_fb)
        loc_track.append(merge_p)

    calib_norm_lndmks = []
    for i in loc_track:
        norm_base = sz_norm // 2
        shp = i - [norm_base, norm_base]
        shp = shp / norm_base
        shp = shp.ravel()
        shp = shp.tolist()
        calib_norm_lndmks.append(shp)

    return calib_norm_lndmks

def calibrator_v1(vf, lndmk_seq):
    vc = cv2.VideoCapture(vf)
    fr = []
    while True:
        succ, img = vc.read()
        if succ:
            fr.append(img)
        else:
            break
    calib_norm_lndmks = landmark_calib_v1(fr, lndmk_seq)
    vc.release()
    return np.array(calib_norm_lndmks)


def calibrator(video_file, landmark_sequence):
    frames = read_frames(video_file)
    calibrated_normalized_landmarks = calibrate_landmark(frames, landmark_sequence)
    return np.array(calibrated_normalized_landmarks)




