# -*- coding: utf-8 -*-
"""
영상 생체역학 "시퀀스" feature 파이프라인 모듈
==============================================
릴리스 1프레임이 아니라 투구 장면 전체 프레임을 시계열로 사용하는 버전.
video_features.py(정적 9-stat 집계)와 같은 소스(03_skeleton.ipynb가 저장한 좌표)를
쓰지만, 여기서는 프레임을 버리지 않고 고정 길이로 리샘플링해 시퀀스 텐서를 만든다.
05_video_sequence_pipeline.ipynb가 이 모듈을 import해서 호출한다.

흐름:
    batch_slot*_seq.parquet (03_skeleton.ipynb, EXTRACT_SEQUENCE=True 출력)
        --merge_sequences-->   프레임 단위 long table + 경기정보(game_pk/pitcher/season)
        --build_pitch_sequences--> 투구별 (T, 9) 각도 시퀀스 (좌투 미러링·정규화 포함)
        --build_game_tensor-->     경기 단위 텐서 (n_games, max_pitches, T, 9) + mask

⚠ 투구 순서 관련 중요한 제약
--------------------------------
02_video_collect.ipynb가 만드는 play_ids_sample.csv는 Statcast 검색 페이지에서
반환된 순서로 play_id를 모은 것이라 실제 pitch_number(투구 순서)를 보존하지 않는다
(README "다음 단계" 항목에도 명시된 한계 — play_id 포함 재수집 전까지는 해결 불가).
따라서 이 모듈이 만드는 텐서의 "투구 축(pitch axis)"은 **순서가 있다고 가정하면 안 된다**.
build_game_tensor()의 투구 축은 set(순서 무관)으로 취급해야 하며, 이를 소비하는
17_video_sequence_experiment.ipynb의 모델도 이 축엔 (순서를 학습하는 CNN 대신)
masked mean/std pooling처럼 순서-불변 집계를 사용한다.
반면 "프레임 축(frame axis, 투구 1개 내부)"은 진짜 시간 순서이므로 CNN을 써도 된다.
"""

import os
import glob
import numpy as np
import pandas as pd


# ── 상수 (03_skeleton.ipynb의 JOINT_NAMES와 반드시 일치해야 함) ─────────────
JOINT_NAMES = [
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# video_features.py의 ANGLE_COLS와 동일 — 각도 정의 자체는 바꾸지 않고
# "정지 1프레임 → 프레임 전체 시퀀스"만 바꾸는 것이 이 모듈의 목적이다.
ANGLE_COLS = [
    'stride_norm', 'arm_slot', 'shoulder_tilt', 'hip_tilt',
    'trunk_dist_norm', 'trunk_angle', 'separation',
    'release_height_norm', 'arm_extension_norm',
]

T_RESAMPLE = 32     # 투구 1개를 몇 스텝으로 리샘플링할지 (가변 프레임 수 → 고정 길이)
MAX_PITCHES = 15    # 경기당 최대 투구 수 (정형 X구간 pitch15와 동일 기준)


# ── 1. 시퀀스 parquet 합치기 + 경기정보 조인 ────────────────────────────────
def _load_play_ids(play_ids_csv):
    play = pd.read_csv(play_ids_csv, dtype={'play_id': str})
    play['game_pk'] = play['game_pk'].astype('int64')
    play['season'] = play['season'].astype('int64')
    return play[['play_id', 'pitcher_id', 'game_pk', 'season']]


def iter_sequence_batches(output_dir, play_ids_csv, slots=(0, 1, 2, 3, 4)):
    """batch_slot*_seq.parquet(03_skeleton.ipynb 신규 출력)을 파일(zip 배치) 단위로
    하나씩 읽어 play_id 조인까지 마친 뒤 yield한다.

    ⚠ merge_sequences()처럼 전체를 한 DataFrame으로 합치지 않는 이유: 이 데이터는
    "프레임 1개 = 1행"인 long format이라 투구(영상) 수만 개 × 프레임 수백 개 =
    수천만 행이 될 수 있다. 전체를 한 번에 메모리에 올리면 Colab 표준 RAM에서
    OOM 위험이 있어, zip(배치) 단위(영상 ~200개분)로 스트리밍 처리한다.
    한 영상의 프레임은 항상 같은 배치 parquet 파일 안에만 있으므로(03_skeleton.ipynb의
    flush_seq가 zip 단위로 저장) 배치 단위로 끊어 처리해도 투구가 잘리지 않는다.
    """
    play_cols = _load_play_ids(play_ids_csv)

    any_file = False
    for slot in slots:
        files = sorted(glob.glob(os.path.join(output_dir, f'batch_slot{slot}_*_seq.parquet')))
        for f in files:
            d = pd.read_parquet(f)
            if not len(d):
                continue
            any_file = True

            merged = d.merge(play_cols, left_on='video_name', right_on='play_id', how='left')
            merged = merged.dropna(subset=['game_pk', 'pitcher_id']).reset_index(drop=True)
            if not len(merged):
                continue
            merged['game_pk'] = merged['game_pk'].astype('int64')
            merged['pitcher_id'] = merged['pitcher_id'].astype('int64')
            merged = merged.rename(columns={'pitcher_id': 'pitcher'}).drop(columns=['play_id'])
            merged['src_slot'] = slot
            yield merged

    if not any_file:
        raise FileNotFoundError(
            f'시퀀스 parquet 없음: {output_dir} slots={slots} '
            f'(03_skeleton.ipynb를 EXTRACT_SEQUENCE=True로 재실행해야 함)'
        )


def merge_sequences(output_dir, play_ids_csv, slots=(0, 1, 2, 3, 4)):
    """batch_slot*_seq.parquet들을 전부 합쳐 하나의 DataFrame으로 반환.

    ⚠ **대규모 데이터에서 메모리(OOM) 위험이 있다** — iter_sequence_batches() 참고.
    소규모 확인/디버깅(슬롯 1개, 배치 몇 개 정도)에서만 쓰고, 실제 파이프라인
    (build_and_save)은 이 함수를 쓰지 않고 build_pitch_sequences_streaming()으로
    배치 단위 스트리밍 처리한다.
    """
    frames = list(iter_sequence_batches(output_dir, play_ids_csv, slots=slots))
    return pd.concat(frames, ignore_index=True)


# ── 2. 좌투 미러링 (프레임 단위 적용 — video_features.mirror_lefties와 동일 로직) ──
def mirror_lefties_seq(df):
    """좌투(L)를 우투 기준으로 통일: x좌표 미러 + left/right 관절 swap.
    long format(1행=1프레임)에 그대로 적용 가능 — 각 행이 이미 독립된 좌표 스냅샷이라
    video_features.mirror_lefties()와 동일한 방식으로 행 단위로 처리하면 된다.
    """
    df = df.copy()
    x_cols = [c for c in df.columns if c.endswith('_x')]
    is_left = (df['hand'] == 'L')
    if is_left.any():
        row_xmax = df.loc[is_left, x_cols].max(axis=1)
        for c in x_cols:
            df.loc[is_left, c] = row_xmax - df.loc[is_left, c]
        for name in JOINT_NAMES:
            if name.startswith('left_'):
                rname = 'right_' + name[len('left_'):]
                for ax in ['x', 'y']:
                    lc, rc = f'{name}_{ax}', f'{rname}_{ax}'
                    if lc in df.columns and rc in df.columns:
                        tmp = df.loc[is_left, lc].copy()
                        df.loc[is_left, lc] = df.loc[is_left, rc].values
                        df.loc[is_left, rc] = tmp.values
    return df


# ── 3. 가변 길이 → 고정 길이 리샘플링 ────────────────────────────────────────
def resample_sequence(xy, T=T_RESAMPLE):
    """가변 프레임 수 좌표 시퀀스 (n_frames, J, 2) → 고정 T 스텝 선형보간.

    프레임 수를 그대로 패딩하지 않고 리샘플링하는 이유: 투구 장면 길이가
    영상마다 다른데(수십~수백 프레임), 동작의 '페이즈'(와인드업→릴리스→팔로우스루
    비율)를 정렬하는 게 프레임 개수를 맞추는 것보다 CNN이 패턴을 배우기 쉽다.
    """
    n = xy.shape[0]
    if n == 1:
        return np.repeat(xy, T, axis=0)
    src_pos = np.linspace(0.0, 1.0, n)
    dst_pos = np.linspace(0.0, 1.0, T)
    out = np.empty((T,) + xy.shape[1:], dtype='float64')
    for j in range(xy.shape[1]):
        out[:, j, 0] = np.interp(dst_pos, src_pos, xy[:, j, 0])
        out[:, j, 1] = np.interp(dst_pos, src_pos, xy[:, j, 1])
    return out


# ── 4. 프레임별 각도 계산 (video_features.compute_angles와 동일 정의, 시간축 벡터화) ──
def _joint_angle(P, a, b, c):
    """b를 꼭짓점으로 한 a-b-c 각도(도). P[name]: (T, 2) 배열(마지막 축이 x,y)."""
    v1 = P[a] - P[b]
    v2 = P[c] - P[b]
    n1 = np.sqrt((v1 ** 2).sum(-1))
    n2 = np.sqrt((v2 ** 2).sum(-1))
    cos = (v1 * v2).sum(-1) / np.where((n1 * n2) < 1e-6, np.nan, n1 * n2)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def compute_angle_sequence(resampled_xy, joint_names=JOINT_NAMES):
    """리샘플링된 (T, J, 2) 좌표 → (T, 9) 각도 시퀀스.
    video_features.compute_angles()와 동일한 9개 각도 정의를 시간축으로 확장한 것
    (우투 기준 — mirror_lefties_seq 적용 후 호출해야 함).
    """
    idx = {name: i for i, name in enumerate(joint_names)}
    P = {name: resampled_xy[:, idx[name], :] for name in joint_names}

    sw = np.sqrt(((P['right_shoulder'] - P['left_shoulder']) ** 2).sum(-1))
    sw = np.where(sw < 1e-6, np.nan, sw)

    ang = {}
    stride = np.sqrt(((P['left_ankle'] - P['right_ankle']) ** 2).sum(-1))
    ang['stride_norm'] = stride / sw
    ang['arm_slot'] = _joint_angle(P, 'right_wrist', 'right_elbow', 'right_shoulder')
    ang['shoulder_tilt'] = np.degrees(np.arctan2(
        P['right_shoulder'][:, 1] - P['left_shoulder'][:, 1],
        P['right_shoulder'][:, 0] - P['left_shoulder'][:, 0]))
    ang['hip_tilt'] = np.degrees(np.arctan2(
        P['right_hip'][:, 1] - P['left_hip'][:, 1],
        P['right_hip'][:, 0] - P['left_hip'][:, 0]))

    hipc = (P['left_hip'] + P['right_hip']) / 2
    shlc = (P['left_shoulder'] + P['right_shoulder']) / 2
    ang['trunk_dist_norm'] = np.sqrt(((shlc - hipc) ** 2).sum(-1)) / sw
    ang['trunk_angle'] = np.degrees(np.arctan2(shlc[:, 1] - hipc[:, 1], shlc[:, 0] - hipc[:, 0]))

    sep = np.abs(ang['shoulder_tilt'] - ang['hip_tilt']) % 360
    ang['separation'] = np.where(sep > 180, 360 - sep, sep)

    ang['release_height_norm'] = (P['right_shoulder'][:, 1] - P['right_wrist'][:, 1]) / sw
    ang['arm_extension_norm'] = (P['right_wrist'][:, 0] - P['right_shoulder'][:, 0]) / sw

    return np.stack([ang[c] for c in ANGLE_COLS], axis=-1)  # (T, 9)


# ── 5. 투구별 (T, 9) 각도 시퀀스 딕셔너리 구성 ──────────────────────────────
def build_pitch_sequences(seq_df, T=T_RESAMPLE):
    """merge_sequences() 결과(long format, 프레임 단위) → 투구별 각도 시퀀스.

    Returns
    -------
    dict: video_name -> {'angles': (T,9) ndarray, 'release_rel': float(0~1),
                          'game_pk': int, 'pitcher': int, 'season': int}
    """
    mirrored = mirror_lefties_seq(seq_df)
    out = {}
    for video_name, g in mirrored.groupby('video_name', sort=False):
        g = g.sort_values('t')
        n = len(g)
        xy = np.zeros((n, len(JOINT_NAMES), 2), dtype='float64')
        for j, name in enumerate(JOINT_NAMES):
            xy[:, j, 0] = g[f'{name}_x'].to_numpy()
            xy[:, j, 1] = g[f'{name}_y'].to_numpy()

        resampled = resample_sequence(xy, T=T)
        angles = compute_angle_sequence(resampled)
        # 어깨너비(sw)가 거의 0인 퇴화 프레임에서 각도가 NaN이 될 수 있음 → 0으로 치환
        # (딥러닝 모델 입력에 NaN이 섞이면 loss가 NaN으로 터지므로 여기서 정리)
        angles = np.nan_to_num(angles, nan=0.0, posinf=0.0, neginf=0.0)

        release_idx = float(g['release_idx'].iloc[0])
        n_frames = float(g['n_frames'].iloc[0])
        release_rel = release_idx / max(n_frames - 1.0, 1.0)  # 리샘플링과 무관한 0~1 상대위치

        out[video_name] = {
            'angles': angles,
            'release_rel': release_rel,
            'game_pk': int(g['game_pk'].iloc[0]),
            'pitcher': int(g['pitcher'].iloc[0]),
            'season': int(g['season'].iloc[0]),
        }
    return out


def build_pitch_sequences_streaming(output_dir, play_ids_csv, slots=(0, 1, 2, 3, 4), T=T_RESAMPLE):
    """iter_sequence_batches()로 배치(zip) 단위로 읽어가며 build_pitch_sequences 로직을 적용.

    merge_sequences() + build_pitch_sequences()와 결과는 동일하지만, 한 번에 메모리에
    올라가는 게 배치 1개(영상 ~200개)분 뿐이라 대규모 데이터에서도 OOM 위험이 없다.
    실제 파이프라인(build_and_save)은 이 함수를 쓴다.
    """
    out = {}
    n_batches = 0
    for batch_df in iter_sequence_batches(output_dir, play_ids_csv, slots=slots):
        out.update(build_pitch_sequences(batch_df, T=T))
        n_batches += 1
        if n_batches % 20 == 0:
            print(f'  ...배치 {n_batches}개 처리, 누적 투구 {len(out):,}개')
    print(f'배치 {n_batches}개 처리 완료, 총 투구 {len(out):,}개')
    return out


# ── 6. 경기 단위 텐서 조립 (투구 축 = 순서 무관 set) ────────────────────────
def build_game_tensor(pitch_seqs, max_pitches=MAX_PITCHES, T=T_RESAMPLE):
    """투구별 시퀀스 딕셔너리 → 경기 단위 텐서.

    투구 축 순서는 임의(수집된 순서)이며 실제 pitch_number가 아니다 — 모듈 상단
    docstring 참고. 학습 시 이 축엔 순서를 학습하는 레이어(CNN/RNN)를 쓰면 안 되고
    mask 기반 mean/std pooling 같은 순서-불변 집계만 사용해야 한다.

    Returns
    -------
    X     : (n_games, max_pitches, T, 9) float32, 0-padded
    X_rel : (n_games, max_pitches) float32, 릴리스 상대위치(0~1), 패딩=0
    mask  : (n_games, max_pitches) float32, 실제 투구=1 / 패딩=0
    meta  : DataFrame[game_pk, pitcher, season, n_pitches_used]
    """
    games = {}
    for v, d in pitch_seqs.items():
        key = (d['game_pk'], d['pitcher'], d['season'])
        games.setdefault(key, []).append(d)

    keys = sorted(games.keys())
    n_games = len(keys)
    X = np.zeros((n_games, max_pitches, T, len(ANGLE_COLS)), dtype='float32')
    X_rel = np.zeros((n_games, max_pitches), dtype='float32')
    mask = np.zeros((n_games, max_pitches), dtype='float32')
    meta_rows = []

    for gi, key in enumerate(keys):
        pitches = games[key][:max_pitches]  # 초과분은 자름 (정형 X구간 pitch15와 동일한 정신)
        for pi, d in enumerate(pitches):
            X[gi, pi] = d['angles']
            X_rel[gi, pi] = d['release_rel']
            mask[gi, pi] = 1.0
        meta_rows.append({'game_pk': key[0], 'pitcher': key[1], 'season': key[2],
                           'n_pitches_used': len(pitches)})

    meta = pd.DataFrame(meta_rows)
    return X, X_rel, mask, meta


# ── 7. end-to-end 일괄 실행 ──────────────────────────────────────────────
def build_and_save(output_dir, play_ids_csv, out_dir, slots=(0, 1, 2, 3, 4),
                    T=T_RESAMPLE, max_pitches=MAX_PITCHES):
    """시퀀스 parquet → 경기 단위 텐서(.npz) + 메타(parquet) 저장까지 일괄 실행.

    ⚠ merge_sequences()(전체를 한 DataFrame으로 합침)를 쓰지 않고
    build_pitch_sequences_streaming()으로 배치 단위 처리한다 — 대규모 데이터에서
    OOM을 피하기 위함(모듈 상단 iter_sequence_batches() docstring 참고).
    """
    os.makedirs(out_dir, exist_ok=True)
    pitch_seqs = build_pitch_sequences_streaming(output_dir, play_ids_csv, slots=slots, T=T)

    X, X_rel, mask, meta = build_game_tensor(pitch_seqs, max_pitches=max_pitches, T=T)
    print(f'경기 텐서: X{X.shape}  (n_games, max_pitches, T, angles)')
    print(f'경기당 평균 투구 수: {meta["n_pitches_used"].mean():.1f} / {max_pitches}')

    npz_path = os.path.join(out_dir, f'video_seq_pitch{max_pitches}_T{T}.npz')
    meta_path = os.path.join(out_dir, f'video_seq_pitch{max_pitches}_T{T}_meta.parquet')
    np.savez_compressed(npz_path, X=X, X_rel=X_rel, mask=mask)
    meta.to_parquet(meta_path, index=False)
    print(f'저장 완료: {npz_path}')
    print(f'저장 완료: {meta_path}')
    return X, X_rel, mask, meta
