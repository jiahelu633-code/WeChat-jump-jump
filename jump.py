import cv2
import subprocess
import math
import time
import sys
import os
import csv
import shutil
import numpy as np
from datetime import datetime

VERSION = "V5.2-Portable"
PRESS_SLOPE = 1.32
PRESS_OFFSET = 29.0

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

REF_WIDTH = 1080
REF_HEIGHT = 2400

RUNTIME_ROOT = os.path.join(
    os.path.expanduser("~"),
    ".jump_helper"
)
DEBUG_DIR = os.path.join(
    RUNTIME_ROOT,
    "runs",
    RUN_ID
)
LOG_PATH = os.path.join(
    DEBUG_DIR,
    "jump_log.csv"
)
GLOBAL_LOG_PATH = os.path.join(
    RUNTIME_ROOT,
    "jump_all.csv"
)
EXPORT_LOG_PATH = os.path.abspath(
    "jump_all.csv"
)

DEVICE_SCREEN_W = None
DEVICE_SCREEN_H = None

ISO_SLOPE = 0.60
BACKGROUND_DIFF = 18
COLOR_TOLERANCE = 12
MIN_RUN_WIDTH = 18
MIN_JUMP_DISTANCE = 100
MAX_JUMP_DISTANCE = 850
MIN_HORIZONTAL_DISTANCE = 120
MAX_COMPONENT_AREA = 250000
MAX_GEOMETRY_ERROR = 110
SCAN_STEP = 6

PLAYER_Y_MIN = 820
PLAYER_Y_MAX = 1720
FALL_Y = 1760
SUPPORT_MIN = 0.16

POST_JUMP_RETRIES = 14
TARGET_RETRIES = 5
STARTUP_RETRIES = 4
MISSING_DEATH_REQUIRED = 3

FAST_LANDING_FRAMES = 2
FAST_LANDING_SUPPORT = 0.90
FAST_LANDING_SPAN = 10

LANDING_STRONG_FRAMES = 3
LANDING_EDGE_FRAMES = 6
LANDING_MOVE_TOLERANCE = 18
LANDING_EDGE_SPAN = 8

TARGET_CONFIRM_TOLERANCE = 55
PLAYER_CONFIRM_TOLERANCE = 28
TARGET_HIGH_CONFIDENCE = 0.72

TRACK_GOOD_CONFIDENCE = 0.45

LANDING_PROBE_MIN_DELAY = 0.45
LANDING_PROBE_MAX_DELAY = 0.70
CALIBRATION_EVERY = 10
SAVE_NORMAL_DEBUG = False
DEBUG_CONFIDENCE_THRESHOLD = 0.80

os.makedirs(DEBUG_DIR, exist_ok=True)


def sync_export_log():
    if not os.path.exists(
        GLOBAL_LOG_PATH
    ):
        return

    try:
        shutil.copy2(
            GLOBAL_LOG_PATH,
            EXPORT_LOG_PATH
        )

        print(
            "✅ 累计日志已导出：",
            EXPORT_LOG_PATH
        )

    except Exception as e:
        print(
            "⚠️ 日志导出失败：",
            e
        )


def require_adb():
    if shutil.which("adb") is None:
        print(
            "❌ 找不到 adb。请先安装 Android Platform Tools，"
            "并确保 adb 已加入 PATH。"
        )
        sys.exit()


def get_device():
    require_adb()

    r = subprocess.run(
        ["adb", "devices"],
        capture_output=True,
        text=True
    )

    devices = []

    for line in r.stdout.splitlines():
        if "\tdevice" not in line:
            continue

        d = line.split("\t")[0].strip()

        if "_adb-tls-connect._tcp" in d:
            continue

        devices.append(d)

    devices = list(dict.fromkeys(devices))

    if not devices:
        print(
            "❌ 没有找到已连接的 Android 设备。\n"
            "USB：开启 USB 调试后连接电脑。\n"
            "无线：先执行 adb pair / adb connect。"
        )
        sys.exit()

    if len(devices) == 1:
        return devices[0]

    print("检测到多个设备：")

    for i, d in enumerate(devices, 1):
        print(f"  {i}. {d}")

    while True:
        try:
            n = int(
                input(
                    "请选择要控制的设备编号："
                )
            )

            if 1 <= n <= len(devices):
                return devices[n - 1]

        except (ValueError, EOFError):
            pass

        print("请输入有效编号。")


DEVICE = get_device()
print("✅ 当前设备：", DEVICE)


def capture_image():
    global DEVICE_SCREEN_W
    global DEVICE_SCREEN_H

    r = subprocess.run(
        [
            "adb", "-s", DEVICE,
            "exec-out",
            "screencap", "-p"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    if r.returncode != 0 or not r.stdout:
        return None

    data = np.frombuffer(
        r.stdout,
        dtype=np.uint8
    )

    raw = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if raw is None:
        return None

    h, w = raw.shape[:2]

    DEVICE_SCREEN_W = w
    DEVICE_SCREEN_H = h

    if h <= w:
        print(
            "❌ 当前不是竖屏。请保持手机竖屏后重试。"
        )
        return None

    if w == REF_WIDTH:
        return raw

    scale = REF_WIDTH / float(w)
    new_h = int(
        round(h * scale)
    )

    return cv2.resize(
        raw,
        (REF_WIDTH, new_h),
        interpolation=cv2.INTER_AREA
    )


def scaled_y(value, img):
    return int(
        round(
            value
            * img.shape[0]
            / REF_HEIGHT
        )
    )

def press(ms):
    if (
        DEVICE_SCREEN_W is None
        or DEVICE_SCREEN_H is None
    ):
        return False

    x = int(
        round(
            DEVICE_SCREEN_W * 0.50
        )
    )
    y = int(
        round(
            DEVICE_SCREEN_H * 0.75
        )
    )

    r = subprocess.run(
        [
            "adb", "-s", DEVICE,
            "shell", "input", "swipe",
            str(x), str(y),
            str(x), str(y),
            str(ms)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return r.returncode == 0


def press_time(distance):
    return int(round(
        PRESS_SLOPE * distance
        + PRESS_OFFSET
    ))


def landing_probe_delay(distance):
    delay = 0.40 + distance * 0.00035

    return min(
        LANDING_PROBE_MAX_DELAY,
        max(
            LANDING_PROBE_MIN_DELAY,
            delay
        )
    )


def _find_player_with_vmax(img, vmax):
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        np.array([115, 45, 20]),
        np.array([140, 255, vmax])
    )

    roi = np.zeros_like(mask)

    roi[
        int(h * 0.24):
        int(h * 0.90),
        :
    ] = 255

    mask = cv2.bitwise_and(
        mask,
        roi
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    count, labels, stats, _ = \
        cv2.connectedComponentsWithStats(mask)

    candidates = []

    for i in range(1, count):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if (
            400 < area < 12000
            and 25 < bw < 180
            and 50 < bh < 260
            and bh > bw * 1.15
        ):
            candidates.append(
                (area, x, y, bw, bh, i)
            )

    if not candidates:
        return None

    _, x, y, bw, bh, label_id = max(
        candidates,
        key=lambda item: item[0]
    )

    ys, xs = np.where(
        labels == label_id
    )

    if len(xs) == 0:
        return None

    foot_y = int(np.max(ys))

    bottom_xs = xs[
        ys >= y + int(bh * 0.82)
    ]

    if len(bottom_xs):
        foot_x = int(np.mean(bottom_xs))
    else:
        foot_x = int(x + bw / 2)

    return foot_x, foot_y


def find_player(img):
    player = _find_player_with_vmax(
        img,
        170
    )

    if player is not None:
        return player

    return _find_player_with_vmax(
        img,
        210
    )

def support_ratio(img, player):
    h, w = img.shape[:2]
    px, py = player

    samples = np.concatenate(
        [
            img[
                int(h * 0.3):
                int(h * 0.7),
                :50
            ].reshape(-1, 3),

            img[
                int(h * 0.3):
                int(h * 0.7),
                w - 50:
            ].reshape(-1, 3)
        ],
        axis=0
    )

    bg = np.median(
        samples,
        axis=0
    ).astype(np.int16)

    x1 = max(0, px - 55)
    x2 = min(w, px + 55)

    y1 = min(h - 1, py + 4)
    y2 = min(h, py + 50)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    patch = img[
        y1:y2,
        x1:x2
    ].astype(np.int16)

    diff = np.max(
        np.abs(patch - bg),
        axis=2
    )

    return float(
        np.mean(diff > 18)
    )


def find_runs(mask):
    xs = np.where(mask)[0]

    if len(xs) == 0:
        return []

    groups = np.split(
        xs,
        np.where(
            np.diff(xs) > 1
        )[0] + 1
    )

    return [
        g for g in groups
        if len(g) >= MIN_RUN_WIDTH
    ]


def get_color_component(
    img,
    img16,
    sx,
    sy,
    cache
):
    seed = tuple(
        int(v)
        for v in img[sy, sx]
    )

    if seed not in cache:
        color = np.array(
            seed,
            dtype=np.int16
        )

        diff = np.max(
            np.abs(
                img16 - color
            ),
            axis=2
        )

        mask = (
            diff <= COLOR_TOLERANCE
        ).astype(np.uint8)

        cache[seed] = \
            cv2.connectedComponentsWithStats(
                mask
            )

    _, labels, stats, centroids = \
        cache[seed]

    label_id = labels[sy, sx]

    if label_id == 0:
        return None

    x = stats[
        label_id,
        cv2.CC_STAT_LEFT
    ]

    y = stats[
        label_id,
        cv2.CC_STAT_TOP
    ]

    bw = stats[
        label_id,
        cv2.CC_STAT_WIDTH
    ]

    bh = stats[
        label_id,
        cv2.CC_STAT_HEIGHT
    ]

    area = stats[
        label_id,
        cv2.CC_STAT_AREA
    ]

    cx, cy = centroids[label_id]

    return {
        "center": (
            int(round(cx)),
            int(round(cy))
        ),
        "bbox": (
            int(x),
            int(y),
            int(bw),
            int(bh)
        ),
        "area": int(area)
    }


def is_duplicate(
    candidates,
    cx,
    cy
):
    return any(
        math.hypot(
            cx - item["center"][0],
            cy - item["center"][1]
        ) < 38
        for item in candidates
    )


def find_platform_candidates(
    img,
    player
):
    h, w = img.shape[:2]
    px, py = player

    candidates = []
    cache = {}
    img16 = img.astype(np.int16)

    start_y = int(h * 0.24)

    end_y = min(
        py - 25,
        int(h * 0.74)
    )

    for y in range(
        start_y,
        end_y,
        SCAN_STEP
    ):
        row = img[y]

        bg = np.median(
            row,
            axis=0
        ).astype(np.int16)

        row_diff = np.max(
            np.abs(
                row.astype(np.int16)
                - bg
            ),
            axis=1
        )

        foreground = (
            row_diff > BACKGROUND_DIFF
        )

        foreground[:40] = False
        foreground[w - 40:] = False

        for run in find_runs(
            foreground
        ):
            sx = int(
                (
                    int(run[0])
                    +
                    int(run[-1])
                ) / 2
            )

            seed_color = img[
                y,
                sx
            ].astype(np.int16)

            if (
                np.max(
                    np.abs(
                        seed_color - bg
                    )
                ) < 22
            ):
                continue

            c = get_color_component(
                img,
                img16,
                sx,
                y,
                cache
            )

            if c is None:
                continue

            cx, cy = c["center"]
            x, by, bw, bh = c["bbox"]
            area = c["area"]

            if (
                area < 600
                or area > MAX_COMPONENT_AREA
                or bw < 35
                or bh < 10
            ):
                continue

            if (
                x <= 10
                or x + bw >= w - 10
                or bw > w * 0.68
                or bh > h * 0.32
            ):
                continue

            dx = cx - px
            dy_up = py - cy

            if (
                dy_up < 40
                or abs(dx)
                < MIN_HORIZONTAL_DISTANCE
            ):
                continue

            distance = math.hypot(
                dx,
                dy_up
            )

            if not (
                MIN_JUMP_DISTANCE
                <= distance
                <= MAX_JUMP_DISTANCE
            ):
                continue

            geometry_error = abs(
                dy_up
                -
                ISO_SLOPE
                *
                abs(dx)
            )

            if (
                geometry_error
                >
                MAX_GEOMETRY_ERROR
            ):
                continue

            if is_duplicate(
                candidates,
                cx,
                cy
            ):
                continue

            c["distance"] = distance
            c["geometry_error"] = \
                geometry_error

            candidates.append(c)

    return candidates


def rect_gap(a, b):
    ax, ay, aw, ah = a["bbox"]
    bx, by, bw, bh = b["bbox"]

    dx = max(
        0,
        max(ax, bx)
        -
        min(ax + aw, bx + bw)
    )

    dy = max(
        0,
        max(ay, by)
        -
        min(ay + ah, by + bh)
    )

    return dx, dy


def cluster_candidates(candidates):
    clusters = []

    for c in candidates:
        placed = False

        for cluster in clusters:
            if any(
                (
                    rect_gap(c, other)[0] <= 42
                    and
                    rect_gap(c, other)[1] <= 42
                    and
                    math.hypot(
                        c["center"][0]
                        - other["center"][0],
                        c["center"][1]
                        - other["center"][1]
                    ) <= 220
                )
                for other in cluster
            ):
                cluster.append(c)
                placed = True
                break

        if not placed:
            clusters.append([c])

    changed = True

    while changed:
        changed = False
        merged = []

        while clusters:
            base = clusters.pop(0)
            i = 0

            while i < len(clusters):
                other = clusters[i]

                if any(
                    (
                        rect_gap(a, b)[0] <= 35
                        and
                        rect_gap(a, b)[1] <= 35
                    )
                    for a in base
                    for b in other
                ):
                    base.extend(
                        clusters.pop(i)
                    )
                    changed = True
                else:
                    i += 1

            merged.append(base)

        clusters = merged

    return clusters


def analyze_cluster(
    player,
    cluster
):
    px, py = player

    min_cy = min(
        c["center"][1]
        for c in cluster
    )

    ranked = []

    for c in cluster:
        cx, cy = c["center"]

        top_penalty = (
            cy - min_cy
        ) * 0.42

        area_bonus = min(
            c["area"],
            40000
        ) / 12000.0

        score = (
            c["geometry_error"]
            +
            top_penalty
            -
            area_bonus
        )

        ranked.append(
            (score, c)
        )

    ranked.sort(
        key=lambda item:
            item[0]
    )

    member_score, member = ranked[0]

    x, y, bw, bh = \
        member["bbox"]

    tx = int(
        round(
            x + bw / 2
        )
    )

    geometry_y = int(
        round(
            py
            -
            ISO_SLOPE
            *
            abs(tx - px)
        )
    )

    cy = member["center"][1]

    ty = int(
        round(
            geometry_y * 0.78
            +
            cy * 0.22
        )
    )

    target = (
        tx,
        ty
    )

    dx = tx - px
    dy_up = py - ty

    geometry_error = abs(
        dy_up
        -
        ISO_SLOPE
        *
        abs(dx)
    )

    topness = max(
        0,
        member["center"][1]
        - min_cy
    )

    cluster_area = sum(
        c["area"]
        for c in cluster
    )

    score = (
        geometry_error
        +
        topness * 0.35
        -
        min(cluster_area, 90000)
        / 18000.0
    )

    return {
        "target": target,
        "member": member,
        "cluster": cluster,
        "score": float(score),
        "geometry_error": float(
            geometry_error
        )
    }


def choose_target(
    candidates,
    player
):
    if not candidates:
        return None

    clusters = cluster_candidates(
        candidates
    )

    choices = [
        analyze_cluster(
            player,
            cluster
        )
        for cluster in clusters
    ]

    choices.sort(
        key=lambda item:
            item["score"]
    )

    best = choices[0]

    if len(choices) == 1:
        gap = 40.0
    else:
        gap = (
            choices[1]["score"]
            -
            best["score"]
        )

    geom = best[
        "geometry_error"
    ]

    confidence = (
        0.48
        +
        min(
            max(gap, 0.0),
            35.0
        ) / 110.0
        +
        max(
            0.0,
            35.0 - geom
        ) / 220.0
    )

    confidence = float(
        min(
            0.98,
            max(
                0.10,
                confidence
            )
        )
    )

    best["confidence"] = \
        confidence

    best["choice_count"] = \
        len(choices)

    return best


def save_jump_debug(
    img,
    player,
    candidates,
    choice,
    n
):
    debug = img.copy()

    cv2.circle(
        debug,
        player,
        16,
        (0, 255, 0),
        -1
    )

    for c in candidates:
        cv2.circle(
            debug,
            c["center"],
            11,
            (255, 255, 0),
            3
        )

    member = choice["member"]
    x, y, bw, bh = \
        member["bbox"]

    cv2.rectangle(
        debug,
        (x, y),
        (x + bw, y + bh),
        (255, 0, 255),
        4
    )

    target = choice[
        "target"
    ]

    cv2.circle(
        debug,
        target,
        20,
        (0, 0, 255),
        -1
    )

    cv2.line(
        debug,
        player,
        target,
        (0, 0, 255),
        5
    )

    cv2.putText(
        debug,
        f"C {choice['confidence']:.2f}",
        (
            max(
                10,
                target[0] - 70
            ),
            max(
                40,
                target[1] - 35
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(
        os.path.join(
            DEBUG_DIR,
            f"jump_{n:03d}.png"
        ),
        debug
    )


def track_target(
    pre_img,
    post_img,
    member,
    target,
    landing_player
):
    h, w = pre_img.shape[:2]

    x, y, bw, bh = \
        member["bbox"]

    margin = 25

    x1 = max(
        0,
        x - margin
    )

    y1 = max(
        0,
        y - margin
    )

    x2 = min(
        w,
        x + bw + margin
    )

    y2 = min(
        h,
        y + bh + margin
    )

    template = pre_img[
        y1:y2,
        x1:x2
    ]

    if (
        template.size == 0
        or
        template.shape[0] < 15
        or
        template.shape[1] < 15
    ):
        return None, 0.0

    lx, ly = landing_player

    sx1 = max(
        0,
        lx - 380
    )

    sx2 = min(
        w,
        lx + 380
    )

    sy1 = max(
        0,
        ly - 420
    )

    sy2 = min(
        h,
        ly + 300
    )

    search = post_img[
        sy1:sy2,
        sx1:sx2
    ]

    if (
        search.shape[0]
        <
        template.shape[0]
        or
        search.shape[1]
        <
        template.shape[1]
    ):
        return None, 0.0

    tg = cv2.cvtColor(
        template,
        cv2.COLOR_BGR2GRAY
    )

    sg = cv2.cvtColor(
        search,
        cv2.COLOR_BGR2GRAY
    )

    te = cv2.Canny(
        tg,
        40,
        120
    )

    se = cv2.Canny(
        sg,
        40,
        120
    )

    if (
        np.count_nonzero(te)
        < 20
    ):
        te = tg
        se = sg

    result = cv2.matchTemplate(
        se,
        te,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, max_loc = \
        cv2.minMaxLoc(result)

    matched_x = (
        sx1
        +
        max_loc[0]
    )

    matched_y = (
        sy1
        +
        max_loc[1]
    )

    shift_x = (
        matched_x
        -
        x1
    )

    shift_y = (
        matched_y
        -
        y1
    )

    tracked = (
        int(
            target[0]
            +
            shift_x
        ),
        int(
            target[1]
            +
            shift_y
        )
    )

    return (
        tracked,
        float(max_val)
    )


def landing_error(
    start_player,
    target,
    tracked_target,
    landing_player
):
    vx = (
        target[0]
        -
        start_player[0]
    )

    vy = (
        target[1]
        -
        start_player[1]
    )

    length = math.hypot(
        vx,
        vy
    )

    if length == 0:
        return None, None

    ux = vx / length
    uy = vy / length

    ex = (
        landing_player[0]
        -
        tracked_target[0]
    )

    ey = (
        landing_player[1]
        -
        tracked_target[1]
    )

    along = (
        ex * ux
        +
        ey * uy
    )

    cross = (
        -ex * uy
        +
        ey * ux
    )

    return (
        float(along),
        float(cross)
    )


CSV_HEADER = [
    "version",
    "run_id",
    "jump",
    "player_x",
    "player_y",
    "target_x",
    "target_y",
    "distance",
    "duration_ms",
    "candidate_count",
    "cluster_count",
    "cluster_size",
    "target_confidence",
    "target_confirmed",
    "geometry_error",
    "result",
    "seconds",
    "landing_x",
    "landing_y",
    "landing_support",
    "tracked_target_x",
    "tracked_target_y",
    "track_confidence",
    "error_along",
    "error_cross",
    "quality"
]


def append_csv(
    path,
    row
):
    exists = os.path.exists(path)

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow(
                CSV_HEADER
            )

        writer.writerow(row)


def write_log(
    n,
    player,
    choice,
    distance,
    ms,
    candidates,
    confirmed,
    result,
    seconds,
    landing_player,
    landing_support,
    tracked_target,
    track_confidence,
    error_along,
    error_cross
):
    target = choice[
        "target"
    ]

    if (
        result == "landed"
        and tracked_target is not None
        and track_confidence is not None
        and track_confidence
        >= TRACK_GOOD_CONFIDENCE
        and error_cross is not None
        and abs(error_cross) <= 80
        and choice[
            "geometry_error"
        ] <= 75
    ):
        quality = "good"
    else:
        quality = "review"

    row = [
        VERSION,
        RUN_ID,
        n,
        player[0],
        player[1],
        target[0],
        target[1],
        round(distance, 2),
        ms,
        len(candidates),
        choice["choice_count"],
        len(choice["cluster"]),
        round(
            choice["confidence"],
            3
        ),
        int(bool(confirmed)),
        round(
            choice["geometry_error"],
            2
        ),
        result,
        round(seconds, 2),
        (
            landing_player[0]
            if landing_player
            else ""
        ),
        (
            landing_player[1]
            if landing_player
            else ""
        ),
        (
            round(
                landing_support,
                3
            )
            if landing_support
            is not None
            else ""
        ),
        (
            tracked_target[0]
            if tracked_target
            else ""
        ),
        (
            tracked_target[1]
            if tracked_target
            else ""
        ),
        (
            round(
                track_confidence,
                3
            )
            if track_confidence
            is not None
            else ""
        ),
        (
            round(
                error_along,
                2
            )
            if error_along
            is not None
            else ""
        ),
        (
            round(
                error_cross,
                2
            )
            if error_cross
            is not None
            else ""
        ),
        quality
    ]

    append_csv(
        LOG_PATH,
        row
    )

    append_csv(
        GLOBAL_LOG_PATH,
        row
    )


def position_span(points):
    if not points:
        return 9999.0

    xs = [
        p[0]
        for p in points
    ]

    ys = [
        p[1]
        for p in points
    ]

    return float(
        max(
            max(xs) - min(xs),
            max(ys) - min(ys)
        )
    )


def get_next_frame(n):
    last_img = None
    last_player = None
    last_support = None

    history = []
    missing_streak = 0
    edge_pending = False

    for attempt in range(
        1,
        POST_JUMP_RETRIES + 1
    ):
        img = capture_image()

        if img is None:
            continue

        last_img = img

        player = find_player(
            img
        )

        if player is None:
            missing_streak += 1
            history.clear()
            edge_pending = False

            print(
                f"落地检测 "
                f"{attempt}/{POST_JUMP_RETRIES}："
                f"找不到人物 "
                f"{missing_streak}/{MISSING_DEATH_REQUIRED}"
            )

            if (
                attempt >= 3
                and
                missing_streak
                >= MISSING_DEATH_REQUIRED
            ):
                cv2.imwrite(
                    os.path.join(
                        DEBUG_DIR,
                        f"gameover_{n:03d}.png"
                    ),
                    img
                )

                return (
                    None,
                    "game_over",
                    last_player,
                    last_support,
                    img
                )

            continue

        missing_streak = 0

        support = support_ratio(
            img,
            player
        )

        last_player = player
        last_support = support

        if player[1] >= scaled_y(FALL_Y, img):
            cv2.imwrite(
                os.path.join(
                    DEBUG_DIR,
                    f"gameover_{n:03d}.png"
                ),
                img
            )

            return (
                None,
                "game_over",
                player,
                support,
                img
            )

        if not (
            scaled_y(PLAYER_Y_MIN, img)
            <= player[1]
            <= scaled_y(PLAYER_Y_MAX, img)
        ):
            print(
                f"落地检测 "
                f"{attempt}/{POST_JUMP_RETRIES}："
                f"位置 {player} 超出范围"
            )

            history.clear()
            edge_pending = False
            continue

        history.append(player)

        if len(history) > \
            LANDING_EDGE_FRAMES:
            history.pop(0)

        fast = False
        strong = False
        edge = False

        if len(history) >= \
            FAST_LANDING_FRAMES:
            recent_fast = history[
                -FAST_LANDING_FRAMES:
            ]

            fast = (
                position_span(recent_fast)
                <= FAST_LANDING_SPAN
                and
                support >= FAST_LANDING_SUPPORT
            )

        if len(history) >= \
            LANDING_STRONG_FRAMES:
            recent = history[
                -LANDING_STRONG_FRAMES:
            ]

            strong = (
                position_span(recent)
                <= LANDING_MOVE_TOLERANCE
                and
                support >= SUPPORT_MIN
            )

        if len(history) >= \
            LANDING_EDGE_FRAMES:
            edge = (
                position_span(history)
                <= LANDING_EDGE_SPAN
            )

        print(
            f"落地检测 "
            f"{attempt}/{POST_JUMP_RETRIES}："
            f"位置 {player} "
            f"支撑 {support:.2f} "
            f"跨度 "
            f"{position_span(history):.1f}px"
        )

        if fast:
            print(
                "⚡ 确认落地：快速"
            )

            return (
                img,
                "landed",
                player,
                support,
                img
            )

        if strong:
            print(
                "✅ 确认落地：普通"
            )

            return (
                img,
                "landed",
                player,
                support,
                img
            )

        if edge:
            if edge_pending:
                print(
                    "✅ 确认落地：边缘/小平台"
                )

                return (
                    img,
                    "landed",
                    player,
                    support,
                    img
                )

            edge_pending = True
            print(
                "🟡 疑似边缘/小平台，再确认一帧"
            )
        else:
            edge_pending = False

    if last_img is not None:
        cv2.imwrite(
            os.path.join(
                DEBUG_DIR,
                f"landing_fail_{n:03d}.png"
            ),
            last_img
        )

    return (
        None,
        "not_landed",
        last_player,
        last_support,
        last_img
    )


def detect_target_once(img):
    if img is None:
        return None, "capture_failed"

    player = find_player(img)

    if player is None:
        return None, "player_missing"

    if player[1] >= scaled_y(FALL_Y, img):
        return None, "game_over"

    candidates = \
        find_platform_candidates(
            img,
            player
        )

    choice = choose_target(
        candidates,
        player
    )

    if choice is None:
        return None, "target_missing"

    return (
        player,
        candidates,
        choice
    ), "ok"


def find_target_with_retry(img):
    current = img
    player_missing = 0

    for attempt in range(
        1,
        TARGET_RETRIES + 1
    ):
        data1, status1 = \
            detect_target_once(
                current
            )

        if status1 == "game_over":
            return (
                None,
                None,
                False,
                "game_over"
            )

        if data1 is None:
            if status1 == \
                "player_missing":
                player_missing += 1
            else:
                player_missing = 0

            print(
                f"🔄 目标检测 "
                f"{attempt}/{TARGET_RETRIES}："
                f"{status1}"
            )

            if (
                player_missing
                >= MISSING_DEATH_REQUIRED
            ):
                return (
                    current,
                    None,
                    False,
                    "game_over"
                )

            current = capture_image()
            continue

        player_missing = 0

        p1, c1, choice1 = data1

        if (
            choice1["confidence"]
            >= TARGET_HIGH_CONFIDENCE
        ):
            return (
                current,
                data1,
                False,
                "ok"
            )

        confirm_img = capture_image()

        data2, status2 = \
            detect_target_once(
                confirm_img
            )

        if data2 is None:
            print(
                f"🔄 低置信目标二次确认 "
                f"{attempt}/{TARGET_RETRIES}："
                f"{status2}"
            )

            current = confirm_img
            continue

        p2, c2, choice2 = data2

        t1 = choice1[
            "target"
        ]

        t2 = choice2[
            "target"
        ]

        player_move = math.hypot(
            p2[0] - p1[0],
            p2[1] - p1[1]
        )

        target_move = math.hypot(
            t2[0] - t1[0],
            t2[1] - t1[1]
        )

        if (
            player_move
            <= PLAYER_CONFIRM_TOLERANCE
            and
            target_move
            <= TARGET_CONFIRM_TOLERANCE
        ):
            print(
                f"✅ 低置信目标二次确认通过 "
                f"C={choice2['confidence']:.2f}"
            )

            return (
                confirm_img,
                data2,
                True,
                "ok"
            )

        print(
            f"🔄 目标不稳定 "
            f"{attempt}/{TARGET_RETRIES}："
            f"人物差 {player_move:.1f}px，"
            f"目标差 {target_move:.1f}px"
        )

        current = confirm_img

    return (
        current,
        None,
        False,
        "target_missing"
    )


print()
print("==============================")
print(f"跳一跳助手 {VERSION}")
print("按压模型：1.32 × 距离 + 29ms")
print("目标：平台聚类 + 顶面优先")
print("高置信目标不强制二次截图")
print("Speed：延后首帧 + 高置信 2 帧快速落地")
print("人物检测：深色优先，兼容紫色平台")
print("适配：自动归一化竖屏分辨率 + 相对按压坐标")
print(f"落点校准：每 {CALIBRATION_EVERY} 跳抽样一次")
print("彩色背景不再参与游戏判断")
print("Control + C = 立即停止")
print("运行数据：~/.jump_helper")
print("结束后导出：当前目录/jump_all.csv")
print("Run ID：", RUN_ID)
print("==============================")


current_img = None
startup_player = None
last_startup_img = None

for attempt in range(
    1,
    STARTUP_RETRIES + 1
):
    img = capture_image()

    if img is None:
        print(
            f"🔄 启动检测 {attempt}/{STARTUP_RETRIES}："
            "截图失败"
        )
        continue

    last_startup_img = img

    player = find_player(img)

    if player is not None:
        current_img = img
        startup_player = player
        break

    print(
        f"🔄 启动检测 {attempt}/{STARTUP_RETRIES}："
        "暂时找不到人物"
    )

if current_img is None:
    if last_startup_img is not None:
        fail_path = os.path.join(
            DEBUG_DIR,
            "startup_player_fail.png"
        )

        cv2.imwrite(
            fail_path,
            last_startup_img
        )

        print(
            "📸 已保存启动失败现场：",
            fail_path
        )

    print(
        "❌ 连续多次仍检测不到人物，"
        "请确认人物已经站在游戏平台上"
    )
    sys.exit()


jump_count = 0


try:
    while True:
        (
            target_img,
            data,
            confirmed,
            status
        ) = find_target_with_retry(
            current_img
        )

        if status == "game_over":
            print(
                "💀 连续检测不到人物，"
                "本局锁定结束"
            )
            break

        if data is None:
            print(
                "🛑 连续多次仍找不到"
                "可靠目标平台，停止"
            )

            if target_img is not None:
                cv2.imwrite(
                    os.path.join(
                        DEBUG_DIR,
                        "target_fail.png"
                    ),
                    target_img
                )

            break

        current_img = target_img

        (
            player,
            candidates,
            choice
        ) = data

        jump_count += 1
        start_time = time.time()

        target = choice[
            "target"
        ]

        distance = math.hypot(
            target[0]
            -
            player[0],
            target[1]
            -
            player[1]
        )

        ms = press_time(
            distance
        )

        print()
        print(
            f"========== 第 "
            f"{jump_count} 跳 =========="
        )

        print(
            "人物：",
            player
        )

        print(
            "目标：",
            target
        )

        print(
            f"距离："
            f"{distance:.1f}px"
        )

        print(
            f"长按："
            f"{ms}ms"
        )

        print(
            f"候选："
            f"{len(candidates)} 个 "
            f"| 平台组："
            f"{choice['choice_count']} 个"
        )

        print(
            f"目标置信："
            f"{choice['confidence']:.2f} "
            f"| 二次确认："
            f"{'是' if confirmed else '否'}"
        )

        debug_saved = False

        if (
            SAVE_NORMAL_DEBUG
            or confirmed
            or choice["confidence"]
            < DEBUG_CONFIDENCE_THRESHOLD
        ):
            save_jump_debug(
                current_img,
                player,
                candidates,
                choice,
                jump_count
            )
            debug_saved = True

        if not press(ms):
            print(
                "🛑 ADB 长按失败"
            )
            break

        probe_delay = landing_probe_delay(
            distance
        )

        time.sleep(
            probe_delay
        )

        (
            next_img,
            result,
            landing_player,
            landing_support,
            post_frame
        ) = get_next_frame(
            jump_count
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        tracked_target = None
        track_confidence = None
        error_along = None
        error_cross = None

        should_calibrate = (
            result == "landed"
            and
            post_frame is not None
            and
            landing_player is not None
            and
            (
                jump_count
                % CALIBRATION_EVERY == 0
                or confirmed
                or choice["confidence"]
                < DEBUG_CONFIDENCE_THRESHOLD
            )
        )

        if should_calibrate:
            (
                tracked_target,
                track_confidence
            ) = track_target(
                current_img,
                post_frame,
                choice[
                    "member"
                ],
                target,
                landing_player
            )

            if tracked_target is not None:
                (
                    error_along,
                    error_cross
                ) = landing_error(
                    player,
                    target,
                    tracked_target,
                    landing_player
                )

                if error_along is not None:
                    direction = (
                        "远"
                        if error_along > 0
                        else "短"
                    )

                    print(
                        f"📊 落点误差："
                        f"{direction} "
                        f"{abs(error_along):.1f}px "
                        f"| 横向 "
                        f"{error_cross:.1f}px "
                        f"| 追踪可信 "
                        f"{track_confidence:.2f}"
                    )

        if (
            result != "landed"
            and not debug_saved
        ):
            save_jump_debug(
                current_img,
                player,
                candidates,
                choice,
                jump_count
            )
            debug_saved = True

        write_log(
            jump_count,
            player,
            choice,
            distance,
            ms,
            candidates,
            confirmed,
            result,
            elapsed,
            landing_player,
            landing_support,
            tracked_target,
            track_confidence,
            error_along,
            error_cross
        )

        print(
            f"本跳耗时："
            f"{elapsed:.2f}s"
        )

        if next_img is None:
            if result == "game_over":
                print(
                    "💀 确认本局结束，程序锁定停止"
                )
            else:
                print(
                    "🛑 无法确认落地，停止"
                )

            break

        current_img = next_img


except KeyboardInterrupt:
    print()
    print(
        "🛑 已人工停止"
    )


sync_export_log()

print()
print("==============================")
print(
    f"结束，本次实际跳跃 "
    f"{jump_count} 次"
)
print(
    "本局日志：",
    LOG_PATH
)
print(
    "累计日志（本地）：",
    GLOBAL_LOG_PATH
)
print(
    "桌面快照：",
    EXPORT_LOG_PATH
)
print(
    "Debug：",
    DEBUG_DIR
)
print("==============================")
