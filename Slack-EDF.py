# =========================================================
# 纯贪心版本：移除GSGWO，使用严格slack约束的贪心策略
# =========================================================

import os
import cv2
import math
import random
import time
import numpy as np
from ultralytics import YOLO

# ===================== 配置区 =====================
OUTPUT_ROOT = r"E:/Software/ultralytics/ultralytics-8.3.40/Position/path planning"
PT_MODEL_PATH = r"E:\Software\ultralytics\ultralytics-8.3.40\Dataset\runs\train(CCFM+Dyhead2)\weights\best.pt"
IMG_PATH = r"E:/Software/ultralytics\ultralytics-8.3.40/Position/dataset/train/images/066.jpg"

TARGET_CLASSES = [0, 2]
SMALL_TH = 30 * 30
MEDIUM_TH = 50 * 50
TIME_DICT = {"Small": 1.0, "Medium": 2.0, "Large": 3.0}

# 平台向上移动（y 减小）
PLATFORM_V = -2.0

SIMULATION_TIME = 120.0
GALVO_MAX_SPEED = 1500.0
GALVO_MAX_ACCEL = 5000.0
TIME_STEP = 0.05

LASER_WIDTH = 864
FPS = 20
VIDEO_CODEC = 'mp4v'

# 全局变量
v1_global = v2_global = 0
uL1_global = uL2_global = uR1_global = uR2_global = 0


# ===================== 工具函数 =====================

def get_new_run_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("runs")]
    run_id = max([int(d.replace("runs", "")) for d in existing], default=0) + 1
    run_dir = os.path.join(base_dir, f"runs{run_id}")
    os.makedirs(run_dir)
    return run_dir


def get_weed_size_label(area):
    if area <= SMALL_TH:
        return "Small"
    elif area <= MEDIUM_TH:
        return "Medium"
    return "Large"


def optimized_center_localization(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return (x1 + x2) // 2, (y1 + y2) // 2
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    mask = (v > 0.35).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return (x1 + x2) // 2, (y1 + y2) // 2
    cx = int(M["m10"] / M["m00"] + x1)
    cy = int(M["m01"] / M["m00"] + y1)
    return cx, cy


# ===================== 振镜动力学 =====================

def get_current_y(p, t):
    """平台向上移动（y 减小），PLATFORM_V < 0"""
    return p["cy"] + PLATFORM_V * t


def theta_of_point(p, t, center_x):
    x = p["cx"] - center_x
    y = get_current_y(p, t)
    return math.atan2(y, x)


def trapezoidal_move_time(dtheta, vmax, amax):
    if dtheta < 1e-6:
        return 0.0
    t_acc = vmax / amax
    theta_acc = 0.5 * amax * t_acc ** 2
    if dtheta <= 2 * theta_acc:
        return 2 * math.sqrt(dtheta / amax)
    return 2 * t_acc + (dtheta - 2 * theta_acc) / vmax


def solve_chase_time(theta0, p, t0, center_x):
    vmax = np.deg2rad(GALVO_MAX_SPEED)
    amax = np.deg2rad(GALVO_MAX_ACCEL)

    def reachable(T):
        theta_t = theta_of_point(p, t0 + T, center_x)
        dtheta = abs(theta_t - theta0)
        move_time = trapezoidal_move_time(dtheta, vmax, amax)
        return move_time <= T

    if not reachable(0.5):
        return float("inf")

    lo, hi = 0.0, 10.0
    for _ in range(15):
        mid = (lo + hi) / 2
        if reachable(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def dynamic_cost(state, p, center_x):
    T_move = solve_chase_time(state["theta"], p, state["time"], center_x)
    if math.isinf(T_move):
        return float("inf"), None
    t_arr = state["time"] + T_move
    theta_arr = theta_of_point(p, t_arr, center_x)
    t_finish = t_arr + p["time"]
    return t_finish, {
        "theta": theta_arr,
        "time": t_finish,
        "move_time": T_move,
        "laser_time": p["time"]
    }


# ===================== Slack 计算（关键）====================

def calculate_slack(p, state, center_x, laser_y_bottom, t_current):
    """
    计算松弛时间 slack：
    - 平台向上移动（PLATFORM_V < 0），杂草相对向下移动（y 增大）
    - 杂草从上边界进入，向下边界移动，从下边界离开
    - 应该先除快离开的草（y 大的，靠近下边界）

    slack = 距离离开下边界的时间 - 移动时间 - 激光照射时间
    """
    current_y = get_current_y(p, t_current)

    # 已经离开下边界（y 太大，出界）
    if current_y > laser_y_bottom:
        return -9999

    # 距离离开下边界还剩多少时间（用绝对速度）
    time_to_exit = (laser_y_bottom - current_y) / abs(PLATFORM_V)

    # 估计移动时间
    theta_now = state["theta"]
    theta_target = math.atan2(current_y - v1_global, p["cx"] - center_x)
    dtheta = abs(theta_target - theta_now)
    vmax = np.deg2rad(GALVO_MAX_SPEED)
    amax = np.deg2rad(GALVO_MAX_ACCEL)
    est_move_time = trapezoidal_move_time(dtheta, vmax, amax)

    # 松弛时间
    slack = time_to_exit - est_move_time - p["time"]

    return slack


# ===================== 核心规划函数（纯贪心版）====================

def dynamic_online_planning(points, center_x, laser_y_top, laser_y_bottom, debug=False):
    global v1_global, v2_global
    v1_global, v2_global = laser_y_top, laser_y_bottom

    t = 0.0
    state = {"theta": 0.0, "time": 0.0}
    pool, done, exec_log = [], [], []
    total_tasks = len(points)

    # 按 y 降序排序（y 大的在前，因为 y 大的靠近下边界，会先离开）
    points = sorted(points, key=lambda p: p["cy"], reverse=True)
    idx = 0
    n_skipped = 0

    if debug:
        print(f"\n【规划调试】激光区: y=[{laser_y_top}, {laser_y_bottom}]")
        print(f"任务按 y 降序: {[p['cy'] for p in points[:5]]}...")

    while t < SIMULATION_TIME:
        # 1. 释放新任务（进入激光区上边界）
        # 平台向上移动，杂草 y 增大，从上边界进入
        while idx < len(points):
            p = points[idx]
            current_y = get_current_y(p, t)

            # 进入上边界：y >= laser_y_top
            if current_y >= laser_y_top:
                p["release"] = t
                p["deadline"] = SIMULATION_TIME
                p["original_cy"] = p["cy"]
                pool.append(p)
                idx += 1
            else:
                break

        if len(done) == total_tasks:
            if debug:
                print(f"[{t:.2f}s] 所有任务完成")
            break

        if not pool:
            t += TIME_STEP
            continue

        # 2. 计算所有任务的 slack
        task_slacks = []
        for i, p in enumerate(pool):
            slack = calculate_slack(p, state, center_x, laser_y_bottom, t)
            task_slacks.append((i, slack, p))

        # 过滤可行的
        feasible = [(i, s, p) for i, s, p in task_slacks if s >= 0]

        if not feasible:
            t += TIME_STEP
            continue

        # 3. 【纯贪心策略】选 slack 最小的（最紧急的）
        feasible.sort(key=lambda x: x[1])
        best_abs_idx, best_slack, best_p = feasible[0]

        if debug and len(done) < 5:
            print(f"[t={t:.2f}] 选择 y={best_p['original_cy']}, slack={best_slack:.3f}, "
                  f"pool={len(pool)}, feasible={len(feasible)}")

        # 4. 执行任务
        cost, new_state = dynamic_cost(state, best_p, center_x)

        if math.isinf(cost):
            pool.pop(best_abs_idx)
            n_skipped += 1
            continue

        state = new_state
        t = state["time"]
        done.append(best_p)

        exec_log.append({
            "t_start": t - best_p["time"],
            "t_end": t,
            "cx": best_p["cx"],
            "cy": best_p["original_cy"],
            "original_cy": best_p["original_cy"],
            "move_time": state["move_time"],
            "laser_time": best_p["time"],
            "slack_before": best_slack
        })

        pool.pop(best_abs_idx)

        if debug and len(done) % 10 == 0:
            print(f"[{t:.2f}s] 完成 {len(done)}/{total_tasks}")

    if debug:
        print(f"执行顺序 (y): {[log['original_cy'] for log in exec_log[:10]]}")
        print(f"总计完成: {len(done)}/{total_tasks}, 跳过: {n_skipped}")

    return done, exec_log, t


# ===================== 可视化 =====================

def draw_galvo_state(frame, points, exec_log, current_t, center_x, center_y,
                     color_done, color_pending, color_processing, galvo_color):
    if not exec_log:
        return frame

    h, w = frame.shape[:2]
    prev_pos = None

    for log in exec_log:
        if log['t_end'] <= current_t:
            pos = (int(log['cx']), int(log.get('original_cy', log['cy'])))
            if prev_pos is not None:
                cv2.line(frame, prev_pos, pos, color_done, 2)
            prev_pos = pos
            cv2.circle(frame, pos, 4, color_done, -1)

    galvo_pos = None
    for i, log in enumerate(exec_log):
        if log['t_start'] <= current_t <= log['t_end']:
            galvo_pos = (int(log['cx']), int(log.get('original_cy', log['cy'])))
            radius = int(10 + 5 * math.sin(current_t * 15))
            cv2.circle(frame, galvo_pos, radius, color_processing, 3)
            cv2.circle(frame, galvo_pos, 5, (255, 255, 255), -1)
            break

        if i < len(exec_log) - 1:
            next_log = exec_log[i + 1]
            if log['t_end'] < current_t < next_log['t_start']:
                ratio = (current_t - log['t_end']) / (next_log['t_start'] - log['t_end'] + 1e-6)
                ratio = max(0, min(1, ratio))
                curr_x = int(log['cx'] + (next_log['cx'] - log['cx']) * ratio)
                curr_y = int(log.get('original_cy', log['cy']) +
                             (next_log.get('original_cy', next_log['cy']) - log.get('original_cy', log['cy'])) * ratio)
                galvo_pos = (curr_x, curr_y)
                cv2.line(frame,
                         (int(log['cx']), int(log.get('original_cy', log['cy']))),
                         galvo_pos, (200, 200, 200), 1)
                break

    if galvo_pos is None:
        if exec_log:
            if current_t < exec_log[0]['t_start']:
                galvo_pos = (int(center_x), int(center_y))
            else:
                last = exec_log[-1]
                galvo_pos = (int(last['cx']), int(last.get('original_cy', last['cy'])))

    if galvo_pos:
        cv2.drawMarker(frame, galvo_pos, galvo_color, cv2.MARKER_CROSS, 20, 3)
        cv2.circle(frame, galvo_pos, 6, (255, 255, 255), -1)
        cv2.circle(frame, galvo_pos, 8, galvo_color, 2)

    for p in points:
        cx, cy = int(p['cx']), int(p['cy'])

        is_done = any(abs(log['cx'] - p['cx']) < 2 and
                      abs(log.get('original_cy', log['cy']) - p['cy']) < 2
                      and log['t_end'] <= current_t for log in exec_log)
        is_processing = any(abs(log['cx'] - p['cx']) < 2 and
                            abs(log.get('original_cy', log['cy']) - p['cy']) < 2
                            and log['t_start'] <= current_t <= log['t_end'] for log in exec_log)

        if is_done:
            color, radius = color_done, 6
        elif is_processing:
            color, radius = color_processing, 8
        else:
            color, radius = color_pending, 4

        cv2.circle(frame, (cx, cy), radius, color, -1)

        if 'time' in p:
            label = f"{p['time']:.0f}s"
            cv2.putText(frame, label, (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return frame


def create_simulation_video(img, left_points, right_points, left_log, right_log,
                            left_center_x, right_center_x, run_dir):
    global v1_global, v2_global, uL1_global, uL2_global, uR1_global, uR2_global

    h, w = img.shape[:2]
    v1, v2 = v1_global, v2_global
    uL1, uL2, uR1, uR2 = uL1_global, uL2_global, uR1_global, uR2_global
    center_y = (v1 + v2) // 2

    left_duration = left_log[-1]['t_end'] if left_log else 0
    right_duration = right_log[-1]['t_end'] if right_log else 0
    total_duration = min(max(left_duration, right_duration), SIMULATION_TIME)

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    video_path = os.path.join(run_dir, "simulation.mp4")
    out = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))

    time_step = 1.0 / FPS
    t = 0.0
    frame_count = 0

    print(f"生成视频: 总时长 {total_duration:.1f}s...")

    while t <= total_duration + time_step:
        frame = img.copy()

        overlay = frame.copy()
        cv2.rectangle(overlay, (uL1, v1), (uL2, v2), (0, 255, 0), -1)
        cv2.rectangle(overlay, (uR1, v1), (uR2, v2), (255, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
        cv2.rectangle(frame, (uL1, v1), (uL2, v2), (0, 255, 0), 2)
        cv2.rectangle(frame, (uR1, v1), (uR2, v2), (255, 0, 0), 2)

        frame = draw_galvo_state(frame, left_points, left_log, t, left_center_x, center_y,
                                 (0, 0, 255), (128, 128, 128), (0, 255, 255), (0, 0, 255))
        frame = draw_galvo_state(frame, right_points, right_log, t, right_center_x, center_y,
                                 (255, 0, 0), (128, 128, 128), (255, 255, 0), (255, 0, 0))

        left_done = sum(1 for log in left_log if log['t_end'] <= t)
        right_done = sum(1 for log in right_log if log['t_end'] <= t)

        cv2.putText(frame, f"Time: {t:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Left: {left_done}/{len(left_points)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(frame, f"Right: {right_done}/{len(right_points)}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)

        bar_width = 400
        bar_x = (w - bar_width) // 2
        progress = min(t / total_duration, 1.0) if total_duration > 0 else 0
        cv2.rectangle(frame, (bar_x, h - 50), (bar_x + bar_width, h - 30), (200, 200, 200), -1)
        cv2.rectangle(frame, (bar_x, h - 50), (bar_x + int(bar_width * progress), h - 30), (0, 255, 0), -1)
        cv2.rectangle(frame, (bar_x, h - 50), (bar_x + bar_width, h - 30), (255, 255, 255), 2)

        out.write(frame)
        t += time_step
        frame_count += 1

    out.release()
    print(f"视频完成: {frame_count} 帧")


def create_static_result(img, left_points, right_points, left_log, right_log, run_dir):
    global uL1_global, uL2_global, uR1_global, uR2_global, v1_global, v2_global
    h, w = img.shape[:2]
    vis = img.copy()

    cv2.rectangle(vis, (uL1_global, v1_global), (uL2_global, v2_global), (0, 255, 0), 2)
    cv2.rectangle(vis, (uR1_global, v1_global), (uR2_global, v2_global), (255, 0, 0), 2)

    for i in range(1, len(left_log)):
        p1 = (int(left_log[i - 1]['cx']), int(left_log[i - 1].get('original_cy', left_log[i - 1]['cy'])))
        p2 = (int(left_log[i]['cx']), int(left_log[i].get('original_cy', left_log[i]['cy'])))
        cv2.line(vis, p1, p2, (0, 0, 255), 2)

    for i in range(1, len(right_log)):
        p1 = (int(right_log[i - 1]['cx']), int(right_log[i - 1].get('original_cy', right_log[i - 1]['cy'])))
        p2 = (int(right_log[i]['cx']), int(right_log[i].get('original_cy', right_log[i]['cy'])))
        cv2.line(vis, p1, p2, (255, 0, 0), 2)

    for p in left_points:
        done = any(abs(l['cx'] - p['cx']) < 1 and
                   abs(l.get('original_cy', l['cy']) - p['cy']) < 1 for l in left_log)
        color = (0, 255, 0) if done else (0, 0, 0)
        cv2.circle(vis, (int(p['cx']), int(p['cy'])), 5, color, -1)

    for p in right_points:
        done = any(abs(l['cx'] - p['cx']) < 1 and
                   abs(l.get('original_cy', l['cy']) - p['cy']) < 1 for l in right_log)
        color = (255, 0, 0) if done else (128, 128, 128)
        cv2.circle(vis, (int(p['cx']), int(p['cy'])), 5, color, -1)

    cv2.imwrite(os.path.join(run_dir, "result.png"), vis)


def save_csv(left_log, right_log, run_dir):
    with open(os.path.join(run_dir, "left_path.csv"), "w") as f:
        f.write("t_start,t_end,cx,cy,original_cy,move_time,laser_time,slack\n")
        for log in left_log:
            f.write(f"{log['t_start']:.3f},{log['t_end']:.3f},{log['cx']:.1f},"
                    f"{log['cy']:.1f},{log.get('original_cy', log['cy']):.1f},"
                    f"{log['move_time']:.3f},{log['laser_time']:.1f},"
                    f"{log.get('slack_before', 0):.3f}\n")

    with open(os.path.join(run_dir, "right_path.csv"), "w") as f:
        f.write("t_start,t_end,cx,cy,original_cy,move_time,laser_time,slack\n")
        for log in right_log:
            f.write(f"{log['t_start']:.3f},{log['t_end']:.3f},{log['cx']:.1f},"
                    f"{log['cy']:.1f},{log.get('original_cy', log['cy']):.1f},"
                    f"{log['move_time']:.3f},{log['laser_time']:.1f},"
                    f"{log.get('slack_before', 0):.3f}\n")


# ===================== 主程序 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("智能除草动态路径规划系统 - 纯贪心版本")
    print("=" * 60)

    start_time = time.time()
    DEBUG_MODE = True

    run_dir = get_new_run_dir(OUTPUT_ROOT)
    print(f"输出目录: {run_dir}")

    print("加载模型...")
    model = YOLO(PT_MODEL_PATH)
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {IMG_PATH}")

    h, w, _ = img.shape
    print(f"图像尺寸: {w}x{h}")

    v_center = h // 2
    v1 = int(v_center - LASER_WIDTH / 2)
    v2 = int(v_center + LASER_WIDTH / 2)
    uL1 = int(w * 0.05)
    uL2 = uL1 + LASER_WIDTH
    uR1 = uL2
    uR2 = uR1 + LASER_WIDTH

    v1_global, v2_global = v1, v2
    uL1_global, uL2_global, uR1_global, uR2_global = uL1, uL2, uR1, uR2

    left_center_x = (uL1 + uL2) / 2
    right_center_x = (uR1 + uR2) / 2

    print(f"激光区: y=[{v1}, {v2}]")
    print(f"平台速度: {PLATFORM_V} (向上移动，y减小)")

    results = model(IMG_PATH)[0]
    all_points = []

    for box in results.boxes:
        cls_id = int(box.cls.item())
        if cls_id not in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)
        cx, cy = optimized_center_localization(img, (x1, y1, x2, y2))
        all_points.append({
            "cx": cx, "cy": cy, "area": area,
            "time": TIME_DICT[get_weed_size_label(area)]
        })

    left_points = [p for p in all_points if uL1 <= p["cx"] < uL2 and v1 <= p["cy"] < v2]
    right_points = [p for p in all_points if uR1 <= p["cx"] < uR2 and v1 <= p["cy"] < v2]

    print(f"左区: {len(left_points)} 个, 右区: {len(right_points)} 个")

    print(f"\n开始规划...")
    left_done, left_log, left_time = dynamic_online_planning(
        left_points, left_center_x, v1, v2, debug=DEBUG_MODE
    )

    right_done, right_log, right_time = dynamic_online_planning(
        right_points, right_center_x, v1, v2, debug=DEBUG_MODE
    )

    plan_time = time.time() - start_time

    print("-" * 60)
    left_duration = left_log[-1]['t_end'] if left_log else 0
    right_duration = right_log[-1]['t_end'] if right_log else 0
    makespan = max(left_duration, right_duration)

    left_success = len([l for l in left_log if l['t_end'] <= SIMULATION_TIME])
    right_success = len([l for l in right_log if l['t_end'] <= SIMULATION_TIME])

    print(f"规划耗时: {plan_time:.2f}秒")
    print(f"左振镜: {left_success}/{len(left_points)} 成功, 用时: {left_duration:.2f}s")
    print(f"右振镜: {right_success}/{len(right_points)} 成功, 用时: {right_duration:.2f}s")
    print(f"系统总完成时间: {makespan:.2f}s")
    print(f"成功率: {(left_success + right_success) / (len(left_points) + len(right_points)) * 100:.1f}%")

    print("\n保存结果...")
    save_csv(left_log, right_log, run_dir)
    create_static_result(img, left_points, right_points, left_log, right_log, run_dir)

    print("生成仿真视频...")
    create_simulation_video(img, left_points, right_points, left_log, right_log,
                            left_center_x, right_center_x, run_dir)

    print(f"\n完成! 总用时: {time.time() - start_time:.2f}秒")
    print(f"所有文件保存在: {run_dir}")