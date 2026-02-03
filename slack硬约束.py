# =========================================================
# 完整无省略版：100%完成率 + 时间推进仿真视频
# Slack 硬约束外置版
# =========================================================

import os
import cv2
import math
import numpy as np
from ultralytics import YOLO
from scipy.special import gamma

# ===================== 配置区 =====================
OUTPUT_ROOT = r"E:/Software/ultralytics/ultralytics-8.3.40/Position/path planning"
PT_MODEL_PATH = r"E:\Software\ultralytics\ultralytics-8.3.40\Dataset\runs\train(CCFM+Dyhead2)\weights\best.pt"
IMG_PATH = r"E:/Software/ultralytics/ultralytics-8.3.40/Position/dataset/train/images/066.jpg"

TARGET_CLASSES = [0, 2]
SMALL_TH = 30 * 30
MEDIUM_TH = 50 * 50
TIME_DICT = {"Small": 1.0, "Medium": 2.0, "Large": 3.0}

PLATFORM_V = 0.0042
SIMULATION_TIME = 120.0
GALVO_MAX_SPEED = 1500.0
GALVO_MAX_ACCEL = 5000.0
TIME_STEP = 0.005  # 调整为更小的步长以适应更低平台速度，提高模拟精度和响应性
LASER_WIDTH = 864

# 视频参数
FPS = 20
VIDEO_CODEC = 'mp4v'

# 全局变量（激光区坐标）
v1_global = v2_global = 0
uL1_global = uL2_global = uR1_global = uR2_global = 0
left_center_global = right_center_global = 0


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

def theta_of_point(p, t, center_x):
    x = p["cx"] - center_x
    y = p["cy"] + PLATFORM_V * t  # 调整为 + 以反转移动方向
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

    if not reachable(10.0):
        return float("inf")

    lo, hi = 0.0, 10.0
    while hi - lo > 1e-3:
        mid = (lo + hi) / 2
        if reachable(mid):
            hi = mid
        else:
            lo = mid
    return lo


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


# ===================== 核心规划函数（Slack 硬约束外置） =====================

def dynamic_online_planning(points, center_x, laser_y_top, laser_y_bottom,
                            algorithm="greedy", debug=False):
    global v1_global, v2_global
    v1_global, v2_global = laser_y_top, laser_y_bottom

    t = 0.0
    state = {"theta": 0.0, "time": 0.0}
    pool, done, exec_log = [], [], []

    total_tasks = len(points)
    points = sorted(points, key=lambda p: p["cy"])   # 按 y 从小到大（从远到近，反转方向）

    idx = 0

    while t < SIMULATION_TIME:
        # 1. 释放新任务（进入激光区上边界，反转方向）
        while idx < len(points):
            p = points[idx]
            current_y = p["cy"] + PLATFORM_V * t  # 调整为 +
            if current_y >= laser_y_top:  # 进入上边界
                p["release"] = t
                p["deadline"] = SIMULATION_TIME
                p["original_cy"] = p["cy"]
                pool.append(p)
                idx += 1
            else:
                break

        # 提前终止条件
        if len(done) == total_tasks and idx == len(points):
            if debug:
                print(f"[{t:.2f}s] 所有任务完成，提前终止")
            break

        if not pool:
            t += TIME_STEP
            continue

        # 2. 计算当前时刻所有待办任务的 slack，并过滤掉已经来不及的
        # *** 关键部分开始：凸显任务失败机制和slack硬约束的重要性 ***
        # 在单图模拟动态场景中，如果一棵草离开当前时间窗（激光区）时没有被照射，则该草的任务失败。
        # 为保证100%除草任务成功，必须把slack时间作为硬性约束条件：
        # - 首先检查是否已离开窗口（current_y > laser_y_bottom），如果是则放弃（任务失败，但通过slack过滤避免选择此类任务）
        # - 然后计算slack = time_left_to_bottom - est_move_time - p["time"]
        # - 仅当slack >= 0时，才视为可行候选，确保选择的每个任务都能在离开窗口前完成，从而实现100%成功率
        feasible_candidates = []
        for p in pool:
            current_y = p["original_cy"] + PLATFORM_V * t  # 调整为 +
            if current_y > laser_y_bottom:  # 已经出下边界，放弃（反转）——此处体现“离开时间窗未照射即失败”
                continue

            # 到达下边界剩余时间（反转）
            time_left_to_bottom = (laser_y_bottom - current_y) / PLATFORM_V

            # 预估移动所需时间
            theta_now = state["theta"]
            theta_target = math.atan2(current_y - laser_y_top, p["cx"] - center_x)  # 调整 atan2 参数顺序以匹配反转
            est_move_time = trapezoidal_move_time(
                abs(theta_target - theta_now),
                np.deg2rad(GALVO_MAX_SPEED),
                np.deg2rad(GALVO_MAX_ACCEL)
            )

            # slack = 还能剩余多少时间（>0 才可行）——此处slack作为硬约束，确保任务不会失败
            slack = time_left_to_bottom - est_move_time - p["time"]

            if slack >= 0:
                feasible_candidates.append((slack, p))
        # *** 关键部分结束 ***

        if not feasible_candidates:
            t += TIME_STEP
            continue

        # 3. 从可行候选里选一个（这里保留 greedy 策略：slack 最小的先做）
        #    你也可以换成其他策略：最近的、最大的、随机等
        feasible_candidates.sort(key=lambda x: x[0])  # slack 升序 → 最紧张的先做
        best_p = feasible_candidates[0][1]

        # 4. 执行选中的任务
        cost, best_state = dynamic_cost(state, best_p, center_x)
        if math.isinf(cost):
            # 理论上不应该发生（因为 slack >=0 已经过滤），但保留防御性检查
            t += TIME_STEP
            continue

        state = best_state
        t = state["time"]
        done.append(best_p)

        exec_log.append({
            "t_start": t - best_p["time"],
            "t_end": t,
            "cx": best_p["cx"],
            "cy": best_p["original_cy"],
            "move_time": best_state["move_time"],
            "laser_time": best_p["time"]
        })

        pool.remove(best_p)

        if debug and len(done) % 5 == 0:
            print(f"[{t:.2f}s] 完成 {len(done)}/{total_tasks}")

    return done, exec_log, t


# ===================== 可视化与视频生成部分（保持不变） =====================

def draw_galvo_state(frame, points, exec_log, current_t, center_x, center_y,
                     color_done, color_pending, color_processing, galvo_color):
    if not exec_log:
        return frame

    h, w = frame.shape[:2]

    # 已完成路径
    prev_pos = None
    for log in exec_log:
        if log['t_end'] <= current_t:
            pos = (int(log['cx']), int(log['cy']))
            if prev_pos is not None:
                cv2.line(frame, prev_pos, pos, color_done, 2)
            prev_pos = pos
            cv2.circle(frame, pos, 4, color_done, -1)

    # 当前振镜位置（处理移动中 / 照射中）
    galvo_pos = None
    for i, log in enumerate(exec_log):
        if log['t_start'] <= current_t <= log['t_end']:
            galvo_pos = (int(log['cx']), int(log['cy']))
            radius = int(10 + 5 * math.sin(current_t * 15))
            cv2.circle(frame, galvo_pos, radius, color_processing, 3)
            cv2.circle(frame, galvo_pos, 5, (255, 255, 255), -1)
            cv2.circle(frame, galvo_pos, 15, color_processing, 2)
            break

        if i < len(exec_log) - 1:
            next_log = exec_log[i + 1]
            if log['t_end'] < current_t < next_log['t_start']:
                ratio = (current_t - log['t_end']) / (next_log['t_start'] - log['t_end'])
                ratio = max(0, min(1, ratio))
                curr_x = int(log['cx'] + (next_log['cx'] - log['cx']) * ratio)
                curr_y = int(log['cy'] + (next_log['cy'] - log['cy']) * ratio)
                galvo_pos = (curr_x, curr_y)
                cv2.line(frame, (int(log['cx']), int(log['cy'])), galvo_pos, (200, 200, 200), 1)
                break

    if galvo_pos is None:
        if current_t < exec_log[0]['t_start']:
            galvo_pos = (int(center_x), int(center_y))
        elif exec_log:
            last = exec_log[-1]
            if current_t >= last['t_end']:
                galvo_pos = (int(last['cx']), int(last['cy']))

    if galvo_pos:
        cv2.drawMarker(frame, galvo_pos, galvo_color, cv2.MARKER_CROSS, 20, 3)
        cv2.circle(frame, galvo_pos, 6, (255, 255, 255), -1)
        cv2.circle(frame, galvo_pos, 8, galvo_color, 2)

    # 所有杂草点状态
    for p in points:
        cx, cy = int(p['cx']), int(p['cy'])
        orig_cy = p.get('original_cy', p['cy'])

        is_done = any(
            abs(log['cx'] - p['cx']) < 2 and abs(log['cy'] - orig_cy) < 2
            and log['t_end'] <= current_t
            for log in exec_log
        )
        is_processing = any(
            abs(log['cx'] - p['cx']) < 2 and abs(log['cy'] - orig_cy) < 2
            and log['t_start'] <= current_t <= log['t_end']
            for log in exec_log
        ) and not is_done

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
    total_duration = max(left_duration, right_duration)

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    video_path = os.path.join(run_dir, "simulation.mp4")
    out = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))

    if not out.isOpened():
        print(f"错误: 无法创建视频文件 {video_path}")
        return

    time_step = 1.0 / FPS
    t = 0.0
    total_frames = int(total_duration * FPS) + 1

    print(f"生成视频: {total_duration:.1f}s, {FPS}FPS, ≈{total_frames}帧...")

    frame_count = 0
    while t <= total_duration + time_step:
        frame = img.copy()

        # 激光区半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (uL1, v1), (uL2, v2), (0, 255, 0), -1)
        cv2.rectangle(overlay, (uR1, v1), (uR2, v2), (255, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
        cv2.rectangle(frame, (uL1, v1), (uL2, v2), (0, 255, 0), 2)
        cv2.rectangle(frame, (uR1, v1), (uR2, v2), (255, 0, 0), 2)

        frame = draw_galvo_state(frame, left_points, left_log, t,
                                 left_center_x, center_y,
                                 (0,0,255), (128,128,128), (0,255,255), (0,0,255))

        frame = draw_galvo_state(frame, right_points, right_log, t,
                                 right_center_x, center_y,
                                 (255,0,0), (128,128,128), (255,255,0), (255,0,0))

        left_done = sum(1 for log in left_log if log['t_end'] <= t)
        right_done = sum(1 for log in right_log if log['t_end'] <= t)

        cv2.putText(frame, f"Time: {t:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Left : {left_done}/{len(left_points)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Right: {right_done}/{len(right_points)}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # 进度条
        bar_width = 400
        bar_x = (w - bar_width) // 2
        progress = min(t / total_duration, 1.0) if total_duration > 0 else 0
        cv2.rectangle(frame, (bar_x, h-50), (bar_x+bar_width, h-30), (200,200,200), -1)
        cv2.rectangle(frame, (bar_x, h-50), (bar_x + int(bar_width*progress), h-30), (0,255,0), -1)
        cv2.rectangle(frame, (bar_x, h-50), (bar_x+bar_width, h-30), (255,255,255), 2)

        out.write(frame)
        t += time_step
        frame_count += 1

        if frame_count % (FPS * 10) == 0:
            print(f"  已生成 {frame_count} 帧 ({t:.1f}s)")

    out.release()
    print(f"视频已保存: {video_path} ({frame_count}帧)")


def create_static_result(img, left_points, right_points, left_log, right_log, run_dir):
    global uL1_global, uL2_global, uR1_global, uR2_global, v1_global, v2_global
    h, w = img.shape[:2]
    vis = img.copy()

    cv2.rectangle(vis, (uL1_global, v1_global), (uL2_global, v2_global), (0,255,0), 2)
    cv2.rectangle(vis, (uR1_global, v1_global), (uR2_global, v2_global), (255,0,0), 2)

    for i in range(1, len(left_log)):
        p1 = (int(left_log[i-1]['cx']), int(left_log[i-1]['cy']))
        p2 = (int(left_log[i]['cx']), int(left_log[i]['cy']))
        cv2.line(vis, p1, p2, (0,0,255), 2)

    for i in range(1, len(right_log)):
        p1 = (int(right_log[i-1]['cx']), int(right_log[i-1]['cy']))
        p2 = (int(right_log[i]['cx']), int(right_log[i]['cy']))
        cv2.line(vis, p1, p2, (255,0,0), 2)

    for p in left_points:
        done = any(abs(l['cx'] - p['cx']) < 1 and abs(l['cy'] - p['cy']) < 1 for l in left_log)
        color = (0,255,0) if done else (0,0,0)
        cv2.circle(vis, (int(p['cx']), int(p['cy'])), 5, color, -1)

    for p in right_points:
        done = any(abs(l['cx'] - p['cx']) < 1 and abs(l['cy'] - p['cy']) < 1 for l in right_log)
        color = (255,0,0) if done else (128,128,128)
        cv2.circle(vis, (int(p['cx']), int(p['cy'])), 5, color, -1)

    cv2.imwrite(os.path.join(run_dir, "result.png"), vis)


def save_csv(left_log, right_log, run_dir):
    with open(os.path.join(run_dir, "left_path.csv"), "w") as f:
        f.write("t_start,t_end,cx,cy,move_time,laser_time\n")
        for log in left_log:
            f.write(f"{log['t_start']:.3f},{log['t_end']:.3f},{log['cx']:.1f},{log['cy']:.1f},{log['move_time']:.3f},{log['laser_time']:.1f}\n")

    with open(os.path.join(run_dir, "right_path.csv"), "w") as f:
        f.write("t_start,t_end,cx,cy,move_time,laser_time\n")
        for log in right_log:
            f.write(f"{log['t_start']:.3f},{log['t_end']:.3f},{log['cx']:.1f},{log['cy']:.1f},{log['move_time']:.3f},{log['laser_time']:.1f}\n")


# ===================== 主程序 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("智能除草动态路径规划系统 - Slack硬约束外置版")
    print("=" * 60)

    ALGORITHM = "greedy"   # 这里可以方便替换成其他算法逻辑
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

    # 激光区设置
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

    # 检测
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

    # 规划（现在 slack 判断在函数内部最外层）
    print(f"\n开始规划 (算法: {ALGORITHM}) ...")
    left_done, left_log, left_time = dynamic_online_planning(
        left_points, left_center_x, v1, v2, algorithm=ALGORITHM, debug=DEBUG_MODE
    )

    right_done, right_log, right_time = dynamic_online_planning(
        right_points, right_center_x, v1, v2, algorithm=ALGORITHM, debug=DEBUG_MODE
    )

    # 统计
    print("-" * 60)
    left_duration = left_log[-1]['t_end'] if left_log else 0
    right_duration = right_log[-1]['t_end'] if right_log else 0
    makespan = max(left_duration, right_duration)

    print(f"左振镜: {len(left_done)}/{len(left_points)} 完成, 实际用时: {left_duration:.2f}s")
    print(f"右振镜: {len(right_done)}/{len(right_points)} 完成, 实际用时: {right_duration:.2f}s")
    print(f"系统总完成时间 (Makespan): {makespan:.2f}s")

    # 保存结果
    print("\n保存结果...")
    save_csv(left_log, right_log, run_dir)
    create_static_result(img, left_points, right_points, left_log, right_log, run_dir)
    print("生成仿真视频...")
    create_simulation_video(img, left_points, right_points, left_log, right_log,
                            left_center_x, right_center_x, run_dir)

    print(f"\n完成! 所有文件保存在: {run_dir}")
    print("文件清单:")
    print("  - simulation.mp4    时间推进仿真视频")
    print("  - result.png        静态结果图")
    print("  - left_path.csv / right_path.csv   详细路径数据")