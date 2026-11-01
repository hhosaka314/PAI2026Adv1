import argparse
import os
import pickle
import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

# メイン
def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    # Genesisの初期化
    gs.init()

    # 各種設定の取得
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # rsl_rl 2.x 用の必須キーを追加
    if "policy" not in train_cfg:
        train_cfg["policy"] = {}
    train_cfg["policy"]["class_name"] = "ActorCritic"

    if "algorithm" not in train_cfg:
        train_cfg["algorithm"] = {}

    train_cfg["algorithm"]["class_name"] = "PPO"

    train_cfg["num_steps_per_env"] = train_cfg.get("runner", {}).get("num_steps_per_env", 24)
    train_cfg["save_interval"] = train_cfg.get("runner", {}).get("save_interval", 50)

    # 環境の初期化
    env = Go2Env(
        num_envs=1,  # 環境数を指定
        env_cfg=env_cfg,  # 環境設定
        obs_cfg=obs_cfg,  # 観測設定
        reward_cfg=reward_cfg,  # 報酬設定
        command_cfg=command_cfg,  # 命令設定
        show_viewer=False,  # ビューアを表示するかどうか
    )

    # OnPolicyRunnerの初期化
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    # モデルの読み込み
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    # 指定されたファイルが存在しない場合、最新の .pt ファイルを自動探査
    if not os.path.exists(resume_path):
        pt_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        
        # 数字部分を取り出して最新（最大値）のものを選択
        latest_file = sorted(pt_files, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
        resume_path = os.path.join(log_dir, latest_file)
        print(f"指定されたモデルが見つからないため、最新のモデルをロードします: {resume_path}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    # 環境のリセット
    obs, _ = env.reset()

    # カメラの追加と録画開始
    cam = env.cam
    cam.start_recording()

    # 推論ループ
    with torch.no_grad():
        for i in range(3150):
        #while True:
            actions = policy(obs)  # ポリシーで行動を計算
            obs, rews, dones, infos = env.step(actions)  # 環境を1ステップ進行

            # ロボットの位置を取得し、カメラの位置と視点を動的に追従させる
            robot_pos = env.robot.get_pos().cpu().numpy()[0]  # ロボットの現在位置 [x, y, z]
            cam.set_pose(
                pos=(robot_pos[0] + 2.0, robot_pos[1] + 1.5, robot_pos[2] + 0.8), # カメラの位置
                lookat=(robot_pos[0], robot_pos[1], robot_pos[2] + 0.2),         # 注視点（ロボット）
            )

    # 動画の保存
    cam.stop_recording(save_to_filename="go2_walking.mp4", fps=50)
    print("動画保存完了: go2_walking.mp4")

if __name__ == "__main__":
    main()
