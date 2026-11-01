import argparse
import os
import pickle
import shutil
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

# 学習設定の取得
def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "num_steps_per_env": 24,       # 1イテレーションあたりの環境ステップ数
        "save_interval": 50,           # チェックポイント保存頻度
        "empirical_normalization": False,

        "algorithm": {  # アルゴリズムの設定
            "class_name" : "PPO",
            "clip_param": 0.2,  # クリッピングのパラメータ
            "desired_kl": 0.01,  # KLダイバージェンスの目標値
            "entropy_coef": 0.01,  # エントロピーの係数
            "gamma": 0.99,  # 割引率
            "lam": 0.95,  # GAEのパラメータ
            "learning_rate": 0.001,  # 学習率
            "max_grad_norm": 1.0,  # 勾配の最大値
            "num_learning_epochs": 5,  # エポック数
            "num_mini_batches": 4,  # ミニバッチの数
            "schedule": "adaptive",  # スケジュール方式
            "use_clipped_value_loss": True,  # 価値関数損失のクリッピングを使用するか
            "value_loss_coef": 1.0,  # 価値関数の損失係数
        },
        "init_member_classes": {},  # メンバークラスの初期化設定
        "policy": {  # ポリシーの設定
            "class_name" : "ActorCritic",
            "activation": "elu",  # 活性化関数
            "actor_hidden_dims": [512, 256, 128],  # Actorネットワークの隠れ層サイズ
            "critic_hidden_dims": [512, 256, 128],  # Criticネットワークの隠れ層サイズ
            "init_noise_std": 1.0,  # 初期ノイズ標準偏差
        },
        "runner": {  # ランナーの設定
            "algorithm_class_name": "PPO",  # 使用するアルゴリズム名
            "checkpoint": -1,  # チェックポイントのインデックス
            "experiment_name": exp_name,  # 実験名
            "load_run": -1,  # 実行のロード設定
            "log_interval": 1,  # ログの間隔
            "max_iterations": max_iterations,  # 最大反復回数
            "num_steps_per_env": 24,  # 環境ごとのステップ数
            "policy_class_name": "ActorCritic",  # ポリシークラス名
            "record_interval": -1,  # 記録間隔
            "resume": False,  # 再開設定
            "resume_path": None,  # 再開パス
            "run_name": "",  # 実行名
            "runner_class_name": "runner_class_name",  # ランナークラス名
            "save_interval": 100,  # 保存間隔
        },
        "runner_class_name": "OnPolicyRunner",  # ランナーのクラス名
        "seed": 1,  # ランダムシード
    }

    return train_cfg_dict


# 設定の取得
def get_cfgs():
    env_cfg = {
        "num_actions": 12,  # 行動の数
        "default_joint_angles": {  # 各関節のデフォルト角度 [ラジアン]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [  # 自由度（Degree of Freedom）の名前
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "kp": 20.0,  # 比例ゲイン
        "kd": 0.5,  # 微分ゲイン
        "termination_if_roll_greater_than": 10,  # 転倒判定（ロール角度）
        "termination_if_pitch_greater_than": 10,  # 転倒判定（ピッチ角度）
        "base_init_pos": [0.0, 0.0, 0.42],  # ベースの初期位置
        "base_init_quat": [0.0, 0.0, 0.0, 1.0],  # ベースの初期クォータニオン
        "episode_length_s": 20.0,  # エピソードの長さ（秒）
        "resampling_time_s": 4.0,  # サンプリング間隔（秒）
        "action_scale": 0.25,  # アクションのスケール
        "simulate_action_latency": True,  # アクション遅延をシミュレートするか
        "clip_actions": 100.0,  # アクションのクリッピング
    }
    obs_cfg = {
        "num_obs": 45,  # 観測値の数
        "obs_scales": {  # 各観測値のスケール
            "lin_vel": 2.0,  # 線速度
            "ang_vel": 0.25,  # 角速度
            "dof_pos": 1.0,  # 自由度の位置
            "dof_vel": 0.05,  # 自由度の速度
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,  # トラッキング誤差の標準偏差
        "base_height_target": 0.3,  # ベースの目標高さ
        "feet_height_target": 0.075,  # 足の目標高さ
        "reward_scales": {  # 報酬のスケール
            "tracking_lin_vel": 1.0,  # 線速度の追跡
            "tracking_ang_vel": 0.2,  # 角速度の追跡
            "lin_vel_z": -1.0,  # Z方向の速度
            "base_height": -50.0,  # ベースの高さ
            "action_rate": -0.005,  # アクション変化率
            "similar_to_default": -0.1,  # デフォルトへの類似性
        },
    }
    command_cfg = {
        "num_commands": 3,  # 命令の数
        "lin_vel_x_range": [0.5, 0.5],  # X方向の線速度範囲
        "lin_vel_y_range": [0, 0],  # Y方向の線速度範囲
        "ang_vel_range": [0, 0],  # 角速度範囲
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

# メイン
def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=100)
    args = parser.parse_args()

    # Genesisの初期化
    gs.init(logging_level="warning")

    # 各種設定の取得
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # ログディレクトリが存在する場合は削除し再作成
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 環境の初期化
    env = Go2Env(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg
    )

    # ランナーの初期化
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    # 設定の保存 (pickle)
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # 学習の開始
    runner.learn(
        num_learning_iterations=args.max_iterations, 
        init_at_random_ep_len=True
    )

if __name__ == "__main__":
    main()