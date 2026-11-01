import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


# 指定範囲でランダムな浮動小数を取得
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

# 環境
class Go2Env:
    # 初期化
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        # デバイス設定
        self.device = torch.device(device)

        # 環境パラメータの設定
        self.num_envs = num_envs  # 環境数
        self.num_obs = obs_cfg["num_obs"]  # 観測値の数
        self.num_privileged_obs = None  # 特権観測値 (未使用)
        self.num_actions = env_cfg["num_actions"]  # 行動の数
        self.num_commands = command_cfg["num_commands"]  # 命令の数

        self.simulate_action_latency = True  # 実ロボットでの1ステップ遅延をシミュレート
        self.dt = 0.02  # 制御周期 (50Hz)
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)  # エピソード長

        # 各種設定の保存
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]  # 観測値スケール
        self.reward_scales = reward_cfg["reward_scales"]  # 報酬スケール

        # シーンの作成
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=int(0.5 / self.dt),
                #max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=(0,)),
            rigid_options=gs.options.RigidOptions(
                #dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # 地面の追加
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # ロボットの追加
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)  # 初期位置
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)  # 初期回転
        self.inv_base_init_quat = inv_quat(self.base_init_quat)  # 初期回転の逆クォータニオン
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # scene.build() の前にカメラを追加して属性として保持
        self.cam = self.scene.add_camera(
            res=(640, 480),
            pos=(2.0, 0.0, 1.0),
            lookat=(0.0, 0.0, 0.5),
            fov=60,
        )

        # シーンのビルド
        self.scene.build(n_envs=num_envs)

        # 関節名からインデックスを取得
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PDコントローラのパラメータの設定
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # 報酬関数の準備と報酬スケールの時間ステップでの調整
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # バッファの初期化
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)  # 基本線速度
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)  # 基本角速度
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)  # 投影重力
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )  # 全体の重力
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)  # 観測バッファ
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)  # 報酬バッファ
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)  # リセットバッファ
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)  # エピソード長バッファ
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)  # 命令
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )  # 命令スケール
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)  # 行動
        self.last_actions = torch.zeros_like(self.actions)  # 前回の行動
        self.dof_pos = torch.zeros_like(self.actions)  # 関節位置
        self.dof_vel = torch.zeros_like(self.actions)  # 関節速度
        self.last_dof_vel = torch.zeros_like(self.actions)  # 前回の関節速度
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)  # 基本位置
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)  # 基本回転
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )  # デフォルト関節位置
        self.extras = dict()  # ロギング用の追加情報

    # 指定された環境インデックスで新しい命令を再サンプリング
    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    # ステップ
    def step(self, actions):
        # 行動を実行して環境を更新
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # バッファの更新
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # 命令を再サンプリング
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # 終了チェック
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        # タイムアウトの処理
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        # リセットが必要な環境をリセット
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # 報酬を計算
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            # 各報酬関数を実行し、スケールを適用
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # 観測値の計算
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        # 前回の行動と関節速度を更新
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # 観測の取得
    def get_observations(self):
        if "observations" not in self.extras:
            self.extras["observations"] = {}

        return self.obs_buf, self.extras

    # 特権的な観測の取得
    def get_privileged_observations(self):
        return None


    # 指定環境のリセット
    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # dofのリセット
        self.dof_pos[envs_idx] = self.default_dof_pos  # デフォルトの関節位置に設定
        self.dof_vel[envs_idx] = 0.0  # 関節速度をゼロにリセット
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # baseのリセット
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # バッファのリセット
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # 補助データの初期化
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            # 報酬関連のデータを初期化
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    # リセット
    def reset(self):
        # すべての環境をリセット
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # 直線速度命令の追従（xy軸）
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    # 角速度命令の追従（yaw）
    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    # z軸方向の線形速度のペナルティ化
    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    # 行動の変化率のペナルティ化
    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # 姿勢がデフォルトから大きく外れた場合のペナルティ化
    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    # ベース高さが目標値から外れた場合のペナルティ化
    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

