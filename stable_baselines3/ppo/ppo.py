import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param use_psm: Whether to use PSM method to improve generalization (Need a pretrained model)
        default: False
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        use_psm: bool = False,
        psm_coef: float = 1.0,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.use_psm = use_psm
        self.psm_coef = psm_coef
        if use_psm:
            self.shuffle = False
        else:
            self.shuffle = True

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        cme_losses = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size, shuffle=self.shuffle):
                actions = rollout_data.actions
                ep_starts = rollout_data.episode_starts
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Get representation for calculating CME Loss
                with th.no_grad():
                    features = self.policy.extract_features(rollout_data.observations)
                    if self.policy.share_features_extractor:
                        latent_pi, _ = self.policy.mlp_extractor(features)
                    else:
                        pi_features, _ = features
                        latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
                if hasattr(self, 'cme_projector'):
                    z = self.cme_projector(latent_pi)  # shape: [2B, k]
                else:
                    z = latent_pi

                # Normalize embeddings for cosine similarity
                z = F.normalize(z, p=2, dim=1)
                # x Samples and y Samples, for contrastive
                spliter = int(self.batch_size/2)
                z_x = z[ : spliter]
                z_y = z[spliter : ]

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # === CME Loss (PSE) ===
                cme_loss = th.tensor(0.0, device=self.device)
                if self.use_psm:
                    assert z_x.shape[0] == z_y.shape[0]
                    B = z_x.shape[0]
                    if B > 1:
                        # Step 1: Compute cosine similarity matrix [B, B]
                        sim_matrix = th.matmul(z_x, z_y.T)  # [B, B]

                        # Step 2: Get PSM-based similarity Γ (shape [B, B])
                        # NOTE: In practice, Γ may come from precomputed PSM on expert trajectories.
                        # For online approximation, one could use current policy to estimate PSM,
                        # but here we assume a method `self.get_psm_sim(observations)` exists.
                        gamma_matrix = self.compute_psm_gamma_matrix(z_x, z_y, ep_starts)  # [B, B], values in (0,1]

                        # Step 3: For each anchor y (row i), find positive x̃_y = argmax_j Γ[i, j]
                        # We follow Eq.(4) in paper: for each y (row i), positive is j* = argmax_j Γ[i,j]
                        pos_idx = th.argmax(gamma_matrix, dim=1)  # [B]
                        pos_mask = th.zeros_like(gamma_matrix, dtype=th.bool)
                        pos_mask[th.arange(B), pos_idx] = True

                        # Numerator: Γ(x̃_y, y) * exp(λ s(x̃_y, y))
                        lambda_temp = 1.0  # or obtain from self.cme_temp
                        exp_sim = th.exp(lambda_temp * sim_matrix)
                        pos_sim = sim_matrix[th.arange(B), pos_idx]  # [B]
                        pos_gamma = gamma_matrix[th.arange(B), pos_idx]  # [B]
                        numerator = pos_gamma * th.exp(lambda_temp * pos_sim)

                        # Denominator: numerator + sum_{x' ≠ x̃_y} (1 - Γ(x', y)) * exp(λ s(x', y))
                        neg_weights = (1.0 - gamma_matrix)
                        neg_weights[pos_mask] = 0.0  # exclude positive
                        denominator = numerator + (neg_weights * exp_sim).sum(dim=1)
                        cme_loss_i = -th.log(numerator / (denominator + 1e-8))  # [B]
                        cme_loss = cme_loss_i.mean()

                        cme_losses.append(cme_loss.item())
                    else:
                        cme_losses.append(0.0)
                else:
                    cme_losses.append(0.0)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if self.use_psm:
                    loss += self.psm_coef * cme_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        if self.use_psm and cme_losses:
            self.logger.record("train/cme_loss", np.mean(cme_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        deterministic: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            deterministic=deterministic,
        )
    
    def compute_psm_gamma_matrix(self,
        latent_pi_x: th.Tensor,
        latent_pi_y: th.Tensor,
        episode_starts: th.Tensor,
        gamma: float = 0.99,
        beta: float = 0.1,
        ) -> th.Tensor:
        """
        Reconstruct episodes from the batch data in the rollout buffer and compute the
        PSM similarity matrix Γ.

        Args:
            latent_pi_x: [B, ...]
            latent_pi_y: [B, ...]
            episode_starts: [B] bool, True indicates the start of a new episode
            gamma: PSM discount factor
            beta: Γ = exp(-d / beta)
            device: computation device

        Returns:
            gamma_matrix: [B, B] similarity matrix
        """
            # B = obs.shape[0]
            # if B == 0:
            #     return th.empty(0, 0, device=device)

        with th.no_grad():
            # Step 1: Get the policy distribution
            dist = self.policy._get_action_dist_from_latent(th.vstack((latent_pi_x, latent_pi_y)))
            B = latent_pi_x.shape[0]
            if B == 0:
                return th.empty(0, 0, device=self.device)

            # Step 2: Group indices by episode
            episode_indices: list[tuple[int, int]] = []
            start_indices = th.where(episode_starts)[0].cpu().numpy().tolist()
            if len(start_indices) == 0:
                start_indices = [0]
            start_indices.append(2*B)

            for i in range(len(start_indices) - 1):
                s, e = start_indices[i], start_indices[i + 1]
                if e > s:
                    episode_indices.append((s, e))

            num_episodes = len(episode_indices)
            if num_episodes == 0:
                return th.ones(B, B, device=self.device)  # fallback

            # Step 3: Initialize the full zero distance matrix
            d_full = np.zeros((B, B), dtype=np.float32)

            # Step 4: Compute PSM for each pair of episodes
            for i in range(int(num_episodes / 2)):
                i_start, i_end = episode_indices[i]
                T_i = i_end - i_start
                # dist_i = dist[i_start:i_end]

                for j in range(int(num_episodes / 2), num_episodes):
                    j_start, j_end = episode_indices[j]
                    T_j = j_end - j_start
                    # dist_j = dist[j_start:j_end]

                    # Construct the COST matrix = DIST(π(x), π(y))
                    if hasattr(dist.distribution, 'probs'):
                        probs_i = dist.distribution.probs.cpu()[i_start:i_end].numpy()  # [T_i, A]
                        probs_j = dist.distribution.probs.cpu()[j_start:j_end].numpy()  # [T_j, A]
                        # Total Variation: 0.5 * L1
                        cost = 0.5 * np.abs(probs_i[:, None, :] - probs_j[None, :, :]).sum(axis=2)  # [T_i, T_j]
                    else:
                        mean_i = dist.distribution.mean.cpu()[i_start:i_end].numpy()  # [T_i, A]
                        mean_j = dist.distribution.mean.cpu()[j_start:j_end].numpy()  # [T_j, A]
                        cost = np.abs(mean_i[:, None, :] - mean_j[None, :, :]).sum(axis=2)  # [T_i, T_j]

                    # Compute the PSM distances via dynamic programming
                    d_pair = self.metric_fixed_point(cost, gamma=gamma)  # [T_i, T_j]

                    # Fill the distances into the global matrix
                    j_start = int(j_start - B)
                    j_end = int(j_end - B)
                    d_full[i_start:i_end, j_start:j_end] = d_pair

            # Step 5: Convert distances to similarity Γ = exp(-d / β)
            gamma_matrix = np.exp(-d_full / beta)
            return th.from_numpy(gamma_matrix).to(self.device)
        
    def metric_fixed_point(self, cost_matrix: np.ndarray, gamma: float = 0.99, eps: float = 1e-5) -> np.ndarray:
        """
        Dynamic programming implementation of J.1: compute the PSM distance matrix
        for deterministic environments.

        Args:
            cost_matrix: [T1, T2], DIST(π(x_i), π(y_j))
            gamma: discount factor
            eps: convergence threshold
        Returns:
            d_matrix: [T1, T2], PSM distances
        """
        d = np.zeros_like(cost_matrix)
        T1, T2 = cost_matrix.shape

        while True:
            d_new = cost_matrix.copy()
            # d[i, j] = cost[i, j] + gamma * d[i+1, j+1]
            if T1 > 1 and T2 > 1:
                d_new[:-1, :-1] += gamma * d[1:, 1:]
            if T1 > 1:
                d_new[:-1, -1] += gamma * d[1:, -1]  # y termination
            if T2 > 1:
                d_new[-1, :-1] += gamma * d[-1, 1:]  # x termination
            # Terminal pair: d[T1-1, T2-1] = cost[T1-1, T2-1] (already set)

            if np.sum(np.abs(d - d_new)) < eps:
                break
            d = d_new
        return d
