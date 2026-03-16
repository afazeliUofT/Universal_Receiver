from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .builders import extract_pilot_mask
from .compat import safe_call_variants
from .config import get_cfg
from .utils import broadcast_like_err, btfnc_to_tensor7, tensor7_to_btfnc


class PaperConfiguredReservoirEstimator(tf.keras.layers.Layer):
    """
    Paper-style configured reservoir / WESN-inspired channel estimator.

    This block keeps the same Sionna PUSCH receiver chain and only changes the
    channel-estimation front-end. It implements a fair, slot-local comparator to
    UPAIR-5G by:

    1) configuring a compact time-frequency basis from channel covariance,
    2) augmenting it with deterministic phase and FIR-style skip modes, and
    3) solving only the slot-local readout coefficients from the current DMRS.

    There is no offline gradient training in this block.
    """

    def __init__(
        self,
        base_estimator: Any,
        resource_grid: Any,
        cov_mat_time: tf.Tensor,
        cov_mat_freq: tf.Tensor,
        cfg: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(trainable=False, name="paper_configured_reservoir_estimator", **kwargs)
        self.base_estimator = base_estimator
        self.cfg = cfg
        self.eps = 1e-6

        self.paper_cfg = dict(get_cfg(cfg, "baselines.paper_configured_reservoir", {}) or {})
        self.time_rank = int(self.paper_cfg.get("time_rank", 6))
        self.freq_rank = int(self.paper_cfg.get("freq_rank", 12))
        self.freq_extra_dct_modes = int(self.paper_cfg.get("freq_extra_dct_modes", 4))
        self.time_extra_phase_slopes_rad = [float(x) for x in self.paper_cfg.get("time_extra_phase_slopes_rad", [-0.24, -0.12, 0.0, 0.12, 0.24])]
        self.extra_mode_prior_scale = float(self.paper_cfg.get("extra_mode_prior_scale", 0.2))
        self.min_prior_var = float(self.paper_cfg.get("min_prior_var", 1e-4))
        self.posterior_jitter = float(self.paper_cfg.get("posterior_jitter", 1e-5))
        self.pilot_weight = float(self.paper_cfg.get("pilot_weight", 1.0))
        self.ls_blend = float(self.paper_cfg.get("ls_blend", 0.10))
        self.err_inflation = float(self.paper_cfg.get("err_inflation", 1.0))

        pilot_mask = tf.cast(extract_pilot_mask(resource_grid), tf.float32)  # [T, F, 1]
        self.pilot_mask = pilot_mask
        self.num_symbols = int(pilot_mask.shape[0] or tf.shape(pilot_mask)[0].numpy())
        self.num_subcarriers = int(pilot_mask.shape[1] or tf.shape(pilot_mask)[1].numpy())

        pilot_mask_np = np.asarray(pilot_mask[..., 0].numpy(), dtype=np.float32)
        dmrs_per_symbol = pilot_mask_np.sum(axis=1)
        self.dmrs_symbol_index = int(np.argmax(dmrs_per_symbol)) if float(np.max(dmrs_per_symbol)) > 0.0 else self.num_symbols // 2

        basis_t, var_t = self._build_time_basis(np.asarray(tf.convert_to_tensor(cov_mat_time).numpy()))
        basis_f, var_f = self._build_freq_basis(np.asarray(tf.convert_to_tensor(cov_mat_freq).numpy()))

        self.basis_time = tf.constant(basis_t, dtype=tf.complex64)
        self.basis_freq = tf.constant(basis_f, dtype=tf.complex64)
        self.prior_var_time = tf.constant(var_t, dtype=tf.float32)
        self.prior_var_freq = tf.constant(var_f, dtype=tf.float32)

        phi_full = self._build_design_matrix(basis_t=basis_t, basis_f=basis_f)  # [T*F, K]
        pilot_indices = np.flatnonzero(pilot_mask_np.reshape(-1) > 0.5).astype(np.int32)
        if pilot_indices.size == 0:
            raise ValueError("Paper comparator could not find any DMRS pilot positions.")
        phi_p = phi_full[pilot_indices, :]

        prior_var_full = np.kron(var_f, var_t).astype(np.float32)
        prior_precision = 1.0 / np.maximum(prior_var_full, self.min_prior_var)

        gram = np.matmul(np.conjugate(phi_p).T, phi_p).astype(np.complex64)
        phi_p_h = np.conjugate(phi_p).T.astype(np.complex64)

        self.design_full = tf.constant(phi_full, dtype=tf.complex64)
        self.design_pilot_h = tf.constant(phi_p_h, dtype=tf.complex64)
        self.gram_pilot = tf.constant(gram, dtype=tf.complex64)
        self.prior_precision = tf.constant(np.diag(prior_precision).astype(np.complex64), dtype=tf.complex64)
        self.pilot_indices = tf.constant(pilot_indices, dtype=tf.int32)
        self.num_basis = int(phi_full.shape[1])
        self.kt = int(basis_t.shape[1])
        self.kf = int(basis_f.shape[1])

    @staticmethod
    def _normalize_columns(mat: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        if mat.size == 0:
            return mat.astype(np.complex64)
        norms = np.linalg.norm(mat, axis=0, keepdims=True)
        norms = np.maximum(norms, eps)
        return (mat / norms).astype(np.complex64)

    def _top_eig_basis(self, cov: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
        cov = np.asarray(cov, dtype=np.complex64)
        cov = 0.5 * (cov + cov.conj().T)
        dim = cov.shape[0]
        if dim == 0:
            return np.zeros((0, 0), dtype=np.complex64), np.zeros((0,), dtype=np.float32)
        rank = max(1, min(int(rank), dim))
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals.real)[::-1]
        eigvals = eigvals.real[order][:rank]
        eigvecs = eigvecs[:, order][:, :rank]
        eigvals = np.maximum(eigvals.astype(np.float32), self.min_prior_var)
        eigvecs = self._normalize_columns(eigvecs)
        return eigvecs, eigvals

    def _build_time_basis(self, cov_mat_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base, var = self._top_eig_basis(cov_mat_time, self.time_rank)
        centered_t = np.arange(self.num_symbols, dtype=np.float32) - float(self.dmrs_symbol_index)
        extra_cols: list[np.ndarray] = []
        for slope in self.time_extra_phase_slopes_rad:
            extra_cols.append(np.exp(1j * slope * centered_t).astype(np.complex64))
        if extra_cols:
            extra = np.stack(extra_cols, axis=1)
            extra = self._normalize_columns(extra)
            base = np.concatenate([base, extra], axis=1) if base.size else extra
            extra_var = np.full((extra.shape[1],), float(np.max(var) if var.size else 1.0) * self.extra_mode_prior_scale, dtype=np.float32)
            var = np.concatenate([var, np.maximum(extra_var, self.min_prior_var)], axis=0) if var.size else extra_var
        return self._normalize_columns(base), np.maximum(var.astype(np.float32), self.min_prior_var)

    def _build_freq_basis(self, cov_mat_freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base, var = self._top_eig_basis(cov_mat_freq, self.freq_rank)
        n = np.arange(self.num_subcarriers, dtype=np.float32)
        extra_cols: list[np.ndarray] = []
        num_extra = max(0, min(self.freq_extra_dct_modes, self.num_subcarriers))
        for k in range(num_extra):
            vec = np.cos(np.pi * (n + 0.5) * k / float(self.num_subcarriers)).astype(np.float32)
            extra_cols.append(vec.astype(np.complex64))
        if extra_cols:
            extra = np.stack(extra_cols, axis=1)
            extra = self._normalize_columns(extra)
            base = np.concatenate([base, extra], axis=1) if base.size else extra
            extra_var = np.full((extra.shape[1],), float(np.max(var) if var.size else 1.0) * self.extra_mode_prior_scale, dtype=np.float32)
            var = np.concatenate([var, np.maximum(extra_var, self.min_prior_var)], axis=0) if var.size else extra_var
        return self._normalize_columns(base), np.maximum(var.astype(np.float32), self.min_prior_var)

    @staticmethod
    def _build_design_matrix(basis_t: np.ndarray, basis_f: np.ndarray) -> np.ndarray:
        # H[t,f] = sum_{k,m} basis_t[t,k] * A[k,m] * conj(basis_f[f,m])
        phi = np.einsum("tk,fm->tfkm", basis_t, np.conjugate(basis_f), optimize=True)
        return phi.reshape(basis_t.shape[0] * basis_f.shape[0], -1).astype(np.complex64)

    def _parse_inputs(self, inputs: Any, *args: Any) -> tuple[tf.Tensor, tf.Tensor]:
        if isinstance(inputs, (tuple, list)):
            if len(inputs) < 2:
                raise ValueError("Expected at least y and no for paper configured reservoir estimator.")
            y, no = inputs[0], inputs[1]
        elif len(args) >= 1:
            y, no = inputs, args[0]
        else:
            raise ValueError("Could not parse estimator inputs.")
        return tf.convert_to_tensor(y), tf.convert_to_tensor(no)

    def _call_base_estimator(self, y: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        out = safe_call_variants(self.base_estimator, y, no)
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise ValueError("Base estimator must return (h_hat, err_var).")
        return tf.convert_to_tensor(out[0]), tf.convert_to_tensor(out[1])

    def _pilot_sigma2(self, err_ls: tf.Tensor, h_ls: tf.Tensor) -> tf.Tensor:
        err_btfnc = tensor7_to_btfnc(broadcast_like_err(err_ls, h_ls))  # [B, T, F, Nr]
        err_brtf = tf.transpose(tf.cast(err_btfnc, tf.float32), [0, 3, 1, 2])
        err_flat = tf.reshape(err_brtf, [tf.shape(err_brtf)[0], tf.shape(err_brtf)[1], -1])
        pilot_err = tf.gather(err_flat, self.pilot_indices, axis=-1)
        sigma2 = tf.reduce_mean(pilot_err, axis=-1)
        return tf.maximum(sigma2 * self.err_inflation, self.eps)  # [B, Nr]

    def _solve_coefficients(self, pilot_obs: tf.Tensor, sigma2: tf.Tensor) -> tf.Tensor:
        # pilot_obs: [B, Nr, Np], sigma2: [B, Nr]
        rhs_data = tf.einsum("kn,brn->brk", self.design_pilot_h, pilot_obs)  # [B, Nr, K]
        weight = tf.cast(self.pilot_weight / tf.maximum(sigma2, self.eps), tf.complex64)  # [B, Nr]
        weighted_gram = weight[..., tf.newaxis, tf.newaxis] * self.gram_pilot[tf.newaxis, tf.newaxis, :, :]
        system = weighted_gram + self.prior_precision[tf.newaxis, tf.newaxis, :, :]
        system = system + tf.cast(self.posterior_jitter, tf.complex64) * tf.eye(self.num_basis, dtype=tf.complex64)[tf.newaxis, tf.newaxis, :, :]
        rhs = weight[..., tf.newaxis] * rhs_data

        flat_system = tf.reshape(system, [-1, self.num_basis, self.num_basis])
        flat_rhs = tf.reshape(rhs, [-1, self.num_basis, 1])
        coeff = tf.linalg.solve(flat_system, flat_rhs)
        return tf.reshape(coeff[..., 0], [tf.shape(pilot_obs)[0], tf.shape(pilot_obs)[1], self.num_basis])

    def _reconstruct(self, coeff: tf.Tensor) -> tf.Tensor:
        # coeff: [B, Nr, K]
        h_flat = tf.einsum("nk,brk->brn", self.design_full, coeff)  # [B, Nr, T*F]
        return tf.reshape(h_flat, [tf.shape(coeff)[0], tf.shape(coeff)[1], self.num_symbols, self.num_subcarriers])

    def estimate_with_configured_reservoir(self, y: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        h_ls, err_ls = self._call_base_estimator(y, no)
        h_btfnc = tensor7_to_btfnc(h_ls)  # [B, T, F, Nr]
        h_brtf = tf.transpose(h_btfnc, [0, 3, 1, 2])  # [B, Nr, T, F]
        h_flat = tf.reshape(h_brtf, [tf.shape(h_brtf)[0], tf.shape(h_brtf)[1], -1])
        pilot_obs = tf.gather(h_flat, self.pilot_indices, axis=-1)  # [B, Nr, Np]

        sigma2 = self._pilot_sigma2(err_ls=err_ls, h_ls=h_ls)
        coeff = self._solve_coefficients(pilot_obs=pilot_obs, sigma2=sigma2)
        h_cfg = self._reconstruct(coeff)

        if self.ls_blend > 0.0:
            h_cfg = (1.0 - self.ls_blend) * h_cfg + self.ls_blend * h_brtf

        h_cfg_btfnc = tf.transpose(h_cfg, [0, 2, 3, 1])
        h_hat = btfnc_to_tensor7(tf.cast(h_cfg_btfnc, tf.complex64))

        pred_pilot = tf.gather(tf.reshape(h_cfg, [tf.shape(h_cfg)[0], tf.shape(h_cfg)[1], -1]), self.pilot_indices, axis=-1)
        pilot_mse = tf.reduce_mean(tf.square(tf.abs(pred_pilot - pilot_obs)), axis=-1)  # [B, Nr]
        total_err = sigma2 + tf.cast(pilot_mse, tf.float32)
        total_err = total_err[:, tf.newaxis, tf.newaxis, :]  # [B,1,1,Nr]
        total_err = tf.broadcast_to(total_err, [tf.shape(h_cfg_btfnc)[0], self.num_symbols, self.num_subcarriers, tf.shape(h_cfg_btfnc)[-1]])
        err_hat = btfnc_to_tensor7(tf.cast(total_err, tf.float32))

        return h_hat, err_hat, h_ls

    def save_basis_artifact(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            basis_time=np.asarray(self.basis_time.numpy()),
            basis_freq=np.asarray(self.basis_freq.numpy()),
            prior_var_time=np.asarray(self.prior_var_time.numpy()),
            prior_var_freq=np.asarray(self.prior_var_freq.numpy()),
            pilot_mask=np.asarray(self.pilot_mask.numpy()),
            pilot_indices=np.asarray(self.pilot_indices.numpy()),
            dmrs_symbol_index=np.asarray([self.dmrs_symbol_index], dtype=np.int32),
        )

    def call(self, inputs: Any, *args: Any, training: bool = False, **kwargs: Any) -> tuple[tf.Tensor, tf.Tensor]:
        del training, kwargs
        y, no = self._parse_inputs(inputs, *args)
        h_hat, err_hat, _ = self.estimate_with_configured_reservoir(y, no)
        return h_hat, err_hat
