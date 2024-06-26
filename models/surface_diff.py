import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm
import sys
sys.path.append('/home/ligl/project/DVAE/generation1')
sys.path.append('/home/guanlueli/project/DVAE/generation1')
from models.common import compose_context, ShiftedSoftplus
from models.EGNN import ATT_EGNN
# from torchph.pershom import vr_persistence_l1
from torch_geometric.nn import radius_graph, knn_graph

from ep_ab.models.dmasif import dMaSIF

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def get_refine_net(config):

    refine_net = ATT_EGNN(
        num_blocks=config.num_blocks,
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        k=config.knn,
        edge_feat_dim=config.edge_feat_dim,
        num_r_gaussian=config.num_r_gaussian,
        num_node_types=config.num_node_types,
        act_fn=config.act_fn,
        norm=config.norm,
        cutoff_mode=config.cutoff_mode,
        ew_net_type=config.ew_net_type,
        num_x2h=config.num_x2h,
        num_h2x=config.num_h2x,
        r_max=config.r_max,
        x2h_out_fc=config.x2h_out_fc,
        sync_twoup=config.sync_twoup,
        x_to_x = config.x_to_x
    )

    return refine_net


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def center_pos(protein_pos, ligand_pos, motif_pos, batch_protein, batch_ligand, batch_motif, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
        motif_pos = motif_pos - offset[batch_motif]
    elif mode == 'ligand':
        offset = scatter_mean(ligand_pos, batch_ligand, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
        motif_pos = motif_pos - offset[batch_motif]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, motif_pos, offset

# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


# %% categorical diffusion related
def index_to_log_onehot_v(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    return sample_index


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


# %%


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Model
class surface_diff(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config

        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight
        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']
        self.guidance = config.guidance

        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # atom type diffusion schedule in log space
        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
            # print('cosine v alpha schedule applied!')
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # model definition
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        self.num_classes_motif = 520
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 2
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        if self.config.protein_ph_feature:
            self.protein_ph_emb = nn.Sequential(
                nn.Linear(3, self.config.ph_dim * 2),
                nn.GELU(),
                nn.Linear(self.config.ph_dim * 2, self.config.ph_dim)
            )
            self.protein_atom_emb = nn.Linear(protein_atom_feature_dim + self.config.ph_dim, emb_dim)
        else:
            self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein', 'ligand]

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                if self.config.ligand_ph_feature:
                    self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.config.ph_dim + 1, emb_dim)
                    self.ligand_ph_emb = nn.Sequential(
                        nn.Linear(3, self.config.ph_dim * 2),
                        nn.GELU(),
                        nn.Linear(self.config.ph_dim * 2, self.config.ph_dim)
                    )
                    self.motif_atom_emb = nn.Linear(520 + 1, emb_dim)
                else:
                    self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
                    self.motif_atom_emb = nn.Linear(520 + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim+ self.model.ph_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            if  self.config.ligand_ph_feature:
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.config.ph_dim, emb_dim)
                self.ligand_ph_emb = nn.Sequential(
                    nn.Linear(3, self.config.ph_dim * 2),
                    nn.GELU(),
                    nn.Linear(self.config.ph_dim * 2, self.config.ph_dim)
                )

                self.motif_atom_emb = nn.Linear(520, emb_dim)
            else:
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)
                self.motif_atom_emb = nn.Linear(520, emb_dim)

        self.refine_net = get_refine_net(config)
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )

        self.v_inference_motif = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, 520),
        )

        self.surface_encoder = dMaSIF(config.masif)

    def forward(self, P_receptor, P_ligand, time_step=None, return_all=False, fix_x=False):

        # time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([init_ligand_v,(time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)], -1)
                input_motif_feat = torch.cat([init_motif_v,(time_step / self.num_timesteps)[batch_motif].unsqueeze(-1)], -1)

            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v
            input_motif_feat = init_motif_v

        patch_feat_receptor, patch_idx_receptor, patch_xyz_receptor, patch_mask_receptor, _, _ = self.point_forward(P_receptor)

        patch_feat_ligand, _, _, patch_mask_ligand, _, _ = self.point_forward(P_ligand)

        # Transformer
        patch_feat_receptor = self.patch_forward(patch_feat_receptor, patch_xyz_receptor, patch_feat_=patch_feat_ligand,
                                                 patch_mask=patch_mask_receptor.unsqueeze(1).unsqueeze(
                                                     2) if patch_mask_receptor is not None else None,
                                                 patch_mask_=patch_mask_ligand.unsqueeze(1).unsqueeze(
                                                     2) if patch_mask_ligand is not None else None)
        pred_coarse = self.classifier(patch_feat_receptor).squeeze(-1)

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        init_motif_h = self.motif_atom_emb(input_motif_feat)

        if self.config.node_indicator:

            bit_mask = torch.tensor([[0, 0], [0, 1], [1, 0]])

            h_protein = torch.cat([h_protein, bit_mask[0].expand(len(h_protein), -1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, bit_mask[1].expand(len(init_ligand_h), -1).to(h_protein)], -1)
            init_motif_h = torch.cat([init_motif_h, bit_mask[2].expand(len(init_motif_h), -1).to(h_protein)], -1)

        h_all, pos_all, batch_all, mask_ligand, mask_pocket, mask_motif, mask_protein = compose_context(


        )

        outputs = self.refine_net(h_all, pos_all, mask_ligand, mask_pocket, mask_motif, mask_protein, batch_all, return_all=return_all, fix_x=fix_x)

        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_motif_pos, final_motif_h = final_pos[mask_motif], final_h[mask_motif]
        # todo add pos
        final_ligand_v = self.v_inference(final_ligand_h)
        final_motif_v = self.v_inference_motif(final_motif_h)

        preds = {


        }

        return preds

    def point_forward(self, P, ):
        feat = self.surface_encoder(P)  # feat: (N, D)
        xyz_dense, mask = to_dense_batch(P['xyz'], P['batch'], fill_value=0.0)  # xyz_dense: (B, L, 3), mask: (B, L)
        feat_dense, _ = to_dense_batch(feat, P['batch'], fill_value=0.0)  # feat_dense: (B, L, D)
        if self.fixed_patches:
            patch_mask = None
            patch_idx = farthest_point_sample(xyz_dense, self.n_patches, mask)  # patch_idx: (B, npatch)
            patch_xyz = index_points(xyz_dense, patch_idx)  # patch_xyz: (B, npatch, 3)
        else:
            # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.fps.html
            patch_idx = fps(P['xyz'], P['batch'], ratio=self.patch_ratio)  # patch_idx: (B x L)
            patch_xyz, patch_mask = to_dense_batch(P['xyz'][patch_idx], P['batch'][patch_idx],
                                                   fill_value=0.0)  # patch_xyz: (B, npatch, 3), patch_mask: (B, npatch)

        dists = square_distance(xyz_dense, patch_xyz)  # dists: (B, L, npatch)
        dists[~mask] = 1e10
        _, group_idx = dists.transpose(1, 2).contiguous().topk(self.n_pts_per_patch,
                                                               largest=False)  # group_idx: (B, npatch, n_pts)
        grouped_feat = index_points(feat_dense, group_idx)  # grouped_feat: (B, npatch, n_pts, D)
        patch_feat = grouped_feat.max(-2)[0]  # patch_feat: (B, npatch, D)
        return patch_feat, patch_idx, patch_xyz, patch_mask, xyz_dense, group_idx


    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch, num_classes):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch, num_classes):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(num_classes)
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch, num_classes):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch, num_classes)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, num_classes)
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch, num_classes):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch, num_classes)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch, num_classes)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch, num_classes):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch, num_classes = num_classes)
        log_half_prob = -torch.log(num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)
        pos_model_mean = a_pos.sqrt() * pos0
        pos_log_variance = torch.log((1.0 - a_pos).sqrt())
        kl_prior = normal_kl(torch.zeros_like(pos_model_mean), torch.zeros_like(pos_log_variance),
                             pos_model_mean, pos_log_variance)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    def compute_ph_Lt(self, predict, target, batch):
        num_graphs = batch.max().item() + 1
        loss_ph_all = 0
        for i in range(num_graphs):
            X_pre = predict[batch==i]
            X_tar = target[batch==i]

            pre_ph = vr_persistence_l1(X_pre, 1, 0)
            tar_ph = vr_persistence_l1(X_tar, 1, 0)
            pre_ph_H0 = pre_ph[0][0][:,1]
            tar_ph_H0 = tar_ph[0][0][:,1]

            pre_ph_H1 = pre_ph[1][1]
            tar_ph_H1 = tar_ph[1][1]

            # print(pre_ph[0][0].shape, tar_ph[0][0].shape, pre_ph[0][1].shape, tar_ph[0][1].shape)

            loss_ph0 = torch.sum(torch.abs(pre_ph_H0 - tar_ph_H0))
            loss_ph1 = torch.sum(torch.abs(pre_ph_H1 - tar_ph_H1))

            # print(loss_ph)
            loss_ph_all = loss_ph_all + loss_ph0 + loss_ph1 * 0.01
        loss_ph_all = loss_ph_all / num_graphs
        # print(loss_ph_all)
        return  loss_ph_all

    def get_diffusion_loss(self, batch, time_step=None,):

        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, motif_pos, offset = center_pos(protein_pos, ligand_pos, motif_pos, batch_protein, batch_ligand, batch_motif, mode=self.center_pos_mode)

        # receptor
        if self.fixed_patches:
            assert batch['min_num_nodes'] >= self.n_patches, f'Surface has {batch["min_num_nodes"]} points than {self.n_patches}, please decrease the patch number.'
        if self.cfg.masif.resolution == 'atom':
            P_receptor = {'atomxyz': batch['atomxyz_receptor'], 'atomtypes': batch['atomtypes_receptor'], 'batch_atom': batch['atomxyz_receptor_batch'],
                          'xyz': batch['xyz_receptor'], 'normals': batch['normals_receptor'], 'batch': batch['xyz_receptor_batch']}
        else:
            P_receptor = {'resxyz': batch['resxyz_receptor'], 'restypes': batch['restypes_receptor'], 'batch_res': batch['resxyz_receptor_batch'], 'xyz': batch['xyz_receptor'],
                          'normals': batch['normals_receptor'], 'batch': batch['xyz_receptor_batch']}

        # ligand
        if self.cfg.masif.resolution == 'atom':
            P_ligand = {'atomxyz': batch['atomxyz_ligand'], 'atomtypes': batch['atomtypes_ligand'],
                        'batch_atom': batch['atomxyz_ligand_batch'], 'xyz': batch['xyz_ligand'],
                        'normals': batch['normals_ligand'], 'batch': batch['xyz_ligand_batch']}
        else:
            P_ligand = {'resxyz': batch['resxyz_ligand'], 'restypes': batch['restypes_ligand'],
                        'batch_res': batch['resxyz_ligand_batch'], 'xyz': batch['xyz_ligand'],
                        'normals': batch['normals_ligand'], 'batch': batch['xyz_ligand_batch']}

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps

        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand, self.num_classes)

        # 3. perturb motif pos and v
        a_pos_motif = a[batch_motif].unsqueeze(-1)  # (num_ligand_atoms, 1)
        if self.config.perturb_motif_pos == True:
            pos_noise_motif = torch.zeros_like(motif_pos)
            pos_noise_motif.normal_()
            # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
            init_motif_pos = a_pos_motif.sqrt() * motif_pos + (1.0 - a_pos_motif).sqrt() * pos_noise_motif  # pos_noise * std
        else:
            init_motif_pos =  motif_pos   # pos_noise * std
        if self.config.perturb_motif_wid == True:
            log_motif_v0 = index_to_log_onehot(motif_wid, self.num_classes_motif)
            init_motif_wid, log_motif_vt = self.q_v_sample(log_motif_v0, time_step, batch_motif, self.num_classes_motif)
        else:
            init_motif_wid = motif_wid

        preds = self(
            P_receptor = P_receptor,
            P_ligand = P_ligand,
            time_step=time_step,
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed
        # atom position
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        loss, loss_pos, loss_v = 0, 0, 0, 0, 0
        # atom pos loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos = torch.mean(loss_pos)

        loss = loss_pos

        # atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand, self.num_classes)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand, self.num_classes)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)

        loss  = loss + loss_v * self.loss_v_weight

        # Persist Homology loss


        return {
            'loss_pos': loss_pos,
            'loss_pos_motif': loss_pos_motif,
            'loss_v': loss_v,
            'loss_v_motif': loss_v_motif,
            'loss_ph': loss_ph,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1)
        }

    @torch.no_grad()
    def sample_diffusion(self, protein_pos, protein_v, protein_ph, batch_protein, batch_is_in_pocket,
                         init_ligand_pos, init_ligand_v,ligand_ph, batch_ligand,  init_motif_pos, init_motif_wid, batch_motif,
                         num_steps=None, center_pos_mode=None, pos_only=False):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, init_motif_pos, offset = center_pos(protein_pos, init_ligand_pos, init_motif_pos, batch_protein, batch_ligand, batch_motif, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        pos_traj_motif, v_traj_motif = [], []
        v0_pred_traj_motif, vt_pred_traj_motif = [], []

        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        motif_pos, motif_wid = init_motif_pos, init_motif_wid
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)

            preds = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                protein_ph=protein_ph,
                batch_protein=batch_protein,
                batch_is_in_pocket=batch_is_in_pocket,

                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                ligand_ph=ligand_ph,
                batch_ligand=batch_ligand,

                init_motif_wid = motif_wid,
                init_motif_pos = motif_pos,
                batch_motif = batch_motif,

                time_step=t
            )

            if self.guidance >= 1:

                is_in_pocket = torch.zeros_like(batch_is_in_pocket)
                batch_is_in_pocket
                preds_uncondi = self(
                    protein_pos=protein_pos,
                    protein_v=protein_v,
                    protein_ph=protein_ph,
                    batch_protein=batch_protein,
                    batch_is_in_pocket=batch_is_in_pocket,

                    init_ligand_pos=ligand_pos,
                    init_ligand_v=ligand_v,
                    ligand_ph=ligand_ph,
                    batch_ligand=batch_ligand,

                    init_motif_wid=motif_wid,
                    init_motif_pos=motif_pos,
                    batch_motif=batch_motif,

                    time_step=t
                )

            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']

                pred_pos_noise_m = preds['pred_motif_pos'] - motif_pos
                pos0_from_e_motif = self._predict_x0_from_eps(xt=motif_pos, eps=pred_pos_noise_m, t=t, batch=batch_motif)
                v0_from_e_motif = preds['pred_motif_v']

            elif self.model_mean_type == 'C0':

                if self.guidance >= 1:
                    pos0_from_e = (1+ self.guidance) * preds['pred_ligand_pos'] - self.guidance * preds_uncondi['pred_ligand_pos']
                    v0_from_e =  preds['pred_ligand_v']

                    pos0_from_e_motif = (1+ self.guidance) * preds['pred_motif_pos'] - self.guidance * preds_uncondi['pred_motif_pos']
                    v0_from_e_motif = preds['pred_motif_v']

                else:
                    pos0_from_e = preds['pred_ligand_pos']
                    v0_from_e = preds['pred_ligand_v']

                    pos0_from_e_motif = preds['pred_motif_pos']
                    v0_from_e_motif = preds['pred_motif_v']
            else:
                raise ValueError

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
            ligand_pos = ligand_pos_next

            if not pos_only:
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand, self.num_classes)
                ligand_v_next = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

            # motif
            pos_model_mean_motif = self.q_pos_posterior(x0=pos0_from_e_motif, xt=motif_pos, t=t, batch=batch_motif)
            pos_log_variance_motif = extract(self.posterior_logvar, t, batch_motif)
            # no noise when t == 0
            nonzero_mask_motif = (1 - (t == 0).float())[batch_motif].unsqueeze(-1)
            motif_pos_next = pos_model_mean_motif + nonzero_mask_motif * (0.5 * pos_log_variance_motif).exp() * torch.randn_like(
                motif_pos)
            motif_pos = motif_pos_next

            if not pos_only:
                log_motif_v_recon = F.log_softmax(v0_from_e_motif, dim=-1)
                log_motif_v = index_to_log_onehot(motif_wid, self.num_classes_motif)
                log_model_prob_motif = self.q_v_posterior(log_motif_v_recon, log_motif_v, t, batch_motif, self.num_classes_motif)
                motif_v_next = log_sample_categorical(log_model_prob_motif)

                v0_pred_traj_motif.append(log_motif_v_recon.clone().cpu())
                vt_pred_traj_motif.append(log_model_prob_motif.clone().cpu())
                motif_v = motif_v_next

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

            ori_motif_pos = motif_pos + offset[batch_motif]
            pos_traj_motif.append(ori_motif_pos.clone().cpu())
            v_traj_motif.append(motif_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        motif_pos = motif_pos + offset[batch_motif]

        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'pos_motif': motif_pos,
            'v_motif': motif_v,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj,

            'pos_traj_motif': pos_traj_motif,
            'v_traj_motif': v_traj_motif,
            'v0_traj_motif': v0_pred_traj_motif,
            'vt_traj_motif': vt_pred_traj_motif
        }


def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)
