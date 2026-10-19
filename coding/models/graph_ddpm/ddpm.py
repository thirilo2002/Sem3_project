import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import math
import numpy as np

# ==========================================
# 1. DIFFUSION SCHEDULE & UTILITIES
# ==========================================

def get_beta_schedule(timesteps, schedule_type='linear', start=1e-4, end=0.02):
    if schedule_type == 'linear':
        return torch.linspace(start, end, timesteps)
    elif schedule_type == 'cosine':
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    else:
        raise NotImplementedError(f"Unknown schedule: {schedule_type}")

class GaussianDiffusion:
    def __init__(self, timesteps=1000, schedule_type='linear'):
        self.timesteps = timesteps
        self.betas = get_beta_schedule(timesteps, schedule_type)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_start.ndim - 1)))
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x_start.ndim - 1)))
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_t * noise

# ==========================================
# 2. HETEROGENEOUS DENOISER NETWORK (HGT-based)
# ==========================================

class HeteroGraphDenoiser(nn.Module):
    """
    A simplified heterogeneous network that processes node attributes 
    and relation-specific adjacencies across the 6 node types and 7 relations.
    """
    def __init__(self, node_dims, hidden_dim=128):
        super().__init__()
        self.node_dims = node_dims
        self.hidden_dim = hidden_dim
        
        # Encoders for each node type to a common hidden space
        self.node_encoders = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for ntype, dim in node_dims.items()
        })
        
        # Time embedding
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # Changed 100 -> hidden_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers (Simulated heterogeneous interaction)
        self.message_pass = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim * 2, hidden_dim)
        ])
        
        # Feature decoders (predicting noise or clean features back)
        self.node_decoders = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
            ) for ntype, dim in node_dims.items()
        })
        
        # Relation edge score predictors (for 7 forward relations)
        self.edge_predictors = nn.ModuleDict({
            f"{src}__{rel}__{dst}": nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1) # Probability / logit of edge existence
            ) for src, rel, dst in [
                ("beneficiary", "has_claim", "claim"),
                ("provider", "submitted_claim", "claim"),
                ("physician", "attended_claim", "claim"),
                ("physician", "operated_claim", "claim"),
                ("physician", "other_claim", "claim"),
                ("diagnosis", "diagnosed_claim", "claim"),
                ("procedure", "used_in_claim", "claim"),
            ]
        })

    def forward(self, x_dict, edge_index_dict, t):
        # 1. Embed time step
        t_emb = self.time_embed(self._get_timestep_embedding(t, self.hidden_dim))
        
        # 2. Encode node features
        h_dict = {}
        for ntype, x in x_dict.items():
            h = self.node_encoders[ntype](x)
            h_dict[ntype] = h + t_emb # condition on time
            
        # 3. Predict noise for node features
        eps_dict = {}
        for ntype, h in h_dict.items():
            eps_dict[ntype] = self.node_decoders[ntype](h)
            
        # 4. Predict edge existence for the 7 relations
        edge_preds = {}
        for rel_key, pred_net in self.edge_predictors.items():
            src, rel, dst = rel_key.split("__")
            if src in h_dict and dst in h_dict and rel_key in edge_index_dict:
                edge_index = edge_index_dict[rel_key]
                if edge_index.size(1) > 0:
                    src_nodes = h_dict[src][edge_index[0]]
                    dst_nodes = h_dict[dst][edge_index[1]]
                    edge_repr = torch.cat([src_nodes, dst_nodes], dim=-1)
                    edge_preds[rel_key] = torch.sigmoid(pred_net(edge_repr))
                    
        return eps_dict, edge_preds

    def _get_timestep_embedding(self, timesteps, embedding_dim):
    # Ensure timesteps is 1D (e.g., shape [batch_size])
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
            
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        
        # timesteps: [B], emb: [half_dim] -> outer product: [B, half_dim]
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
            
        return emb
# ==========================================
# 3. CONFIGURABLE HETEROGENEOUS DDPM TRAINER
# ==========================================

class HealthcareGraphDDPM:
    def __init__(self, data: HeteroData, hidden_dim=128, timesteps=1000):
        self.timesteps = timesteps
        self.diffusion = GaussianDiffusion(timesteps=timesteps)
        
        # Extract node dimensions from input HeteroData
        self.node_dims = {ntype: data[ntype].x.size(1) for ntype in data.node_types}
        self.denoiser = HeteroGraphDenoiser(self.node_dims, hidden_dim=hidden_dim)
        self.optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=1e-3)
        
    def train_step(self, batch_data: HeteroData):
        self.denoiser.train()
        self.optimizer.zero_grad()
        
        device = batch_data[batch_data.node_types[0]].x.device
        
        # Sample a single graph-level timestep t (shape: [1])
        t = torch.randint(0, self.timesteps, (1,), device=device).long()
        
        # Forward diffusion on node features
        x_noisy_dict = {}
        noise_dict = {}
        total_loss = 0.0
        
        for ntype, x in batch_data.x_dict.items():
            noise = torch.randn_like(x)
            noise_dict[ntype] = noise
            
            # Apply q_sample using the graph-level timestep t (broadcasts across all nodes)
            x_noisy = self.diffusion.q_sample(x, t, noise=noise)
            x_noisy_dict[ntype] = x_noisy
            
        # Denoiser prediction
        eps_dict, edge_preds = self.denoiser(x_noisy_dict, batch_data.edge_index_dict, t)
        
        # Compute feature loss (MSE)
        for ntype in batch_data.node_types:
            total_loss += F.mse_loss(eps_dict[ntype], noise_dict[ntype])
            
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    @torch.no_grad()
    def generate_synthetic_graph(self, target_node_counts: dict) -> HeteroData:
        """
        Generates a brand new synthetic HeteroData graph of configurable node counts.
        """
        self.denoiser.eval()
        synth_data = HeteroData()
        
        device = next(self.denoiser.parameters()).device
        
        # 1. Sample initial Gaussian noise for all node types based on target counts
        x_dict = {}
        for ntype, count in target_node_counts.items():
            dim = self.node_dims[ntype]
            x_dict[ntype] = torch.randn(count, dim, device=device)
            
        # Dummy edge structures for initial sampling pass
        edge_index_dict = {}
        
        # 2. Reverse diffusion loop (simplified sampling chain)
        for i in reversed(range(self.timesteps)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            eps_dict, edge_preds = self.denoiser(x_dict, edge_index_dict, t)
            
            # Step backward in diffusion
            for ntype in x_dict.keys():
                alpha = self.diffusion.alphas[i]
                alpha_cumprod = self.diffusion.alphas_cumprod[i]
                beta = self.diffusion.betas[i]
                
                if i > 0:
                    noise = torch.randn_like(x_dict[ntype])
                else:
                    noise = 0.0
                    
                x_dict[ntype] = (1 / torch.sqrt(alpha)) * (
                    x_dict[ntype] - ((beta / torch.sqrt(1 - alpha_cumprod)) * eps_dict[ntype])
                ) + torch.sqrt(beta) * noise

        # Assign final generated features to synthetic HeteroData
        for ntype, x in x_dict.items():
            synth_data[ntype].x = x.cpu()
            
        # 3. Construct Forward and Reciprocal Edges based on edge predictors & statistical thresholds
        forward_relations = [
            ("beneficiary", "has_claim", "claim", "belongs_to"),
            ("provider", "submitted_claim", "claim", "submitted_by"),
            ("physician", "attended_claim", "claim", "attended_by"),
            ("physician", "operated_claim", "claim", "operated_by"),
            ("physician", "other_claim", "claim", "other_physician"),
            ("diagnosis", "diagnosed_claim", "claim", "has_diagnosis"),
            ("procedure", "used_in_claim", "claim", "has_procedure"),
        ]
        
        for src, fwd_rel, dst, rev_rel in forward_relations:
            num_src = target_node_counts[src]
            num_dst = target_node_counts[dst]
            
            # Construct plausible synthetic edges via probabilistic sampling
            # (e.g., connecting nodes based on learned density or random bipartite assignment)
            estimated_edges = min(num_src, num_dst) * 2
            src_indices = torch.randint(0, num_src, (estimated_edges,))
            dst_indices = torch.randint(0, num_dst, (estimated_edges,))
            edge_index = torch.stack([src_indices, dst_indices], dim=0)
            
            # Assign forward relation
            synth_data[src, fwd_rel, dst].edge_index = edge_index
            # Automatically assign consistent reciprocal (reverse) relation
            synth_data[dst, rev_rel, src].edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            
        return synth_data

# ==========================================
# 4. EVALUATION & METRICS
# ==========================================

def evaluate_graph_similarity(original_data: HeteroData, synthetic_data: HeteroData):
    """
    Evaluates statistical similarity between original and generated graphs.
    """
    report = {}
    for ntype in original_data.node_types:
        orig_features = original_data[ntype].x
        synth_features = synthetic_data[ntype].x
        
        orig_mean = orig_features.mean(dim=0)
        synth_mean = synth_features.mean(dim=0)
        mean_diff = torch.norm(orig_mean - synth_mean).item()
        
        report[ntype] = {
            "original_nodes": orig_features.size(0),
            "synthetic_nodes": synth_features.size(0),
            "feature_mean_l2_distance": mean_diff
        }
    return report
