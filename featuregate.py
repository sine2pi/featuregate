class FeatureGate(nn.Module):
    def __init__(n, dims, num_features=4, top_k=2):
        super().__init__()
        n.num_features = num_features
        n.top_k = top_k
        
        n.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(dims, dims//2), nn.GELU(), nn.Linear(dims//2, 1), nn.Sigmoid()) 
            for _ in range(num_features)
        ])
        
        n.s_router = nn.Linear(dims, num_features)
        n.h_router = nn.Linear(dims, num_features)
        n.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        n.bias_q = nn.Linear(dims, dims)
        n.bias_k = nn.Linear(dims, dims)
        
    def compute_feature_bias(n, x, xa, head_dim):
        q = n.bias_q(x)
        k = n.bias_k(xa)
        bias = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        return bias.mean(1)
    
    def forward(n, x, features):
        B, L, D = x.shape
        x_pooled = x.mean(1)

        list = list(features.values())
        g_scores = []
        
        for idx, (g_fn, feat) in enumerate(zip(n.gates, list)):
            if feat is None:
                g_scores.append(torch.zeros(B, 1, device=device))
            else:
                base_score = g_fn(x_pooled)
                
                attn_bias = n.compute_feature_bias(x, feat, D)
                influence = attn_bias.mean(-1, keepdim=True)
                
                combined = base_score * (1 + 0.3 * torch.tanh(influence))
                g_scores.append(combined)
        
        gates = torch.stack(g_scores, dim=-1).squeeze(1)
        
        soft_w = torch.softmax(n.s_router(x_pooled), dim=-1)
        
        h_logits = n.h_router(x_pooled)
        vals, idx = torch.topk(h_logits, n.top_k, dim=-1)
        hard_w = torch.zeros_like(soft_w)
        hard_w.scatter_(-1, idx, torch.softmax(vals, dim=-1))

        alpha = torch.sigmoid(n.alpha)
        routing_w = alpha * hard_w + (1 - alpha) * soft_w
        
        final = routing_w * gates
        final = final / (final.sum(-1, keepdim=True) + 1e-8)

        stacked = torch.stack([f for f in list if f is not None], dim=1)

        weighted = torch.einsum('bf,bfkd->bkd', final, stacked)
        return weighted
