import torch
import pickle
import torch.nn as nn
import torch.nn.functional as Fuct
import numpy as np
from transformers import GPT2Model
from typing import Optional, Tuple, Union
from peft import LoraConfig, get_peft_model

class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb

from dataclasses import dataclass

@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
class PFA_noG(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=1, dropout_rate=0.0):
        super(PFA_noG, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2",
            attn_implementation="eager",
            output_attentions=True,
            output_hidden_states=True
        )

        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U
        self.device = device
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.lora_rank = 16

        self.lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            lora_dropout=self.dropout_rate,
            target_modules=['q_attn', 'c_attn'],
            bias="none",
            use_dora=True,
        )

        self.gpt2 = get_peft_model(self.gpt2, self.lora_config)

        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def custom_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:

        output_attentions = (
            output_attentions if output_attentions is not None
            else self.gpt2.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.gpt2.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.gpt2.config.use_cache
        return_dict = return_dict if return_dict is not None else self.gpt2.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.gpt2.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.gpt2.wte(input_ids)

        position_embeds = self.gpt2.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        presents = () if use_cache else None

        for i, (block, layer_past) in enumerate(zip(self.gpt2.h, past_key_values)):
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]

            if use_cache:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)

        hidden_states = self.gpt2.ln_f(hidden_states)
        hidden_states = hidden_states.view((-1,) + input_shape[1:] + (hidden_states.size(-1),))

        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )

    def forward(self, x):
        """
        Args:
            x: input embeddings [batch_size, sequence_length, hidden_dim]
        """
        output = self.custom_forward(
            inputs_embeds=x
        ).last_hidden_state

        output = self.dropout(output)
        return output




class LearnableAlphaEmbeddingSelector(nn.Module):
    def __init__(self, projection_dim=None, init_alpha=0.5):
        """
        Args:
            projection_dim: 若需要线性投影，可指定维度
            init_alpha: α 的初始值（节点级权重），范围 [0,1]
        """
        super().__init__()

        # 可学习参数 α（使用 sigmoid 约束在 [0,1]）
        self.alpha_raw = nn.Parameter(torch.tensor(init_alpha))

        # 可选线性投影层（默认恒等映射）
        self.similarity_proj = (
            nn.Linear(projection_dim, projection_dim, bias=False)
            if projection_dim is not None
            else nn.Identity()
        )

    def select_top_k_embeddings(self, query, key_embeddings, k):
        """
        动态选择最相关的 K 个词嵌入（节点级 + 图级相似度混合）

        Args:
            query: [B, N, D]  —— 时空节点特征
            key_embeddings: [V, D]  —— GPT2 词嵌入矩阵
            k: 选取的 Top-K 数量

        Returns:
            selected_keys: [B, k, D]
            topk_indices: [B, k]
            fused_similarity: [B, V]
            alpha: 当前 batch 下的有效融合系数
        """
        B, N, D = query.shape
        V = key_embeddings.size(0)

        # 1️⃣ 归一化向量
        query_norm = Fuct.normalize(query, p=2, dim=-1)  # [B, N, D]
        key_norm = Fuct.normalize(key_embeddings, p=2, dim=-1)  # [V, D]

        # 2️⃣ 节点级相似度
        node_sim = torch.matmul(query_norm, key_norm.T)  # [B, N, V]
        node_sim_pool, _ = node_sim.max(dim=1)

        # 3️⃣ 图级相似度（平均后）
        graph_query = query_norm.mean(dim=1)  # [B, D]
        graph_sim = torch.matmul(graph_query, key_norm.T)  # [B, V]

        # 4️⃣ α 融合（Sigmoid约束）
        alpha = torch.sigmoid(self.alpha_raw)  # 标量 ∈ [0,1]
        fused_similarity = alpha * node_sim_pool + (1 - alpha) * graph_sim  # [B, V]

        # 5️⃣ 选择 Top-K
        topk_values, topk_indices = torch.topk(fused_similarity, k=k, dim=-1)  # [B, k]

        # 6️⃣ 提取对应词嵌入
        key_expanded = key_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, V, D]
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, k, D]
        selected_keys = torch.gather(key_expanded, 1, idx_expanded)  # [B, k, D]

        return selected_keys, topk_indices, fused_similarity, alpha



# 新增new memory


class GraphMemoryPoolV2(nn.Module):
    """
    更稳定的图记忆池版本：
    1. memory 只作为可训练参数，不在 forward 中手工改写
    2. self-memory / neighbor-memory 都从 memory pool 读取
    3. 三路 softmax 融合：x / self_mem / nbr_mem
    4. 支持 top-r 稀疏读取，减少噪声
    """

    def __init__(
        self,
        num_nodes=250,
        memory_size=16,
        feature_dim=768,
        dropout=0.1,
        top_r=4,
        temperature=0.7
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.top_r = top_r
        self.temperature = temperature

        # key-value memory
        self.mem_keys = nn.Parameter(
            torch.randn(num_nodes, memory_size, feature_dim) * 0.02
        )
        self.mem_vals = nn.Parameter(
            torch.randn(num_nodes, memory_size, feature_dim) * 0.02
        )

        self.q_proj = nn.Linear(feature_dim, feature_dim)

        # 三路融合：x / self_mem / nbr_mem
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 3)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.layer_norm = nn.LayerNorm(feature_dim)

        # 方便后面做辅助 loss / 监控
        self.last_aux = {}

    def _normalize_adj(self, adj_mx):
        adj = adj_mx + torch.eye(adj_mx.size(0), device=adj_mx.device, dtype=adj_mx.dtype)
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    def _topr_read(self, q, keys, vals):
        """
        q:    [B, N, D]
        keys: [N, M, D]
        vals: [N, M, D]

        return:
            mem_out:    [B, N, D]
            route_prob: [B, N, M]
        """
        B, N, D = q.shape
        M = keys.size(1)
        r = min(self.top_r, M)

        q = Fuct.normalize(q, p=2, dim=-1)
        keys = Fuct.normalize(keys, p=2, dim=-1)

        # [B, N, M]
        sim = torch.einsum("bnd,nmd->bnm", q, keys)

        # 保存完整 routing 概率，后续可做均衡 loss
        route_prob = torch.softmax(sim / self.temperature, dim=-1)

        # 稀疏 top-r 读取
        topv, topi = torch.topk(sim, k=r, dim=-1)                 # [B,N,r]
        alpha = torch.softmax(topv / self.temperature, dim=-1)    # [B,N,r]

        vals_expand = vals.unsqueeze(0).expand(B, -1, -1, -1)     # [B,N,M,D]
        idx = topi.unsqueeze(-1).expand(-1, -1, -1, D)            # [B,N,r,D]
        picked_vals = torch.gather(vals_expand, 2, idx)           # [B,N,r,D]

        mem_out = (alpha.unsqueeze(-1) * picked_vals).sum(dim=2)  # [B,N,D]
        return mem_out, route_prob

    def forward(self, x, adj_mx):
        """
        x: [B, N, D]
        adj_mx: [N, N]
        """
        B, N, D = x.shape
        assert N == self.num_nodes, f"Expected {self.num_nodes} nodes, got {N}"

        q = self.q_proj(x)  # [B,N,D]

        # ===== 1) self memory read =====
        self_mem, self_route = self._topr_read(q, self.mem_keys, self.mem_vals)

        # ===== 2) neighbor memory read =====
        adj_norm = self._normalize_adj(adj_mx)                    # [N,N]
        nbr_keys = torch.einsum("ij,jmd->imd", adj_norm, self.mem_keys)
        nbr_vals = torch.einsum("ij,jmd->imd", adj_norm, self.mem_vals)

        nbr_mem, nbr_route = self._topr_read(q, nbr_keys, nbr_vals)

        # ===== 3) 三路融合 =====
        fusion_logits = self.fusion_mlp(torch.cat([x, self_mem, nbr_mem], dim=-1))  # [B,N,3]
        fusion_w = torch.softmax(fusion_logits, dim=-1)

        fused = (
            fusion_w[..., 0:1] * x +
            fusion_w[..., 1:2] * self_mem +
            fusion_w[..., 2:3] * nbr_mem
        )  # [B,N,D]

        enhanced_x = self.layer_norm(x + self.output_proj(fused))

        # 记录辅助统计，后面如果要加正则/可视化会很方便
        self.last_aux = {
            "self_route": self_route,
            "nbr_route": nbr_route,
            "fusion_w": fusion_w,
        }

        return enhanced_x



class ST_LLM_topk_memory_nog2(nn.Module):
    def __init__(
            self,
            device,
            adj_mx,
            input_dim=3,
            num_nodes=170,
            input_len=12,
            output_len=12,
            llm_layer=6,
            U=1,
            topk=1000,
            memory_size=50,
            gcn_layers=2
    ):
        super().__init__()

        self.device = device
        self.adj_mx = torch.tensor(adj_mx, dtype=torch.float32).to(self.device)
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.llm_layer = llm_layer
        self.U = U
        self.dropout_rate = 0.1
        print("u的值",U)
        # 新增
        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.1, batch_first=False)
        self.similarity_proj = nn.Linear(768, 768)
        self.top_k = topk  # 可调节的超参数
        self.topk_saved = False

        if num_nodes == 170 or num_nodes == 207:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48

        gpt_channel = 256
        to_gpt_channel = 768

        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

        self.Temb = TemporalEmbedding(time, gpt_channel)
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        self.in_layer = nn.Conv2d(gpt_channel * 3, to_gpt_channel, kernel_size=(1, 1))
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # regression
        self.regression_layer = nn.Conv2d(to_gpt_channel, self.output_len, kernel_size=(1, 1))

        self.graph_memory_pool = GraphMemoryPoolV2(
            num_nodes=num_nodes,
            memory_size=16,  # 先别太大
            feature_dim=to_gpt_channel,
            dropout=0.1,
            top_r=4,
            temperature=0.7
        )
        # GPT2
        self.gpt = PFA_noG(device=self.device, gpt_layers=self.llm_layer, U=self.U, dropout_rate=self.dropout_rate)

        # word embedding
        self.gpt_word_emb = self.gpt.gpt2.wte.weight.detach().to(self.device)  # GPT-2 embedding 矩阵

        self.layer_norm = nn.LayerNorm(to_gpt_channel)
        self.embedding_selector = LearnableAlphaEmbeddingSelector(
            projection_dim=to_gpt_channel,
            init_alpha=0.5
        )
        #测试新的输出层
        self.proj = nn.Sequential(
            nn.Linear(768, 768 // 2),  # 768 → 384
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768 // 2, output_len)  # 384 → 12
        )

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




    def forward(self, history_data):

        data = history_data.permute(0, 3, 2, 1)
        B, T, S, F = data.shape
        # print(data.shape) #[64, 12, 250, 3]

        # Temporal Embedding
        tem_emb = self.Temb(data)
        # print(tem_emb.shape) #[64, 256, 250, 1]

        node_emb = []
        node_emb.append(
            self.node_emb.unsqueeze(0)
            .expand(B, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )

        input_data = data.permute(0, 3, 2, 1)  # [32, 2, 207, 12]
        input_data = input_data.transpose(1, 2).contiguous()  # [32, 207, 2, 12]
        input_data = (input_data.view(B, S, -1).transpose(1, 2).unsqueeze(-1))
        # print(input_data.shape) #[64, 36, 250, 1]
        input_data = self.start_conv(input_data)
        # print(input_data.shape)#[64, 36, 250, 1]

        data_st = torch.cat([input_data] + [tem_emb] + node_emb, dim=1)
        # print(f"After cat: data_st shape: {data_st.shape}, type: {type(data_st)}")#64, 768, 250, 1

        data_st = self.in_layer(data_st)
        # print(f"After in_layer: data_st shape: {data_st.shape}, type: {type(data_st)}")

        data_st = Fuct.leaky_relu(data_st)
        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)

        # print(f"After permute: data_st shape: {data_st.shape}, type: {type(data_st)}")#[64, 250, 768]
        # 新增
        # === Cross-Attention 部分 ===
        # Q = 时间特征 (data_st)
        # K, V = GPT-2 词嵌入
        #gpt_word_emb = self.gpt_word_emb.unsqueeze(0).expand(data_st.size(0), -1, -1)  # [B, 50257, 768]
        # 选择最相关的K个词嵌入(如K=1000)
        #gpt_word_emb = self.gpt_word_emb  # [50257, 768]

        # === 使用TopK优化的交叉注意力 ===
        B, N, D = data_st.shape

        # 选择最相关的K个词嵌入
        selected_embeddings, topk_indices, fused_similarity, alpha = \
            self.embedding_selector.select_top_k_embeddings(
                query=data_st,
                key_embeddings=self.gpt_word_emb,
                k=self.top_k
            )


        q = data_st.transpose(0, 1).contiguous()  # [N, B, D] = [250, 64, 768]

        # 关键：k和v也要按照 [序列长度, batch, 特征] 的格式
        k = selected_embeddings.transpose(0, 1).contiguous()  # [k, B, D] = [1000, 64, 768]
        v = selected_embeddings.transpose(0, 1).contiguous()  # [k, B, D] = [1000, 64, 768]

        # 现在形状正确了
        attn_out, attn_weights = self.cross_attn(q, k, v)  # [N, B, D]
        attn_out = attn_out.transpose(0, 1).contiguous()  # [B, N, D]

        # 残差连接

        data_st = data_st + attn_out
        data_st = self.layer_norm(data_st)

        data_st = self.graph_memory_pool(data_st, self.adj_mx)

        '''# === Cross-Attention 按节点分批 ===
        N_chunk = 10
        B, N, D = data_st.shape
        attn_out_list = []

        for i in range(0, N, N_chunk):
            x_chunk = data_st[:, i:i + N_chunk, :]  # [B, N_chunk, D]

            # transpose for MultiheadAttention
            q = x_chunk.transpose(0, 1).contiguous()  # [N_chunk, B, D]
            k = v = gpt_word_emb.transpose(0, 1).contiguous()  # [V, B, D]

            attn_out_chunk, _ = self.cross_attn(q, k, v)  # [N_chunk, B, D]
            attn_out_chunk = attn_out_chunk.transpose(0, 1)  # [B, N_chunk, D]
            attn_out_list.append(attn_out_chunk)

        data_st = torch.cat(attn_out_list, dim=1)  # [B, N, D]

        print(f"After cross_attn: data_st shape: {data_st.shape}, type: {type(data_st)}")

        # === 再送入 GPT2 处理 ==='''

        outputs = self.gpt(data_st)


        outputs = self.proj(outputs)

        outputs = outputs.permute(0, 2, 1).unsqueeze(-1)
        # print(outputs.shape) #[64, 768, 250, 1]

        # regression
        #outputs = self.regression_layer(outputs)

        # print(outputs.shape) #[64, 12, 250, 1]

        return outputs

