import torch
import xformers
import xformers.ops as xops
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, query_dim=768, context_dim=1024,
                 heads=8, dropout=0.0,
                 use_xformers=True,
                 ):
        super().__init__()
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.heads = heads
        self.dim_head = query_dim // heads
        self.scale = 1 / (self.dim_head ** 0.5)
        self.out_proj = nn.Linear(query_dim, query_dim)


        self.norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_xformers = use_xformers

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)

        return tensor


    def forward(self, x, context):
        b, n, _ = x.shape

        resid_x = x
        norm_x = self.norm(x)

        q = self.head_to_batch_dim(self.to_q(norm_x))
        k = self.head_to_batch_dim(self.to_k(context))
        v = self.head_to_batch_dim(self.to_v(context))

        if self.use_xformers:
            attn_output = xops.memory_efficient_attention(
                q.contiguous(), k.contiguous(), v.contiguous(), scale=self.scale,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            attn_output = (attn @ v).transpose(1, 2).reshape(b, n, -1)

        attn_output = self.batch_to_head_dim(attn_output)

        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        x = resid_x + attn_output

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.net(self.norm(x))


class TransformerLayer(nn.Module):

    def __init__(self, query_dim=768, context_dim=1024,
                 heads=8, dropout=0.0,
                 use_xformers=True, ff_mult=4, use_cross_attn=False):

        super().__init__()
        self.self_attn = Attention(query_dim=query_dim,
                 context_dim=query_dim,
                 heads=heads,
                 dropout=dropout,
                 use_xformers=use_xformers,)
        if use_cross_attn:
            self.cross_attn = Attention(query_dim=query_dim,
                    context_dim=context_dim,
                    heads=heads,
                    dropout=dropout,
                    use_xformers=use_xformers,)
        else:
            self.cross_attn = None

        self.ff = FeedForward(query_dim, mult=ff_mult, dropout=dropout)
        self.gradient_checkpointing = False

    def forward(self, x, context):
        if self.gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(self.self_attn, x, x)
            if self.cross_attn is not None:
                x = torch.utils.checkpoint.checkpoint(self.cross_attn, x, context)
            x = torch.utils.checkpoint.checkpoint(self.ff, x)

        else:
            x = self.self_attn(x, x)
            if self.cross_attn is not None:
                x = self.cross_attn(x, context)
            x = self.ff(x)

        return x

class ResnetBlock2D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        intermediate_channels=None,
        dropout=0.0,
        groups=16,
        eps=1e-6,
        first_transpose=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dropout = torch.nn.Dropout(dropout)

        if first_transpose:
            conv_class = torch.nn.ConvTranspose2d
            conv_kwargs = {"kernel_size": 4, "stride": 2, "padding": 1}
        else:
            conv_class = torch.nn.Conv2d
            conv_kwargs = {"kernel_size": 3, "stride": 1, "padding": 1}

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = conv_class(in_channels, intermediate_channels if intermediate_channels is not None else out_channels,
                                             **conv_kwargs)

        self.norm2 = torch.nn.GroupNorm(num_groups=groups, 
                                            num_channels=intermediate_channels if intermediate_channels is not None else out_channels, 
                                            eps=eps, affine=True)
        self.conv2 = torch.nn.Conv2d(intermediate_channels if intermediate_channels is not None else out_channels, 
                                    out_channels, 
                                    kernel_size=3, stride=1, padding=1)

        self.nonlinearity = nn.SiLU()

    def forward(self, input_tensor):
        hidden_states = input_tensor

        hidden_states = self.nonlinearity(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)

        hidden_states = hidden_states + input_tensor

        hidden_states = self.nonlinearity(self.norm2(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        output_tensor = hidden_states
        #output_tensor = input_tensor + hidden_states

        return output_tensor

class ConvAndAct(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                    padding, 
                    transpose=False, 
                    output_padding=None,
                    bias=True,
                    norm="after",
                    groups=16
                    ):
                
        super().__init__()
        if transpose:
            model_cls = nn.ConvTranspose2d
        else:
            model_cls = nn.Conv2d

        kwargs = {}
        if output_padding is not None:
            kwargs['output_padding'] = output_padding

        self.conv = model_cls(in_channels, out_channels, kernel_size, stride, padding, bias=bias, **kwargs)
        self.act = nn.SiLU()
        if norm is not None:
            num_chan = out_channels if norm == "after" else in_channels
            self.norm = nn.GroupNorm(num_groups=groups, num_channels=num_chan, eps=1e-6, affine=True)
            self.do_norm = norm
        else:
            self.norm = None
            self.do_norm = None

    def forward(self, x):

        if self.do_norm == "before":
            x = self.norm(x)
        
        x = self.conv(x)

        if self.do_norm == "after":
            x = self.norm(x)

        return self.act(x)



# class Decoder(torch.nn.Module):

#     def __init__(self, num_layers=6, query_dim=768, context_dim=1024,
#                 image_size = 256,
#                 patch_size = 8,
#                  heads=8, 
#                  dropout=0.0,
#                  use_xformers=True, 
#                  ff_mult=4, 
#                  use_cross_attn=True,
#                  extra_conext_tokens=4,
#                  ):
#         super().__init__()

#         latent_size = image_size // patch_size

#         self.context_mapper = nn.Linear(context_dim, context_dim * extra_conext_tokens)

#         self.mlp = nn.Sequential(
#             nn.Linear(context_dim, context_dim * 2),
#             nn.GELU(),
#             nn.Linear(context_dim * 2, context_dim),
#         )

#         self.learned_query = nn.Parameter(torch.randn(1, query_dim, latent_size, latent_size) * 0.02)
#         self.layers = nn.ModuleList([
#             TransformerLayer(query_dim=query_dim,
#                  context_dim=context_dim,
#                  heads=heads,
#                  dropout=dropout,
#                  use_xformers=use_xformers,
#                  ff_mult=ff_mult,
#                  use_cross_attn=use_cross_attn,
#                  )
#             for _ in range(num_layers)
#         ])

#         self.mapper = nn.ModuleList([
#             ConvAndAct(query_dim, query_dim//2, 4, 2, 1, transpose=True),
#             ConvAndAct(query_dim//2, query_dim//4, 4, 2, 1, transpose=True),
#             ConvAndAct(query_dim//4, query_dim//8, 4, 2, 1, transpose=True),
#             ConvAndAct(query_dim//8, 3, 3, 1, 1)
#         ])
#         self.latent_size = latent_size
#         self.gradient_checkpointing = False

#     def enable_gradient_checkpointing(self):
#         self.gradient_checkpointing = True
#         for layer in self.layers:
#             layer.gradient_checkpointing = True

#     def forward(self, context):

#         # prepare context: (b, query_size) -> (b, ctx tokens, query_dim)
#         b, query_size = context.shape
#         context = self.context_mapper(context)
#         context = context.reshape(b, -1, query_size)

#         context = self.mlp(context)

#         x = self.learned_query.expand(b, -1, -1, -1)
#         x = x.permute(0, 2, 3, 1).reshape(b, self.latent_size * self.latent_size, -1)
#         for layer in self.layers:
#             x = layer(x, context)
#         x = x.reshape(b, self.latent_size, self.latent_size, -1).permute(0, 3, 1, 2)
#         for layer in self.mapper:
#             if self.gradient_checkpointing:
#                 x = torch.utils.checkpoint.checkpoint(layer, x)
#             else:
#                 x = layer(x)
#         return x




class Decoder(torch.nn.Module):

    def __init__(self, num_layers=6, query_dim=768, context_dim=1024,
                image_size = 256,
                patch_size = 8,
                z_dim = 64,
                 heads=8, 
                 dropout=0.0,
                 use_xformers=True, 
                 ff_mult=4, 
                 use_cross_attn=True,
                 extra_conext_tokens=4,
                 ):
        super().__init__()

        latent_size = image_size // patch_size

        self.context_mapper = nn.Linear(context_dim, context_dim * extra_conext_tokens)

        self.mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim * 2),
            nn.GELU(),
            nn.Linear(context_dim * 2, context_dim),
        )

        self.scale_in_layers = nn.ModuleList([
            ConvAndAct(z_dim, z_dim*16, 4, 1, 0, transpose=True),
            TransformerLayer(query_dim=z_dim*16, context_dim=context_dim, heads=heads, dropout=dropout, use_xformers=use_xformers, ff_mult=ff_mult, use_cross_attn=use_cross_attn),
            ConvAndAct(z_dim*16, z_dim*12, 4, 2, 1, transpose=True),
            TransformerLayer(query_dim=z_dim*12, context_dim=context_dim, heads=heads, dropout=dropout, use_xformers=use_xformers, ff_mult=ff_mult, use_cross_attn=use_cross_attn),
            ConvAndAct(z_dim*12, z_dim*12, 4, 2, 1, transpose=True),
            TransformerLayer(query_dim=z_dim*12, context_dim=context_dim, heads=heads, dropout=dropout, use_xformers=use_xformers, ff_mult=ff_mult, use_cross_attn=use_cross_attn),
            #ConvAndAct(z_dim*12, z_dim*12, 4, 2, 1, transpose=True),
            ]
        )

        self.layers = nn.ModuleList([
            TransformerLayer(query_dim=query_dim,
                 context_dim=context_dim,
                 heads=heads,
                 dropout=dropout,
                 use_xformers=use_xformers,
                 ff_mult=ff_mult,
                 use_cross_attn=use_cross_attn,
                 )
            for _ in range(num_layers)
        ])

        # self.mapper = nn.ModuleList([
        #     ConvAndAct(query_dim, query_dim//2, 4, 2, 1, transpose=True),
        #     ConvAndAct(query_dim//2, query_dim//4, 4, 2, 1, transpose=True),
        #     ConvAndAct(query_dim//4, query_dim//8, 4, 2, 1, transpose=True),
        #     ConvAndAct(query_dim//8, 3, 3, 1, 1, norm=None)
        # ])

        # self.mapper = nn.ModuleList([
        #     ConvAndAct(query_dim, query_dim//2, 3, 1, 1),
        #     ConvAndAct(query_dim//2, query_dim//4, 3, 1, 1),
        #     ConvAndAct(query_dim//4, query_dim//8, 3, 1, 1),
        #     ConvAndAct(query_dim//8, 3, 3, 1, 1, norm=None)
        # ])

        self.mapper = nn.ModuleList([
            ResnetBlock2D(query_dim, query_dim//2, intermediate_channels=query_dim),
            ResnetBlock2D(query_dim//2, query_dim//4, intermediate_channels=query_dim//2),
            ResnetBlock2D(query_dim//4, query_dim//8, intermediate_channels=query_dim//4),
            ConvAndAct(query_dim//8, 3, 3, 1, 1, norm=None)
        ])

        self.latent_size = latent_size
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        for layer in self.layers:
            layer.gradient_checkpointing = True

    def forward(self, x, context):

        # prepare context: (b, query_size) -> (b, ctx tokens, query_dim)
        b, context_size = context.shape
        context = self.context_mapper(context)
        context = context.reshape(b, -1, context_size)

        context = self.mlp(context)

        #
        for i, layer in enumerate(self.scale_in_layers):
            if i % 2 == 0:
                x = layer(x)
            else:
                b, c, h, w = x.shape
                x = x.reshape(b, -1, h * w).permute(0, 2, 1)
                x = layer(x, context)
                x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        
        b, c, h, w = x.shape
        x = x.reshape(b, -1, h * w).permute(0, 2, 1)

        for layer in self.layers:
            x = layer(x, context)
        x = x.reshape(b, self.latent_size, self.latent_size, -1).permute(0, 3, 1, 2)
        for i, layer in enumerate(self.mapper):
            if self.gradient_checkpointing:
                if i!=3:
                    x = F.interpolate(x, scale_factor=2, mode='nearest')
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                if i!=3:
                    x = F.interpolate(x, scale_factor=2, mode='nearest')
                x = layer(x)
        return x