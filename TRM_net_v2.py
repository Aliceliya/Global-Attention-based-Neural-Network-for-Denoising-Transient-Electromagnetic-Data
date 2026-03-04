import torch
import torch.nn as nn
from testswin import SwinTransformer


class denoising_model(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate

        # 多尺度卷积模块（保持不变）
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.GELU(),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(4),
            nn.GELU(),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(8),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(4),
            nn.GELU(),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(8),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.GELU(),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        # 修改1: 在融合层后添加Dropout
        self.fusion = nn.Sequential(
            nn.Conv1d(in_channels=192, out_channels=192, kernel_size=1),
            nn.Dropout(dropout_rate),  # 添加Dropout
            nn.Conv1d(in_channels=192, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)  # 较小的Dropout
        )

        # 修改2: Transformer层添加dropout和layer normalization改进
        self.TRM = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,  # 增加前馈网络维度
                dropout=dropout_rate,  # 添加dropout
                activation='gelu',  # 使用GELU激活
                batch_first=False,  # 保持原有格式
                norm_first=True  # Pre-norm结构，训练更稳定
            ) for _ in range(6)
        ])

        # 残差块（保持不变，但可以添加dropout）
        self.res1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.3),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        self.res11 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1)

        self.res2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.3),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1024),
            nn.GELU()
        )
        self.res22 = nn.Conv1d(in_channels=256, out_channels=1024, kernel_size=1)

        self.res3 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.3),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        self.res33 = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=1)

        self.res4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.3),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.res44 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1)
        self.Relu = nn.GELU()

        # Swin Transformer（保持不变）
        self.swin = SwinTransformer(
            in_chans=1,
            patch_size=2,
            window_size=(8, 16, 32, 64),
            embed_dim=64,
            depths=(4, 4, 4, 4),
            num_heads=(2, 4, 8, 16),
            num_classes=1024
        )

        self.norm = nn.LayerNorm(64)

        # 修改3: 改进的注意力机制，添加dropout
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(192, 12, 1),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Conv1d(12, 192, 1),
            nn.GELU()
        )

        # 上采样模块（可以添加dropout）
        self.up_sample = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),

            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),

            nn.Conv1d(16, 1, kernel_size=1)
        )

        # 修改4: 改进的最终调整层
        self.fc_adjust = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(dropout_rate * 0.2),  # 添加dropout
            # 可以考虑添加LayerNorm
            # nn.LayerNorm(1024),
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.TransformerEncoderLayer):
            # Transformer层的特殊初始化
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x_log, training=True):
        # 多尺度特征提取
        x1 = self.conv1(x_log)
        x2 = self.conv2(x_log)
        x3 = self.conv3(x_log)

        # 特征融合
        fused = torch.cat([x1, x2, x3], dim=1)
        print(fused.shape)
        attn = self.attention(fused)
        x = self.fusion(fused * attn)
        print(x.shape)

        # 修改5: Transformer层使用循环结构，便于添加dropout控制
        res = x
        x_permuted = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)

        for transformer_layer in self.TRM:
            x_permuted = transformer_layer(x_permuted)

        x = self.norm(x_permuted).permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        x = self.Relu(res + x)

        # 残差连接
        res = x

        res1 = self.res11(x)
        x = self.res1(x)
        x = self.Relu(res1 + x)

        res2 = self.res22(x)
        x = self.res2(x)
        x = self.Relu(res2 + x)

        res3 = self.res33(x)
        x = self.res3(x)
        x = self.Relu(res3 + x)

        res4 = self.res44(x)
        x = self.res4(x)
        x = self.Relu(res4 + x)

        x = self.Relu(res + x)

        # Swin Transformer
        x = self.swin(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 上采样
        x = self.up_sample(x)

        # 最终调整
        x = x.squeeze(1)
        x = self.fc_adjust(x)

        return x


if __name__ == '__main__':
    x = torch.zeros(64, 1, 1024)
    net = denoising_model(dropout_rate=0.1)
    print(net(x).shape)
    print(f"Total parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

