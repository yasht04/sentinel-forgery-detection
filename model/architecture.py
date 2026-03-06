import torch
import torch.nn as nn
from transformers import DistilBertModel
import timm


class VisionBranch(nn.Module):
    def __init__(self, pretrained=False):
        super(VisionBranch, self).__init__()
        backbone = timm.create_model("efficientnet_b4", pretrained=pretrained)
        self.features = backbone.conv_stem
        self.bn0 = backbone.bn1
        self.blocks = backbone.blocks
        self.conv_head = backbone.conv_head
        self.bn1 = backbone.bn2
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.bn0(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn1(x)
        spatial_features = x
        pooled = self.global_pool(x)
        pooled = pooled.flatten(1)
        return spatial_features, pooled


class TextBranch(nn.Module):
    def __init__(self, pretrained=False):
        super(TextBranch, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding


class FusionModule(nn.Module):
    def __init__(self, vision_dim=1792, text_dim=768, fusion_dim=512, num_heads=8):
        super(FusionModule, self).__init__()
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )

    def forward(self, vision_pooled, text_embedding):
        v = self.vision_proj(vision_pooled)
        t = self.text_proj(text_embedding)
        v_seq = v.unsqueeze(1)
        t_seq = t.unsqueeze(1)
        attn_output, _ = self.cross_attn(query=v_seq, key=t_seq, value=t_seq)
        attn_output = attn_output.squeeze(1)
        combined = torch.cat([v, attn_output], dim=1)
        fused = self.fusion_layer(combined)
        logit = self.classifier(fused)
        return logit, fused


class UNetDecoder(nn.Module):
    def __init__(self, in_channels=1792):
        super(UNetDecoder, self).__init__()

        def decoder_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.dec1 = decoder_block(1792, 512)
        self.dec2 = decoder_block(512, 256)
        self.dec3 = decoder_block(256, 128)
        self.dec4 = decoder_block(128, 64)
        self.dec5 = decoder_block(64, 32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.final_conv(x)
        return self.sigmoid(x)


class ForgeryDetector(nn.Module):
    def __init__(self):
        super(ForgeryDetector, self).__init__()
        self.vision_branch = VisionBranch(pretrained=False)
        self.text_branch = TextBranch(pretrained=False)
        self.fusion = FusionModule(vision_dim=1792, text_dim=768, fusion_dim=512)
        self.unet_decoder = UNetDecoder(in_channels=1792)

    def forward(self, image, input_ids, attention_mask):
        spatial_features, pooled_features = self.vision_branch(image)
        text_embedding = self.text_branch(input_ids, attention_mask)
        class_logit, _ = self.fusion(pooled_features, text_embedding)
        heatmap = self.unet_decoder(spatial_features)
        return class_logit, heatmap
