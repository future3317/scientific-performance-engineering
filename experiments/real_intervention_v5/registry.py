"""A performance-blind roster, with conservative aliases and family-only splitting.

Each entry names an architecture publication/design, not a width, depth, batch,
source wrapper, or pretrained checkpoint. Shared ancestry is explicit; family
independence remains a sampling assumption, not something model names prove.
"""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import subprocess

from .common import digest, file_digest

TORCHBENCH = "/home/workspace/lrh/DATA/evospe-rt-v2/torchbench"
OVERLAY = "/home/workspace/lrh/RESULTS/evospe-rt-v2/20261018-one-shot-deterministic-b1-r2/dependencies"

# source, model, lineage, design basis. ViT training variants and ResNeXt are
# deliberately absent as extra families; duplicate TorchBench wrappers stay aliases.
TV = [
    ("resnet18", "resnet", "Residual networks; includes ResNeXt/wide/SE aliases"),
    ("densenet121", "densenet", "Dense concatenating connectivity"),
    ("mobilenet_v3_small", "mobilenet", "Inverted residual mobile lineage; v2/v3 aliases"),
    ("efficientnet_b0", "efficientnet", "Compound-scaled MBConv lineage"),
    ("convnext_tiny", "convnext", "Modernized depthwise ConvNet; v2 aliases"),
    ("regnet_y_400mf", "regnet", "RegNet design-space family; X/Y aliases"),
    ("shufflenet_v2_x0_5", "shufflenet", "Channel-shuffle split branches"),
    ("mnasnet0_5", "mnasnet", "MnasNet searched block topology"),
    ("squeezenet1_0", "squeezenet", "Fire squeeze/expand topology"),
    ("vgg11", "vgg", "VGG sequential topology; RepVGG conservatively merged"),
    ("alexnet", "alexnet", "AlexNet topology"),
    ("vit_b_16", "vit", "ViT token transformer; DeiT/BEiT/CaiT/CrossViT aliases not extra groups"),
    ("swin_t", "swin", "Shifted-window hierarchical transformer"),
    ("maxvit_t", "maxvit", "Multi-axis block/grid attention"),
    ("inception_v3", "inception", "Inception factorized multibranch lineage"),
]
TIMM = [
    ("ghostnet_100", "ghostnet", "Ghost feature synthesis"),
    ("hrnet_w18_small", "hrnet", "Parallel high-resolution fusion"),
    ("dla34", "dla", "Hierarchical deep-layer aggregation"),
    ("nfnet_f0", "nfnet", "Normalizer-free scaled residual design"),
    ("legacy_xception", "xception", "Extreme separable Inception topology"),
    ("nasnetalarge", "nasnet", "NASNet cell lineage; PNASNet merged"),
    ("dpn68", "dpn", "Dual residual/dense paths"),
    ("ese_vovnet19b_slim", "vovnet", "One-shot feature aggregation"),
    ("darknet17", "darknet", "Darknet convolutional topology; CSP aliases merged"),
    ("selecsls42", "selecsls", "Selective long/short-range connections"),
    ("convmixer_768_32", "convmixer", "Patch convolutional mixer"),
    ("mixer_s16_224", "mlp_mixer", "Token/channel MLP mixing"),
    ("gmlp_ti16_224", "gmlp", "Spatial gating unit"),
    ("resmlp_12_224", "resmlp", "Residual affine token MLP"),
    ("coat_lite_tiny", "coat", "Co-scale conv-attention transformer"),
    ("coatnet_0_rw_224", "coatnet", "Staged MBConv/relative-attention CoAtNet"),
    ("pvt_v2_b0", "pvt", "Spatial-reduction pyramid transformer"),
    ("twins_pcpvt_small", "twins", "Locally grouped/global subsampled attention"),
    ("tnt_s_patch16_224", "tnt", "Inner/outer transformer tokens"),
    ("pit_ti_224", "pit", "Pooling-based token hierarchy"),
    ("nest_tiny", "nest", "Nested local transformer aggregation"),
    ("xcit_nano_12_p16_224", "xcit", "Cross-covariance channel attention"),
    ("volo_d1_224", "volo", "Outlooker local aggregation"),
    ("levit_128s", "levit", "LeViT convolution/attention pyramid"),
    ("mobilevit_xxs", "mobilevit", "MobileViT unfold/fold transformer blocks"),
    ("efficientformer_l1", "efficientformer", "Dimension-consistent meta blocks"),
    ("efficientvit_b0", "efficientvit", "EfficientViT linear attention (B lineage)"),
    ("edgenext_xx_small", "edgenext", "Split depthwise transpose attention"),
    ("fastvit_t8", "fastvit", "RepMixer hybrid topology"),
    ("focalnet_tiny_srf", "focalnet", "Focal modulation"),
    ("gcvit_xxtiny", "gcvit", "Global-context vision attention"),
    ("hiera_tiny_224", "hiera", "Hiera masked hierarchy topology"),
    ("davit_tiny", "davit", "Dual spatial/channel attention"),
    ("fasternet_t0", "fasternet", "Partial convolution topology"),
    ("repvit_m0_9", "repvit", "RepViT structural reparameterized mobile mixing"),
    ("sequencer2d_s", "sequencer2d", "Bidirectional spatial LSTM mixing"),
    ("poolformer_s12", "poolformer", "Nonparametric pooling token mixing; MetaFormer variants merged"),
    ("visformer_tiny", "visformer", "Convolutional early stages with late attention"),
]
TB = [
    ("resnet18", "resnet", "TorchBench wrapper of existing ResNet lineage"),
    ("pytorch_unet", "unet", "U-Net encoder/decoder skip topology"),
    ("dlrm", "dlrm", "DLRM embedding/interactions"),
    ("pytorch_CycleGAN_and_pix2pix", "cyclegan_generator", "CycleGAN generator topology"),
    ("maml", "maml", "TorchBench MAML miniature network"),
    ("fastNLP_Bert", "bert", "BERT text transformer; requires local data/dependencies"),
]


def build_registry(seed=20261021):
    import timm
    import torchvision.models as tv

    models = {"torchvision": set(tv.list_models()), "timm": set(timm.list_models())}
    versions = {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "timm", "triton", "numpy")}
    tb_sha = subprocess.check_output(["git", "-C", TORCHBENCH, "rev-parse", "HEAD"], text=True).strip()
    all_lineages = sorted({r[1] for rows in (TV, TIMM, TB) for r in rows}, key=lambda s: digest([seed, s]))
    splits = {family: ("development", "calibration", "final")[i % 3] for i, family in enumerate(all_lineages)}
    rows = []
    for source, entries in (("torchvision", TV), ("timm", TIMM), ("torchbench", TB)):
        for name, lineage, basis in entries:
            if source == "torchbench":
                path = Path(TORCHBENCH) / "torchbenchmark/models" / name / "__init__.py"
                available = path.is_file()
                source_sha = file_digest(path) if available else None
            else:
                available = name in models[source]
                entry = timm.models.model_entrypoint(name) if source == "timm" and available else (tv.get_model_builder(name) if available else None)
                import inspect
                path = inspect.getsourcefile(entry) if entry else None
                source_sha = file_digest(path) if path else None
            rows.append({"candidate_id": f"{source}:{name}", "model": name, "source": source,
                         "family_id": lineage, "lineage_id": lineage, "lineage_basis": basis,
                         "split": splits[lineage], "available": available,
                         "source_file_sha256": source_sha, "source_version": tb_sha if source == "torchbench" else versions[source],
                         "pretrained": None if source == "torchbench" else False,
                         "initialization": "SOURCE_NATIVE_NO_NETWORK" if source == "torchbench" else "SEEDED_RANDOM_WEIGHTS",
                         "shape": [3, 299 if name == "legacy_xception" or name == "inception_v3" else 331 if name == "nasnetalarge" else 224, 299 if name == "legacy_xception" or name == "inception_v3" else 331 if name == "nasnetalarge" else 224],
                         "input_kind": "TORCHBENCH_NATIVE_SEEDED" if source == "torchbench" else "SEEDED_RANDOM",
                         "downloads_allowed": False, "dataset_accuracy_claim": False})
    # First source in the frozen roster is the sole representative. Aliases are
    # counted, never substituted after seeing correctness or performance.
    seen = set()
    for row in rows:
        row["representative"] = row["family_id"] not in seen
        seen.add(row["family_id"])
    return rows, versions, tb_sha


def assert_split_integrity(rows):
    by_family = {}
    for row in rows:
        family = row["family_id"]
        if family in by_family and by_family[family] != row["split"]:
            raise ValueError("lineage appears in multiple splits")
        by_family[family] = row["split"]
    return by_family
