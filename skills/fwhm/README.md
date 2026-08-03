# FWHM Skill

离线、可审计地测量荧光珠或细线结构的横向与轴向 FWHM，并生成候选标注、逐 ROI 审核证据和审核后统计。

## 快速开始

从 Skill 根目录运行：

```powershell
python scripts\run_fwhm.py beads "<input>" `
  --pixel-size-xy-nm 30.1 --pixel-size-z-nm 96.9

python scripts\run_fwhm.py lines "<input>" `
  --pixel-size-xy-nm 30.1 --pixel-size-z-nm 96.9 `
  --theoretical-lateral-resolution-nm 100 `
  --theoretical-axial-resolution-nm 290
```

打开已有结果的人工审核：

```powershell
python scripts\run_fwhm.py review "<result-directory>"
```

审核服务只监听 `127.0.0.1`。关闭浏览器页面不会停止服务；回到启动它的终端按 `Ctrl+C`。

## 线条审核契约

对于 `ZYX` 线条数据：

- 横向拟合和横向审核使用候选峰值 Z 的同一单层图像，不使用多层投影。
- `candidates-XY.tif` 只在峰值 Z 层标出横向法线；`candidates-Z.tif` 在实际 Z profile 范围逐层标出轴向候选。
- 横向与轴向分别使用 `review-assets/XY/`、`review-assets/Z/` 和独立人工决定；删除一个方向不会删除另一个方向。
- 审核器每页最多 150 项，先加载 30 项，接近页底时每次再加载 30 项。

## 文档索引

- [SKILL.md](SKILL.md) — Agent 使用入口、参数收集、运行和结果解释规则。
- [references/method.md](references/method.md) — 测量公式、质量筛选、输出和审核语义。
- [docs/decisions/ADR-001-direction-specific-line-review.md](docs/decisions/ADR-001-direction-specific-line-review.md) — 选择峰值 Z 单层和 XY/Z 独立审核的设计决策。
- [docs/solutions/conventions/direction-specific-review-contracts-3d-line-fwhm.md](docs/solutions/conventions/direction-specific-review-contracts-3d-line-fwhm.md) — 方向特定审核契约及其回归保护。

`docs/decisions/` 保存长期设计决策及取舍；修改线条测量或审核数据模型前应先查阅。`docs/solutions/` 保存已解决问题、踩坑经验和可复用修复方案；调整相邻流程或排查回归时应先检索。
