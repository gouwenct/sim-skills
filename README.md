# sim-skills

这是一个用 Git 管理的 MP-SIM 与 MF-SIM 可复用 Skill 集合，供 Codex、Claude、OpenCode 及其他能够发现 `~/.agents/skills` 的 Agent 使用。

仓库维护规则见 [AGENTS.md](AGENTS.md)。修改任何 Skill 时，必须重新核对其真实用法，并同步更新本文档对应章节。

## Skill 列表

- `r-3dmpsim`：编排从数据采集到 3D 重建的完整流程。
- `r-mfsim`：编排 MP Z 切片与 MF 重建流程。
- `mp-data-ingest`：下载并整理相机原始数据。
- `mp-preprocess`：定位科研项目的 real 入口，并提供 padding 工具。
- `simcheck`：计算保留相位信息的 SIM 质量指标。
- `mp-reconstruct`：定位并运行科研项目的 3D 重建入口。
- `mf-reconstruct`：切分 MP 堆栈，并运行科研项目的 MF 入口。
- `raw-check`：从相机原始数据计算初步 FPJ。
- `fwhm`：测量微球或细线结构的半高宽，并保留可审计的 ROI 证据。
- `decorr`：测量横向和轴向去相关分辨率。

可复用的预处理、检查和测量脚本保存在各自所属的 Skill 内。正式的 `Run_real_MP*.py`、`0-run5*.py`、`Run_MF*.py` 入口及其算法依赖、物理参数、标定 OTF、运行数据、凭据和 GPU 驱动仍保留在科研项目中。

## Python 环境

工具型 Skill 按以下顺序选择 Python：

1. 用户明确提供的 `SIM_SKILLS_PYTHON`。
2. 一个依赖完整的 `3DMPSIM*` Conda 环境。
3. 本仓库共享的 `.venv`。

如果存在多个兼容的 `3DMPSIM*` 环境，必须询问用户选择。如果一个也没有，必须先征得用户确认，再创建仅含 CPU 工具依赖的共享环境：

```powershell
& (Join-Path $repo 'environment.ps1') -CreateUv
```

UV 环境不安装 PyTorch、CuPy、CUDA 或重建依赖。只有选中的既有环境已经提供兼容的 CUDA PyTorch 时，`decorr` 才能使用 CUDA。科研项目正式入口始终使用项目指定的环境，不能使用这个 UV 后备环境。

## 手动操作文档

下列示例使用占位符，避免写死某台电脑的绝对路径：

```powershell
$repo = '<sim-skills仓库目录>'
$python = '<选定的python.exe>'
```

使用 `environment.ps1` 检查可用工具环境。存在多个兼容的 `3DMPSIM*` 环境时，应显式选择一个：

```powershell
$env:SIM_SKILLS_PYTHON = $python
& "$repo\environment.ps1" -Profile core
```

不要使用共享 UV 环境运行科研项目中的 `Run_real_MP*.py`、`0-run5*.py` 或 `Run_MF*.py`。

### `r-3dmpsim`：完整 3D MP-SIM 工作流

这是 Agent 编排型 Skill，不是独立脚本。调用时需明确提供项目路径、原始数据来源、批次、样本和目标阶段：

```text
使用 $r-3dmpsim 处理项目 <project-root>。处理原始批次 <raw-batch>，
样本为 <samples>，依次执行 ingest、real、padding 和 SIMcheck，之后只估计参数。
在我确认前不要运行完整重建。
```

各阶段使用的 Skill、输入和输出如下：

| Stage | 使用的 Skill | 输入 | 输出 |
| --- | --- | --- | --- |
| Stage 1：下载与整理 | `mp-data-ingest` | 网盘链接、直接 URL 或本地路径；原始批次目录 | 规范化的 `<raw-batch>/<sample>/`，cam2 TIFF 与 cam1 位于同一样本目录 |
| Stage 2A：real 预处理 | `mp-preprocess` | 规范化相机批次、`peizhun_file`、项目 `Run_real_MP*.py` 入口 | `<project>/code/real/dataOut/<batch>/<sample>/Dr/<track>/*.tif` |
| Stage 2B：padding | `mp-preprocess` | Stage 2A 生成的 Dr TIFF | 同级 `<track>_padding/*.tif`，默认尺寸为 462×462×120 |
| Stage 3：质量检查 | `simcheck` | 保留相位的 padded A TIFF、采集维度和页序 | `SIMcheck_V2/` 下的 MCNR、FPJ、统计表、报告和预览 |
| Stage 4：参数估计或重建 | `mp-reconstruct` | padded A 批次、标定 OTF、项目 `0-run5*.py` 入口 | 参数、日志、可选 `P/lihe_*.tif`；确认完整重建时才输出重建体 |
| Stage 5：去相关分辨率（可选） | `decorr` | Stage 4 的单个重建 TIFF 或严格的 27 样本集 | `decorr-results/` 下的分辨率数据、质控图和批次报告 |
| Stage 6：FWHM（可选） | `fwhm` | 已确认的微球或细线重建 TIFF | `FWHM-results/` 下的测量表、ROI 证据、图像和 `SUMMARY.md` |

Agent 写入文件前必须展示解析后的路径和参数。FWHM 与去相关检测是可选的重建后 Skill，不是强制阶段。

`r-3dmpsim` 本身没有命令行参数。Agent 工作流参数包括项目路径、数据源和目标目录、批次、样本、`peizhun_file`、预处理 FANWEI、`W_order`、`save_reg_only`、padding 尺寸、SIMcheck 检查项和时间点、重建 FANWEI、Dr 轨道、OTF、`just_cfai`、`loadflag`、设备、输出级别和写入许可。

### `r-mfsim`：完整 MF-SIM 工作流

这也是 Agent 编排型 Skill。需要提供科研项目，以及一个或多个已有的 120 帧 MP Dr 时间点：

```text
使用 $r-mfsim 处理项目 <project-root>。把 <source_0003.tif> 切分到
<sample>/Dr/01-488_padding_slice，然后运行匹配的项目 Run_MF 入口。
```

各阶段使用的 Skill、输入和输出如下：

| Stage | 使用的 Skill | 输入 | 输出 |
| --- | --- | --- | --- |
| Stage 1：Z 切片 | `mf-reconstruct` | 一个 `(120,H,W)` 的 MP Dr 时间点，即 8 Z × 3 方向 × 5 相位 | `MF-slice_z00.tif` 至 `MF-slice_z07.tif` 共 8 个 `(15,H,W)` TIFF |
| Stage 2：MF 重建 | `mf-reconstruct` | Stage 1 切片目录、项目 `Run_MF*.py` 入口、OTF 和匹配的 3D 光学参数 | 由项目入口指定目录中的 MF 参数、日志和重建结果 |
| Stage 3：分辨率测量（可选） | `decorr` 或 `fwhm` | MF 重建 TIFF；FWHM 还要求样本已确认是微球或细线 | `decorr-results/` 或 `FWHM-results/` 测量结果 |

`r-mfsim` 本身没有命令行参数。Agent 工作流参数包括项目路径、批次、样本、源 Dr 轨道、时间点、切片输出目录、覆盖策略、OTF、FANWEI、匹配的 3D 光学参数、`loadflag`、`just_cfai`，以及执行切片、重建或两者。

### `mp-data-ingest`：下载并整理原始数据

支持北大网盘分享、百度网盘分享、直接 URL 或本地路径：

```powershell
& $python -B "$repo\skills\mp-data-ingest\scripts\download_netdisk.py" `
  --url '<URL或本地路径>' `
  --dest '<原始数据日期目录>' `
  --name '<样本名>'
```

参数说明：

| 参数 | 必需 | 说明 |
| --- | --- | --- |
| `--url` | 是 | 分享链接、直接下载 URL 或本地文件/文件夹路径。 |
| `--dest` | 是 | 下载或复制到的目标日期目录。 |
| `--name` | 否 | `--dest` 下的样本目录名；省略时使用来源名称。 |
| `--password` | 否 | 分享提取码或密码。 |
| `--extract` | 否 | 下载后解压 ZIP、RAR 或 7Z；默认不解压。 |
| `--kind` | 否 | 强制来源类型：`auto`、`pku`、`baidu`、`direct` 或 `local`；默认 `auto`。 |

命令完成后会打印实际输出路径。文件夹型来源通常生成 `<原始数据日期目录>/<样本名>/`；不解压的直接文件会保留为 `<原始数据日期目录>` 下的文件。分享链接有提取码时增加 `--password`，直接下载压缩包并需要解压时增加 `--extract`。百度网盘分享要求 `PATH` 中存在已登录的 `BaiduPCS-Go`。

先预览原始数据布局变更，再显式执行：

```powershell
& $python -B "$repo\skills\mp-data-ingest\scripts\move_cam2_tiffs.py" `
  '<原始数据批次>'
& $python -B "$repo\skills\mp-data-ingest\scripts\move_cam2_tiffs.py" `
  '<原始数据批次>' --execute
```

参数说明：

| 参数 | 必需 | 说明 |
| --- | --- | --- |
| `data_root` | 是 | 待整理的原始批次根目录，只处理第一层样本目录。 |
| `--execute` | 否 | 实际执行重命名和移动；省略时仅预览计划。 |

该脚本会给第一层样本文件夹编号，并且只把各样本 `cam2/` 目录直属的 TIFF 移到对应样本目录。它不支持任意 A/B/C 数据路由。

### `mp-preprocess`：科研项目 real 入口与 padding

使用科研项目环境运行正式 real 预处理入口：

```powershell
& '<项目python>' -B `
  '<project-root>\code\real\<Run_real_MP入口.py>'
```

正式入口不是统一 CLI，参数保存在项目脚本中：

| 配置项 | 说明 |
| --- | --- |
| `path_input` | 规范化相机批次的输入路径。 |
| `peizhun_file` | 配准/校准数据路径。 |
| `FANWEI` | 本次处理的样本索引；必须先展示入口实际目录映射。 |
| `W_order` | 相机/方向校正顺序，必须与实验采集约定一致。 |
| `save_reg_only` | `0` 输出校正 A；`1` 还输出仅配准 B。重建只使用 A。 |

在项目入口中配置 `path_input`、`peizhun_file`、`FANWEI`、`W_order` 和 `save_reg_only`。典型输出为：

```text
<project-root>/code/real/dataOut/<batch>/<sample>/Dr/01-488/*.tif
```

对一个批次目录下所有匹配的 Dr TIFF 执行 padding：

```powershell
& $python -B "$repo\skills\mp-preprocess\scripts\expand_images.py" `
  '<project-root>\code\real\dataOut\<batch>' --x 462 --y 462 --z 120
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `root_dir` | 必填 | `dataOut` 下的批次根目录。 |
| `--x` | `462` | padding 后的 X 像素数。 |
| `--y` | `462` | padding 后的 Y 像素数。 |
| `--z` | `120` | padding 后的 Z/页数。 |
| `--suffix` | `_padding` | 输出轨道目录后缀。 |

工具会为每个源轨道写入带 `_padding` 后缀的同级轨道，例如 `Dr/01-488_padding/<同名文件>.tif`。已存在的输出文件会跳过。

### `simcheck`：对保留相位的数据执行 MCNR 与 FPJ

SIMcheck 必须明确提供采集维度和 TIFF 页序，不能仅凭页数推断页序。

处理一个预处理 TIFF：

```powershell
& $python -B "$repo\skills\simcheck\scripts\run_simcheck.py" `
  '<sample>\Dr\01-488_padding\sample_0003.tif' `
  --sim-mode 3d-mp-sim --n-z 8 --n-directions 3 --n-phases 7 `
  --page-order ZAP --checks mcnr,fpj --device auto
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input` | 必填 | 保留相位的 TIFF 或文件夹。 |
| `--sim-mode` | `auto` | `2d-sim`、`3d-sim`、`3d-mp-sim`、`4b-sim`、`generic` 或 `auto`；`auto` 只根据已提供的方向数和相位数选模式。 |
| `--n-z` | 必填 | 每通道、每时间点的 Z 层数。 |
| `--n-directions` | 必填 | 采集方向数。 |
| `--n-phases` | 必填 | 每方向相位数。 |
| `--n-channels` | `1` | 通道数。 |
| `--n-timepoints` | `1` | 每个 TIFF 的时间点数；当前兼容布局只接受 `1`。 |
| `--page-order` | 必填 | TIFF 页轴从慢到快的顺序，例如 `ZAP`；可用轴为 `C,T,Z,A,P`。 |
| `--output-root` | 自动 | 输出根目录；单文件默认写到其父目录的 `SIMcheck_V2/`。 |
| `--name-glob` | `*.tif*` | 文件夹输入时筛选文件名的 glob。 |
| `--recursive` | 关闭 | 递归搜索子目录；省略时只处理输入目录直属文件。 |
| `--checks` | `mcnr,fpj` | 要运行的检查，支持 `mcnr`、`fpj` 或二者。 |
| `--mcnr-z-half-window` | `1` | MCNR 计算使用的 Z 半窗口，必须不小于 0。 |
| `--device` | `auto` | `auto`、`cuda` 或 `cpu`。 |
| `--no-preview` | 关闭 | 不生成 PNG 预览。 |
| `--overwrite` | 关闭 | 允许覆盖既有输出，必须先获得用户确认。 |
| `--dry-run` | 关闭 | 只打印执行计划，不写结果。 |

直接输入 TIFF 且不指定 `--output-root` 时，结果写入该 TIFF 的父目录下：

```text
<sample>/Dr/01-488_padding/SIMcheck_V2/
├─ SIMcheck_V2_batch_MCNR.csv
├─ SIMcheck_V2_batch_MCNR.md
├─ SIMcheck_V2_report.md
├─ SIMcheck_V2_stats.csv
└─ raw/...
```

递归处理批次目录下匹配的 TIFF：

```powershell
& $python -B "$repo\skills\simcheck\scripts\run_simcheck.py" `
  '<project-root>\code\real\dataOut\<batch>' `
  --name-glob '*_0003.tif' --recursive `
  --sim-mode 3d-mp-sim --n-z 8 --n-directions 3 --n-phases 7 `
  --page-order ZAP --checks mcnr,fpj --device auto `
  --output-root '<批次结果目录>\SIMcheck_V2'
```

文件夹模式会保留每个源文件的相对父路径，并增加 TIFF 文件名作为一层目录：

```text
<批次结果目录>/SIMcheck_V2/<源文件相对父路径>/<源文件名>/...
```

省略 `--recursive` 时，只处理输入文件夹直属的 TIFF。大批量运行前先使用 `--dry-run`；只有用户确认覆盖时才使用 `--overwrite`。

### `raw-check`：从 cam1/cam2 计算初步 FPJ

传入一个直接包含可识别 cam1 和 cam2 TIFF 的目录：

```powershell
& $python -B "$repo\skills\raw-check\scripts\fpj_raw_cam12.py" `
  --input-dir '<raw-data-A>' `
  --n-z 8 --n-directions 3 --n-phases 5
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input-dir` | 无 | 自动查找 cam1/cam2 TIFF 的目录；不能与 `--cam1`、`--cam2` 同用。 |
| `--cam1` | 无 | 显式指定 cam1 原始 TIFF。 |
| `--cam2` | 无 | 显式指定 cam2 原始 TIFF 或其所在目录；与 `--cam1` 配套使用。 |
| `--output-root` | `<cam1父目录>/raw_fpj` | 输出目录。 |
| `--n-z` | `1` | 每时间点 Z 层数。 |
| `--n-directions` | `3` | 采集方向数。 |
| `--n-phases` | `5` | 每方向相位数。 |
| `--peak-orders` | 自动 | FPJ 标注的峰级数；省略时按采集模式决定。 |
| `--sample-evenly` | 关闭 | 在全部时间点中均匀选择最多 10 个；省略时取前 10 个。 |
| `--simcheck-dir` | 同仓库 `simcheck` | 覆盖 SIMcheck Skill 路径，仅在自动发现失败时使用。 |
| `--no-preview` | 关闭 | 只生成 TIFF，不生成 PNG 预览。 |
| `--overwrite` | 关闭 | 覆盖既有结果，必须先获得用户确认。 |

也可以改用 `--cam1 <cam1.tif> --cam2 <cam2.tif或文件夹>`。该命令不会递归处理多个样本目录。默认最多采样前 10 个时间点，并写入：

```text
<raw-data-A>/raw_fpj/
├─ region4_tif/FPJ_region4_T###.tif
├─ region5_tif/FPJ_region5_T###.tif
├─ region4_png/FPJ_region4_T###.png
└─ region5_png/FPJ_region5_T###.png
```

使用 `--sample-evenly` 可让采样时间点均匀分布；只有确实要替换既有结果时才使用 `--overwrite`。

### `mp-reconstruct`：运行科研项目 3D 重建入口

该 Skill 刻意不保存重建算法副本。使用科研项目环境运行所选入口：

```powershell
& '<项目python>' -B `
  '<project-root>\code\recon\<0-run5入口.py>'
```

正式入口不是统一 CLI，运行前需要确认这些项目配置：

| 配置项 | 说明 |
| --- | --- |
| 输入/输出路径 | 必须是本次实验的绝对路径，并保持批次身份一致。 |
| `FANWEI` | 要处理的样本索引；必须依据入口实际排序后的目录映射确认。 |
| OTF | 与采集和光学参数匹配的标定 OTF 路径。 |
| `Dr_sub` | 选择校正 A 下实际使用的 Dr 轨道，通常为 `01-488_padding`。 |
| `loadflag` | 项目定义的参数加载模式，必须按所选入口核对。 |
| `just_cfai` | `1` 只估计参数并保留 cfai 产物；`0` 才运行完整重建。 |
| 参数设备 | 参数估计使用的 CPU/GPU 设备，必须与项目环境兼容。 |
| 输出级别 | 控制日志、中间结果和完整重建体的保存范围。 |

运行前必须打印入口实际排序后的样本映射，并确认 `FANWEI`、绝对输入输出路径、OTF、`Dr_sub`、`loadflag`、`just_cfai`、设备和输出级别。输出位置由项目入口定义；典型产物包括参数文件、日志、`P/lihe_*.tif` 和可选的完整重建体。默认使用 `just_cfai=1`。

### `mf-reconstruct`：Z 切片与科研项目 MF 重建

切分一个 120 帧 Dr 时间点：

```powershell
& $python -B "$repo\skills\mf-reconstruct\scripts\slice_mp_dr_to_mf.py" `
  --src '<sample>\Dr\01-488_padding\sample_0003.tif' `
  --out-dir '<sample>\Dr\01-488_padding_slice'
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--src` | 必填 | 单个 MP Dr 时间点 TIFF，默认约定形状为 `(120,H,W)`。 |
| `--out-dir` | 必填 | 8 个 MF Z 切片的输出目录。 |
| `--n-z` | `8` | Z 层数。 |
| `--n-frames` | `15` | 每个 Z 切片的帧数，即 3 方向 × 5 相位。 |
| `--name-prefix` | `MF-slice` | 输出文件名前缀。 |
| `--overwrite` | 关闭 | 覆盖既有切片，必须先获得用户确认。 |

输出目录包含 `MF-slice_z00.tif` 至 `MF-slice_z07.tif` 共 8 个 `(15,H,W)` TIFF。之后单独运行正式项目 MF 入口：

```powershell
& '<项目python>' -B `
  '<project-root>\code\2D-MF-recon\<Run_MF入口.py>'
```

`Run_MF*.py` 不是统一 CLI，需确认：`FANWEI` 使用入口实际未排序的 `Path.iterdir()` 映射；dx、光学设置、notch、`mear`、W、`loadflag` 和 `just_cfai` 应与所选 3D 入口匹配；MF 固定使用 `flag_N=5`、`Nshifts=5`、`Nangles=3`。

运行前确认实际未排序的 `Path.iterdir()` FANWEI 映射，并与所选 3D 入口的物理参数保持一致。不要使用 `flag_N=3`；MF 使用 `flag_N=5`、`Nshifts=5` 和 `Nangles=3`。

### `fwhm`：微球或细线结构半高宽

处理已确认的 MP-SIM 微球数据：

```powershell
& "$repo\skills\fwhm\fwhm.cmd" beads '<微球TIFF或文件夹>' `
  --preset mp-sim-beads-v1
```

如果只需要同一批接近横向与轴向箱图中位数的微球 ROI：

```powershell
& "$repo\skills\fwhm\fwhm.cmd" beads '<微球TIFF或文件夹>' `
  --preset mp-sim-beads-v1 --median-nearby-count 5
```

处理已确认的细线结构：

```powershell
& "$repo\skills\fwhm\fwhm.cmd" lines '<细线TIFF或文件夹>' `
  --pixel-size-xy-nm 30.1 --pixel-size-z-nm 96.9 `
  --theoretical-lateral-resolution-nm 100 `
  --theoretical-axial-resolution-nm 290
```

`beads` 与 `lines` 的公共参数：

| 参数 | 说明 |
| --- | --- |
| `input` | TIFF 文件或目录。 |
| `--output` | 新结果目录；省略时自动创建带时间戳目录。 |
| `--pixel-size-xy-nm`、`--pixel-size-z-nm` | XY 每像素和 Z 每层的纳米采样值。 |
| `--axes` | `auto`、`YX` 或 `ZYX`；默认 `auto`。 |
| `--channel`、`--timepoint` | 多通道或多时间点数据中要分析的零基索引。 |
| `--recursive` | 递归处理输入目录。 |
| `--reuse-last` | 复用同一父目录最近确认的采样值，只能在用户确认连续批次后使用。 |
| `--review` | 测量结束后打开 ROI 审查器。 |
| `--no-open` | 不自动打开生成的报告或审查页面。 |

`beads` 专用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--preset` | `strict-single-bead-v1` | 微球检测预设；确认是 MP-SIM 微球时使用 `mp-sim-beads-v1`。 |
| `--exclude-z-start`、`--exclude-z-end` | 由预设决定 | 从开头/结尾排除的 Z 层数。 |
| `--max-candidates` | `500` | 最多分析的候选微球数。 |
| `--detection-snr` | `100` | 候选检测 SNR 阈值。 |
| `--min-fit-snr` | `5` | 接受拟合的最低 SNR。 |
| `--min-fit-r2` | `0.90` | 接受拟合的最低 R²。 |
| `--edge-margin-fraction` | `0.10` | 图像边缘排除比例。 |
| `--median-nearby-count` | `0` | 按横向/轴向箱图中位数联合排序，保留同一批 ROI 的数量；`0` 保留全部。 |

`lines` 专用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--theoretical-lateral-resolution-nm` | 必填 | 理论横向分辨率。 |
| `--theoretical-axial-resolution-nm` | 必填 | 理论轴向分辨率。 |
| `--exclude-z-start`、`--exclude-z-end` | 各 `1` | 从开头/结尾排除的 Z 层数。 |
| `--edge-margin-fraction` | `0` | 图像边缘排除比例。 |
| `--minimum-snr` | `5` | 接受细线 ROI 的最低 SNR。 |
| `--maximum-curvature-degrees` | `15` | 允许的最大局部弯曲角。 |
| `--maximum-tilt-degrees` | `10` | 允许的最大轴向倾斜角。 |
| `--maximum-rois-per-line` | `3` | 每条细线最多选取的 ROI 数。 |

文件夹输入默认不递归，只有增加 `--recursive` 才递归。默认输出到新建的带时间戳目录。文件夹输入时为：

```text
<输入文件夹父目录>/FWHM-results/<输入名称>-<时间戳>/
```

输入单个 TIFF 时为 `<输入文件父目录的父目录>/FWHM-results/<输入名称>-<时间戳>/`。

应先阅读结果中的 `SUMMARY.md`。手动检查已有结果的 ROI：

```powershell
& "$repo\skills\fwhm\fwhm.cmd" review '<FWHM结果目录>'
```

`review` 参数：`result` 是已有 FWHM 结果目录；`--no-open` 不自动打开浏览器；`--port` 指定本地审查器端口，默认 `0` 表示自动选择。

### `decorr`：去相关分辨率

在 CPU 上测量一个重建 TIFF：

```powershell
& "$repo\skills\decorr\decorr.cmd" measure '<重建结果.tif>' `
  --xy-pixel-nm 30.1 --z-pixel-nm 96.9 --axial-device cpu
```

`measure` 参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input` | 必填 | 单个 3D 重建 TIFF。 |
| `--output` | 自动 | 新结果目录；省略时在输入数据文件夹旁创建。 |
| `--xy-pixel-nm` | 必填 | XY 每像素纳米数。 |
| `--z-pixel-nm` | 无 | Z 每层纳米数；需要轴向测量时提供。 |
| `--seed` | `20250124` | 主去相关计算的随机种子。 |
| `--audit-seeds` | 空 | 额外审计种子列表，用于检查结果稳定性。 |
| `--axial-device` | `cpu` | 轴向计算设备：`cpu`、`cuda` 或 `auto`。 |

默认输出到新建的带时间戳目录：

```text
<输入文件夹父目录>/decorr-results/<输入文件名>-<时间戳>/
```

已有环境可能支持 CUDA、但允许回退到 CPU 时可使用 `--axial-device auto`。共享 UV 环境不安装 PyTorch。

正式 27 样本批处理使用单独的严格命令：

```powershell
& "$repo\skills\decorr\decorr.cmd" batch `
  --input-root '<正式27样本目录>' --axial-device auto
& "$repo\skills\decorr\decorr.cmd" render `
  '<正式27样本父目录>\decorr-results\<输入目录名>'
```

`batch` 参数：`--input-root` 是必填的正式 27 样本目录；`--output-root` 可指定结果根目录；`--audit-seeds` 提供额外审计种子；`--axial-device` 选择 `cpu`、`cuda` 或 `auto`，默认 `cpu`。`render` 只有一个必填位置参数 `output_root`，指向已完成计算的批次结果目录。

批处理结果包含 `samples/`、`formal_progress.csv` 和 `decorr_results.xlsx`；渲染后增加 `INDEX.md`、质控页面和论文图片。正式批处理器会拒绝不符合其精确 27 文件约定的输入集。

## 安装或更新

推荐检出目录为 `$env:USERPROFILE\.agents\sim-skills`，但仓库可以克隆到任意位置，不要写死完整用户目录。安装器从自身所在位置解析 Skill 来源，并在当前用户的 `.agents\skills` 下为每个 Skill 创建一个目录联接，使所有受支持 Agent 使用同一份 Git 管理文件。如果某个 Agent 使用其他共享 Skill 目录，可通过 `-TargetRoot` 指定。

```powershell
$repo = Join-Path $env:USERPROFILE '.agents\sim-skills'

if (Test-Path -LiteralPath (Join-Path $repo '.git')) {
    git -C $repo pull --ff-only
} else {
    gh repo clone gouwenct/sim-skills $repo
}

& (Join-Path $repo 'install.ps1')
```

GitHub 仓库是私有仓库，因此需要先登录 GitHub CLI。安装器对已经指向当前检出目录的联接是幂等的；默认不会替换真实目录或指向其他位置的联接。只有用户明确确认迁移后，才可备份冲突项并切换所有联接：

```powershell
& (Join-Path $repo 'install.ps1') -MigrateExisting
```

备份写入 `.agents/skills-backup/<时间戳>/`，安装器不会删除备份。可用 `-BackupRoot` 指定其他备份位置。

AI 安装器应执行同样的顺序：克隆或仅快进拉取、运行 `install.ps1`、报告冲突但不改动冲突项，最后验证 10 个联接都解析到当前检出目录。未经用户明确确认，不得使用 `-MigrateExisting` 或 `-CreateUv`。
