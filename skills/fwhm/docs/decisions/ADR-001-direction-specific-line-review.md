# ADR-001: 3D 线条按方向独立审核

## Status

Accepted

## Date

2026-07-31

## Context

`ZYX` 线条候选可产生两种证据：横向宽度来自局部线条法线上的 XY profile，轴向宽度来自同一 XY 位置的 Z profile。若横向审核显示多层投影，人工看到的结构可能不是实际参与拟合的像素；若 XY/Z 共用标记和决定，拒绝一个方向会错误影响另一个方向。

## Decision

- 从候选 Z profile 选择峰值层，横向测量和横向审核都只使用该单层。
- 分别生成 `candidates-XY.tif` 与 `candidates-Z.tif`，并保留 Z 维度和 ROI 序号。
- 分别保存 `review-assets/XY/` 与 `review-assets/Z/`。
- 使用方向特定决定和纳入字段；XY 与 Z 独立保留或删除。

## Alternatives Considered

### 多层投影用于横向审核

投影更容易看到附近结构，但会引入未参与横向拟合的 Z 层，因此不能作为该拟合的审核证据。

### 每个 ROI 共用一个审核决定

实现更简单，但无法表达“横向受邻线干扰、轴向仍合格”或相反情况。

## Consequences

- 审核图与实际拟合输入一致，候选可追溯到原始 Z 层。
- 同一 ROI 会产生最多两个审核项和两个决定。
- 下游统计必须分别读取 `reviewed_included_xy` 与 `reviewed_included_z`。
- 浏览器分页和增量加载可以调整，但不得改变方向键、证据内容或决定语义。

实现与回归证据见 [方向特定审核契约](../solutions/conventions/direction-specific-review-contracts-3d-line-fwhm.md)。
