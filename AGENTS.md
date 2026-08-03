# AGENTS.md

## 文档语言

- 根目录 `README.md` 的说明文字统一使用简体中文。
- 命令、代码标识符、文件名、参数名和必须保留的专业英文术语可以保持原文。
- 新增或修改 README 内容时，不得重新引入成段英文说明。

## Skill 更新与 README 同步

每次新增、删除、重命名或修改 `skills/<skill-name>/` 下任何内容时，必须：

1. 完整阅读该 Skill 的 `SKILL.md`。
2. 检查相关 `scripts/`、`references/`、测试和 `agents/openai.yaml`。
3. 重新判断该 Skill 的实际触发场景、输入、输出、递归行为、默认路径、覆盖策略、环境依赖和科研项目边界。
4. 通过真实 `--help`、代码或测试确认用法，不得只沿用旧 README。
5. 重写或更新根目录 README 中该 Skill 的手动操作章节和示例。
6. 即使用法结论未改变，也必须确认 README 对应章节仍与实现一致。
7. Skill 与 README 的修改应在同一提交中完成。

## 验证

- 检查 README 中引用的脚本真实存在。
- 检查 Markdown 代码围栏成对。
- 修改 Skill 后运行对应结构校验和相关测试。
- 不得把项目正式入口或算法副本重新放入 Skill；real、MP 重建、MF 重建的正式入口仍属于科研项目。
