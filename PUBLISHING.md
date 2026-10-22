# 发布步骤(公开仓 + Zenodo DOI)

## 0. 发布前检查

```bash
git status --short        # 应只看到源码/文档/图,不应有 data/ 或 .state/
```

确认:

- [x] 没有个人数据(`data/` 被 `.gitignore` 排除)
- [x] 没有本地绝对路径(README 里的路径都是占位符)
- [x] 有 `LICENSE`(GPL-3.0)和 `CITATION.cff`
- [x] 有 `.zenodo.json`(Zenodo 元数据)

## 1. 建公开仓并推送

仓库名建议 `manifold-dynamic-memory`(代码包名保持 `semantic_memory`,这是实现细节,
不冲突)。

GitHub 的 **About** 描述可以直接用:

```
Extractive, de-duplicated recall over a personal message archive: ordered passages with provenance and honest refusal, plus an MCP server. v0.1.0 is a lexical baseline; the manifold-based index is the research target.
```

```bash
git init
git add .
git commit -m "MDM v0.1.0: extractive recall over a message archive"
git branch -M main
git remote add origin https://github.com/<you>/manifold-dynamic-memory.git
git push -u origin main
```

## 2. 启用 Zenodo 集成(必须先于 release)

1. 用 GitHub 账号登录 <https://zenodo.org>;
2. 右上角头像 → **Settings** → **GitHub**;
3. 在仓库列表里找到 `manifold-dynamic-memory`,把开关打开(只有 public 仓会出现)。

## 3. 生成 DOI:建一个 GitHub Release

1. 仓库页 → **Releases** → **Draft a new release**;
2. Tag:`v0.1.0`,Target:`main`;
3. 标题:`v0.1.0 — first public prototype`;
4. 发布 → Zenodo 会自动归档并**生成 DOI**(形如 `10.5281/zenodo.1234567`),同时出现在 release 页面和 Zenodo 记录里。

## 4. 回填 DOI

拿到 DOI 后:

- 把 `README.md` 里 `10.5281/zenodo.XXXXXXX` 换成真实 DOI;
- 在 `CITATION.cff` 里加一行 `doi: 10.5281/zenodo.xxx`;
- 可选:在 README 顶部加徽章

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxx.svg)](https://doi.org/10.5281/zenodo.xxx)
```

## 5. 之后

- 每次想发新版本:改版本号 → 建新 release → Zenodo 生成**新 DOI**;
- 旧 DOI 永远指向旧版本,适合引用;
- 想引用"最新"可以用 Zenodo 的 concept DOI(同一概念下所有版本)。

## 提醒

公开仓意味着**内容永久可被检索与引用**。两个已知取舍:

1. 公开披露会**破坏专利新颖性**(如果以后想申请专利,必须在公开前申请);
2. 数据、私人对话、第三方 AI 输出**永远不要**进仓。
