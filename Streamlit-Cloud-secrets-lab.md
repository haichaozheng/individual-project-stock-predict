# W10 实验三：带密钥的真项目部署（本地 / 云端兼容写法）

> 给谁：第 3 节当堂起笔，课后完成。  
> 前提：已完成实验一、实验二——个人项目仓库已在课上推上 GitHub（还没推的先回 `github-lab.md` 第 3-4 节），并且已经部署过无密钥的最小应用。  
> 产出：你自己的 v0.4 在公网可访问，**AI / 取数功能在云端也正常**（本地能跑的，云端也能跑）。  
> 你的项目完全没用到密钥（不用 tushare / 大模型 API）？那跳过第 2、5 节，其余照做。  
> 全程约 30 分钟（前提：v0.4 本地已能跑；本地还有报错的先修好再来）。

---

## 0. 先搞懂问题：为什么本地能跑，云端 AI 就失效？

做个侦探式推理，三步：

1. 你本地的密钥放在 `.env` 文件里，代码用 `load_dotenv()` 读它——本地能跑；
2. 但 `.env` 在 `.gitignore` 名单上，**没有进 GitHub 仓库**（这是对的！密钥永不入库）；
3. Streamlit Cloud 那台云电脑**只能拿到仓库里的东西**——所以它拿不到 `.env`，代码里读密钥读了个空，AI 功能失效。

结论：**密钥必须用另一条通道单独交给云端**——这就是平台的 **Secrets**（加密的密钥仓库，存在 Streamlit Cloud 那边，不入 git、不进仓库，别人看不见）。

于是密钥的存放地点变成了两处，但代码只有一份：

| 环境 | 密钥放在哪 | 谁来读 |
|---|---|---|
| 本地 | `.env` 文件 | `load_dotenv()` + `os.getenv()`（W7 学的） |
| 云端 | 平台 Secrets | `st.secrets["键名"]` |

**本实验的核心，就是把代码改成「两条路都认」的兼容写法**——同一份代码，本地跑读 `.env`，云端跑读 Secrets，不用维护两个版本。

---

## 1. 部署前检查清单（2 分钟自查，消除 80% 的部署后报错）

对自己的**真项目**文件夹逐项打勾：

- [ ] `requirements.txt` 存在，且**用到的包都写全了**——生成方法：在已激活 `.venv` 的终端里执行 `python -m pip freeze > requirements.txt`（把当前环境全部包冻结成清单）；注意：用了 `from dotenv import load_dotenv` 的，清单里必须有 `python-dotenv`（这是本实验最容易漏的一个）
- [ ] `.env` 在 `.gitignore` 中，且 `git status` 里看不到它（`git status` 就是最快的安检门）
- [ ] 代码里无硬编码路径 `C:\...`（云端是 Linux 机器，Windows 路径一概无效；用 `pathlib.Path` 或相对路径）
- [ ] 数据文件 >50MB 的不进 git（改为启动时拉取，或先不上传）
- [ ] 密钥代码用 `os.getenv()` / `st.secrets` 读取（没有把 key 直接写在代码里——写在代码里 = 随仓库公开）

未通过的当场改，全过再往下走。

---

## 2. 改造代码：兼容本地 / 云端的密钥写法（本实验核心）

### 2.1 你现在的代码（多数同学 app.py 的顶部长这样）

```python
from dotenv import load_dotenv
import os

load_dotenv(override=True)  # 本地读 .env

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
```

问题：`load_dotenv` 只认本地的 `.env` 文件——云端没有这个文件，两行 `os.getenv` 全部返回 `None`。

> 你的代码和上面**略有出入**（比如 `load_dotenv` 里带了路径参数、变量名不同、只有一把密钥）都没关系——一会儿的替换范围是「从 `load_dotenv` 到 `os.getenv` 取密钥」的**那一整段**，照你自己的段对号入座。

### 2.2 改成兼容写法

把上面那段**整体替换**为：

```python
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv(override=True)  # 本地：读 .env；云端：没有这个文件，自动跳过，不报错

def get_key(name: str):
    """兼容写法：本地从 .env 读，云端从 Streamlit Secrets 读——同一份代码两处都能跑"""
    value = os.getenv(name)        # ① 先试本地：.env 已被 load_dotenv 装进环境变量
    if value:
        return value
    try:
        return st.secrets[name]    # ② 再试云端：从平台 Secrets 读
    except Exception:
        return None                # ③ 两处都没有 → 返回 None，交给下面的报错逻辑提示

TUSHARE_TOKEN = get_key("TUSHARE_TOKEN")
DEEPSEEK_API_KEY = get_key("DEEPSEEK_API_KEY")
```

后面的代码**一行都不用改**——`TUSHARE_TOKEN`、`DEEPSEEK_API_KEY` 这两个变量名和原来完全一样，只是取值的地方多了一条云端通道。

### 2.3 这段代码为什么这样写（三问三答）

**问：为什么不干脆全用** `st.secrets`**？**
答：`st.secrets` 在本地要求你另建一份 `.streamlit/secrets.toml` 文件——等于本地搞一套、云端一套，两处维护。而课程从 W7 起的约定就是 `.env` 存密钥，兼容写法让本地工作流**保持不变**。

**问：为什么** `os.getenv` **在前、**`st.secrets` **在后？**
答：本地跑的时候第 ① 步就命中，根本不会碰到第 ② 步；云端跑的时候 `.env` 不存在，第 ① 步落空，第 ② 步命中。顺序无所谓对错，但「本地优先」让你断网调试时不依赖任何云端配置。

**问：**`try/except` **那段在防什么？**
答：防「两边都没有」的情况——比如本地没有 `.streamlit/secrets.toml`，`st.secrets[name]` 会直接抛异常（不是返回 None）。兜住之后返回 `None`，你原来代码里的「密钥缺失 → 页面显示友好提示」逻辑（W7 写过）就能正常接住，而不是整个应用崩掉。

### 2.4 顺手加一个「体检开关」（排障神器，强烈推荐）

在读取密钥的代码后面加几行（**注意：显示的是"有没有配好"，绝不显示密钥本身的值**）：

```python
with st.sidebar.expander("🔧 密钥体检（部署排障用）"):
    st.write("TUSHARE_TOKEN：", "已配置" if TUSHARE_TOKEN else "❌ 缺失")
    st.write("DEEPSEEK_API_KEY：", "已配置" if DEEPSEEK_API_KEY else "❌ 缺失")
```

部署后如果 AI 功能失效，先展开侧栏看体检：哪项 ❌ 就是哪条通道断了——是 Secrets 没配、还是键名打错了，一眼定位。排障完可以留着（对用户无害）也可以删掉再 push。

### 2.5 检查 requirements.txt

兼容写法用到了 `dotenv` 和 `streamlit`，确认清单里有（缺哪个补哪个，补完记得 push）：

```text
streamlit
python-dotenv

#（以及你项目用到的其他包：pandas、plotly、tushare、openai……）
```

---

## 3. 本地回归验证：别把「能跑」改坏了

改完先本地跑一遍（改代码 → 本地验证 → 再 push，这个顺序能省你大量排障时间）。

在**你真项目的文件夹**打开终端（实验二第 2 节的老办法：资源管理器进入项目文件夹 → 地址栏输 `powershell` → 回车），激活虚拟环境——`.venv` 建过的直接激活，没建过的先建（命令与实验二完全相同）：

```text
.\.venv\Scripts\activate
```

然后启动应用：

```text
streamlit run app.py
```

本地跑**应该和改造前一模一样**（因为走的还是 `.env` 那条路）——展开侧栏「密钥体检」，两项都应是「已配置」。确认无误，`Ctrl + C` 停止。

---

## 4. 推上 GitHub（第 4 次日常循环）

```text
git add .
git commit -m "改造：密钥兼容本地 .env 与云端 Secrets"
git push
```

---

## 5. 云端配置 Secrets

现在代码已经会「从云端 Secrets 读密钥」了，但云端还没收到密钥——这一步就是**把密钥安全地交给 Streamlit Cloud**。

### 5.1 找到配置入口

1. 打开 <https://share.streamlit.io/>（已登录）；
2. 部署真项目：如果实验二里只部署过练手项目，现在再点 **New app** → 选你真项目的仓库 / `main` 分支 / 入口文件（W7 模板默认是 `app.py`；你改过文件名的就填实际那个，以代码里跑 `streamlit run` 的那个文件为准）→ Deploy（流程与实验二完全相同）；
3. 打开你的应用页面，点**右下角的 Manage app** → 齿轮 **Settings** → **Secrets** 标签。

（已部署过的应用也可以随时回来改：列表页 → 应用右下角 `⋮` → Settings → Secrets。）

### 5.2 填入密钥（TOML 格式）

在 Secrets 文本框里**逐行**填（键名要和代码里 `get_key("...")` 的字符串**完全一致，区分大小写**）：

```text
TUSHARE_TOKEN = "你的tushare令牌"
DEEPSEEK_API_KEY = "你的deepseek密钥"
```

> ⚠️ **大小写是高频翻车点**：W7 的部分材料里出现过 `Deepseek_API_KEY` 这种混合大小写——如果你的 `.env` 和代码里用的就是它，Secrets 里也必须**一字不差照抄你自己的键名**，别照搬本文档的 `DEEPSEEK_API_KEY`。拿不准就打开本地 `.env` 看第一眼：等号左边是什么，Secrets 里就写什么。

三个格式雷区（TOML 格式，错一个字符就解析失败）：

- 每行都是 `键名 = "值"`——**等号**、**英文双引号**一个都不能少；
- 值两头的引号是英文引号 `"..."`，不是中文引号 `"..."`；
- 不要多余的缩进、不要在行尾加分号或逗号。

点 **Save**。Secrets 只存在 Streamlit Cloud 那边——**不进仓库、不入 git**，与「`.env` 永不入库」是同一条铁律的云端版。

### 5.3 让它生效

改完 Secrets 后应用可能需要重启：Manage app 菜单里点 **Reboot**（或等几分钟自动重启），然后刷新页面。

---

## 6. 验收

1. 展开侧栏「密钥体检」：两项都是「已配置」；
2. 实际用一次 AI / 取数功能：**云端跑出了和本地一样的结果**——恭喜，你的真项目完整上云了；
3. 手机（用流量）打开 URL 再验一次；
4. 把 URL 发到课程群——这是 W11 触达的起点。

---

## 7. 常见问题速查

| 症状 | 原因 | 解法 |
|---|---|---|
| New app 的下拉列表里找不到自己**真项目**的仓库 | 真项目仓库是课后自建的，建仓时选了 Private，或 Streamlit 授权没给全 | 解法同实验二第 6 节第 1 条：GitHub 仓库 Settings → Danger Zone → Change visibility 改 Public；或 Streamlit 授权页重新授权 All repositories |
| AI 功能失效，体检显示 ❌ 缺失 | Secrets 没配 / 键名不一致 / TOML 格式错 | 回第 5 节逐项核对：键名大小写、等号、英文双引号；改完 **Reboot** |
| 云端报 `ModuleNotFoundError: No module named 'dotenv'` | requirements.txt 漏了 `python-dotenv` | 清单补上 → add → commit → push（自动重新部署） |
| 云端报 `ModuleNotFoundError`（其他包） | 清单漏包 / 没提交 | 同上；用得全就 `pip freeze` 全量冻结 |
| 本地能跑，云端报找不到数据文件 | 路径写死 `C:\...`，或大文件没进仓库 | 改 `pathlib.Path` 相对路径；大文件改启动时拉取 |
| 改了 Secrets 没生效 | 应用没重启 | Manage app → Reboot，再刷新 |
| 体检已配置但功能仍报错 | 密钥本身的问题（欠费 / 失效） | 本地用同一份 `.env` 跑一次对比——本地也报错就不是部署问题，是密钥问题 |

---

## 8. 收尾

- URL 已发课程群；W11 当堂把链接发班级群/社区，反馈滚到 W14；
- 当日在 `journal.md` 追加 1 条：今天做了什么 / 关键决定（尤其记下：密钥管理为什么是「本地 .env + 云端 Secrets」双通道）/ 学习并掌握的新东西 / 卡点 / 互助记录（没有则写「无」）；
- 一句话带走本周：**部署就三件事——依赖（requirements.txt）、密钥（.env → Secrets）、数据（随仓库 / 现场拉取）；代码用兼容写法，一份两头跑。**
