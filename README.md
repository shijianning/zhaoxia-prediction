# 朝霞晚霞每日预测

一个本地运行的 Python 项目：**每天自动预测你所在城市的朝霞/晚霞出现概率**，并**利用网络上的朝霞晚霞分享帖子作为训练标签，不断训练模型、提高准确率**。

## 原理

火烧云（朝霞/晚霞）本质是气象现象，能否出霞主要取决于日出/日落时段的：

| 因素 | 利于出霞的条件 |
|------|--------------|
| 云层结构 | 中高云（卷云、高积云）适中，承接红光；太低太厚则遮挡 |
| 低云 | 不能太厚，否则挡住地平线 |
| 湿度 | 适中（约 50-60%），过高成雾霾、过低平淡 |
| 降水 | 无雨 |
| 风速 | 适中，静风偏灰、大风扬尘 |
| 大气通透度(AOD) | 气溶胶光学厚度越低天空越通透、火烧云越鲜艳（参考 SunsetBot 核心指标） |
| 水平能见度 | 越高低空霾/雾越少、红橙光衰减越少、颜色越艳（吸收 sunset-prediction 能见度因子） |

流程：

```
拉取气象数据(Open-Meteo 免费)
        │
        ├── 未来预报 → 特征工程 → 规则评分 + ML 概率 → 融合评级 → HTML 报告
        │
        └── 历史数据 → 特征工程 → 组装标签(帖子观测 + 坏天气负样本 + 弱监督)
                                        │
                                        └── 训练分类器 → 保存模型
```

## 快速开始

1. 安装 Python 3.9+，然后安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行（训练 + 预测 + 生成报告）：

```bash
python daily_run.py
```

运行后会在 `output/report.html` 生成网页报告，用浏览器打开即可查看；同时终端打印摘要。

只想预测、跳过训练：

```bash
python daily_run.py --predict-only
```

## 全国预测地图

既然气象规则引擎与地点无关，本项目支持对**全国城市**一次性预测，生成一张中国地图形式的朝霞/晚霞预测地图：

```bash
python national_map.py              # 预测 data/cities.json 中全部城市，生成 output/national_map.html
python national_map.py --cities 10  # 只跑前 10 个城市（调试）
python national_map.py --workers 8  # 指定并发数
```

地图说明：

- 使用**腾讯地图 GL JS**（合规代理模式，前端不携带 key，经 WorkBuddy 本地代理），坐标已由 WGS-84 转为 GCJ-02。
- 每个城市一个彩色圆点，颜色/数字表示出霞评分 0–100；支持切换「今天/明天」和「朝霞/晚霞」，点击城市查看详情。
- 支持**散点 / 热力**两种视图切换（工具栏「视图」按钮）：热力图用颜色深浅连续呈现全国出霞强度分布，颜色体系与散点一致（灰→黄→橙→红 = 平淡→世纪大烧）。
- 省会城市带名称标注。全国地图使用「气象规则引擎」评分（与地点无关）；西安单城报告才额外叠加本地 ML 模型。
- `data/cities.json` 内置 31 个省级行政中心 + 约 70 个重点城市，可自行增删。

## 配置

编辑 `config.yaml`：

- `city`：城市名 + 经纬度 + 时区（换城市只需改这里）
- `forecast.days`：预测未来几天
- `model.type`：`logistic` 或 `rf`
- `data.history_days`：训练回溯的历史天数
- `third_party.geovisearth`：星图云官方火烧云预报 API 的 token 占位（见下）
- `database`：本地 MySQL 连接（host/port/user/password/database）
- `notify`：微信推送（Server酱），见下「微信推送」

## 本地 MySQL 存储

预测历史、城市列表、训练标注会自动写入本地 MySQL（库名 `zhaoxia`），MySQL 未启动时自动回退到文件存储、不影响预测。

**表结构**：

| 表 | 内容 | 唯一键 |
|----|------|--------|
| `cities` | 城市名 + 经纬度 + 是否省会 | `name` |
| `predictions` | 每日朝霞/晚霞预测历史（城市、日期、时段、评分、鲜艳度、AOD…） | `(city, date, window)` |
| `posts` | 训练观测标注 | `(date, window)` |

**启动/停止 MySQL**（免安装版，数据目录 `C:\Users\21134\mysql`）：

```bash
# 双击运行，或命令行：
C:\Users\21134\mysql\db_start.bat   # 启动（最小化窗口运行，勿关）
C:\Users\21134\mysql\db_stop.bat    # 停止
```

运行 `python daily_run.py` 或 `python national_map.py` 时会自动建库建表、写入预测结果。

**查询数据**（示例）：

```bash
"C:\Users\21134\mysql\bin\mysql.exe" -u root zhaoxia -e "SELECT city,date,window,score,vivid FROM predictions WHERE date=CURDATE() ORDER BY score DESC LIMIT 10;"
```

也可以用 Navicat / DBeaver / DataGrip 连接 `127.0.0.1:3306`（用户 root，密码空）。

## 如何用帖子训练模型（提高准确率）

模型准确率取决于标签质量。项目提供三种标签来源，按可靠性排序：

### 1. 手动/导出观测标注（最可靠）

编辑 `data/raw/posts.csv`，每行一条观测：

```csv
date,window,glow,source,note
2026-09-18,evening,1,weibo,西安晚霞刷屏
2026-09-19,morning,0,manual,早上阴天没看到
```

- `date`：日期 `YYYY-MM-DD`
- `window`：`morning`(朝霞) / `evening`(晚霞)
- `glow`：`1` 看到了好霞，`0` 没看到/平淡
- `source`：来源（weibo / xiaohongshu / manual ...）
- `note`：备注

你平时刷到「西安晚霞」「西安朝霞」「火烧云」的帖子，随手记一条即可，模型会越练越准。

### 2. 文本关键词自动标注（半自动）

`src/posts.py` 里的 `infer_label_from_text()` 可从帖子正文自动推断是否出霞。你可以把爬到的帖子正文按日期归档后，用它批量生成标注再写入 `posts.csv`。

### 3. 坏天气自动负样本（全自动）

雨天/全阴天基本不可能出霞，训练时会自动把这些天标记为负例，无需手动标注。

> 说明：微博/小红书等平台对爬虫有反爬限制，直接自动抓取不稳定。因此本项目采用「干净的标注格式 + 手动/导出补充」作为可靠路径；自动抓取建议用平台官方开放接口或你自己的 Cookie 定时导出，再写入 `posts.csv`。

## 交叉验证（多气象模型对比 + 多源集成）

单靠一个气象模型容易「自说自话」。本项目额外拉取三个来自不同国家气象机构的独立模式，对同一地点、同一日出/日落窗口分别评分，互相印证：

| 模式 | 机构 |
|------|------|
| GFS | 美国 NOAA |
| ICON | 德国 DWD |
| GEM | 加拿大 CMC |

报告里每个窗口会显示三个模型的评分、分差和置信度：

- 分差 ≤ 10 → 置信度「高」，结论可信；
- 分差 ≤ 25 → 置信度「中」；
- 分差 > 25 → 置信度「低」，该时段云况不稳定，预报仅供参考。

此外，最终评分采用**多源集成**：把「规则分 + ML 概率」的基础分与三模型交叉验证均值按 **7 : 3** 加权融合，让多个独立气象模式的结论直接参与打分（结论一致时更可信，分歧大时自动拉低极端值）。

> 关于「对比第三方工具（SunsetBot / 莉景天气等）」：
> - 这些工具都是小程序/App，**没有公开 API**，用搜索引擎也拿不到它们每天的结构化预测数值（只能搜到介绍文章和截图），无法自动化对比。
> - 但本项目已**吸收 SunsetBot 的核心方法论**——「气溶胶光学厚度(AOD) + 鲜艳度分级」，用 Open-Meteo 空气质量接口免费拿到同类 AOD 数据，把「会不会出霞」升级为「出了霞有多鲜艳」，并对齐了「微烧/小烧/中烧/大烧」摄影圈分级。
> - 若想接入真正的第三方数值对比，**星图云(GeoVisEarth)开放平台**提供官方「火烧云预报 API」（返回朝霞/晚霞质量与等级，未来 3 天）。已在本项目预留接入点（见下方「星图云第三方对比」）。

### 星图云第三方对比（第 4 方，可选）

在 `config.yaml` 的 `third_party.geovisearth` 填入 token 后即可把星图云官方预报作为第 4 个对比源：

1. 到 <https://open.geovisearth.com> 注册并完成开发者认证，在「火烧云预报 API」下单/开通；
2. 在控制台创建应用、获取 token；
3. 编辑 `config.yaml`：

```yaml
third_party:
  geovisearth:
    enabled: true
    token: "你的token"
    base_url: "https://api.open.geovisearth.com/v2/glow/fc/idxV2"
    productCode: "..."   # 控制台开通接口后获取
    dataCode: "..."      # 控制台开通接口后获取
    meteCode: "..."      # 控制台开通接口后获取
```

接入点位于 `src/crosscheck.py` 的 `run_geovisearth_crosscheck()`（已预留调用桩与解析 TODO），主预测 `src/predict.py` 会自动挂载，报告里每个窗口会显示「🌐 星图云官方预报」一行。

## 每日天气概览

单城报告的每个日期卡片顶部会显示当天天气概览：**天气现象（☀️晴/⛅多云/☁️阴/🌧️雨…）+ 最高最低温 + 日出日落时间 + 全天降水概率与降水量**。这些数据来自 Open-Meteo 的逐日字段（`weather_code` / `temperature_2m_max/min` / `sunrise` / `sunset` / `precipitation_sum` / `precipitation_probability_max`），与朝霞晚霞评分共用同一次请求，无额外开销。

## 鲜艳度指数（0-10）

除 0-100 评分和「微烧/小烧/中烧/大烧/优质大烧/世纪大烧」分级外，每个窗口还会输出一个 **0-10 鲜艳度指数**（🔥，对齐 chromasky 的 ChromaSky™ 指数），便于跨城市、跨日期横向对比。全国地图也用同一评分体系统一色阶。

## 分享卡片

单城报告右上角有「**📤 生成分享卡片**」按钮，点击后用前端 Canvas 把当天**最佳出霞机会**画成一张可保存的分享图（城市 + 日期/时段 + 评分/分级/🔥指数 + 关键气象因素 + 一句话提示），右下角「下载图片」保存为 PNG，手机端长按也可直接保存转发（融合 weather-sunset-predictor 的分享卡片能力）。

## 微信推送（Server酱）

当预测出现达到阈值的「火烧云」窗口时，可自动推送到微信（融合 sunsetbot / ohyep-sunsetglow 的推送能力）：

1. 到 <https://sct.ftqq.com> 用微信登录，获取你的 **SendKey**（`SCT` 开头）；
2. 复制 `config.local.yaml.example` 为 `config.local.yaml`，填入 SendKey（**不要填进 `config.yaml`**——那是会提交到公开仓库的文件，密钥放这里会泄露；`config.local.yaml` 已被 `.gitignore` 排除）：

```yaml
# config.local.yaml
notify:
  enabled: true
  sckey: "SCT你的SendKey"
  threshold: 70    # 预测分达到该值（对应"大烧"）才推送
```

3. 运行 `python daily_run.py`，若当天最佳窗口达到阈值，会自动收到微信消息（标题 + 日期/时段/评分/鲜艳度指数）。未配置 `sckey` 时静默跳过，不影响预测。免费版每天 5 条推送。

## 每天自动运行

### 方式 A：Windows 计划任务

在「任务计划程序」新建任务，操作设为运行 `run_daily.bat`，触发器设为每天固定时间（建议早晨 6:30 之前跑，能预测当天朝霞）。

或用命令行（PowerShell，管理员）：

```powershell
schtasks /Create /TN "ZhaoxiaPrediction" /TR "\"<项目路径>\run_daily.bat\"" /SC DAILY /ST 06:30
```

把 `<项目路径>` 换成本项目实际路径。注意 `run_daily.bat` 里的 `python` 需在 PATH 中，否则改成完整路径。

### 方式 B：WorkBuddy 定时自动化

也可以在 WorkBuddy 里建一个每日自动化，让它每天执行 `python daily_run.py`。

## 目录结构

```
zhaoxia-prediction/
├── config.yaml           # 配置
├── daily_run.py          # 每日入口（单城训练+预测+报告）
├── national_map.py       # 全国预测地图入口
├── run_daily.bat         # Windows 一键运行
├── requirements.txt
├── data/
│   ├── cities.json       # 全国城市列表（WGS-84）
│   ├── raw/posts.csv     # 观测标注数据
│   └── model/            # 训练好的模型(自动生成)
├── src/
│   ├── weather.py        # 天气数据获取
│   ├── features.py       # 特征工程
│   ├── glow_rules.py     # 规则评分
│   ├── model.py          # ML 模型
│   ├── posts.py          # 帖子数据接入
│   ├── crosscheck.py     # 多气象模型 + 星图云交叉验证
│   ├── train.py          # 训练
│   ├── predict.py        # 预测
│   ├── database.py       # MySQL 存储层
│   ├── notify.py         # 微信推送(Server酱)
│   ├── national_map.py   # 全国地图预测/渲染
│   └── report.py         # HTML 报告
└── output/
    ├── report.html       # 单城报告
    └── national_map.html # 全国地图
```

## 部署到自己的服务器

项目可部署到任意 Linux/Windows 服务器（能访问 Open-Meteo 即可）。以 Windows 服务器为例。

### 全自动脚本（推荐）

服务器（Windows）上**以管理员身份**双击 `setup.bat`，即可全自动完成：自动下载并安装 Python → 装依赖 → 首次生成报告和全国地图 → 启动网页服务 → 注册定时任务。全程无需手动装任何东西。

```bat
setup.bat         # 全自动：装 Python + 依赖 + 首跑 + 网页服务 + 定时任务
deploy.bat        # （备选）假设已装好 Python 的一键部署
start_server.bat  # 手动重启网页服务
```

自动注册的两个计划任务（`setup.bat` 会创建）：

- `ZhaoxiaDaily`：每天 06:30 自动预测 + 微信推送；
- `ZhaoxiaWeb`：开机自动启动网页服务（端口 8080）。

> 依赖下载走国内镜像（华为云 Python + 清华 pip），服务器在国内也能快速完成。

### 一键脚本（已装 Python）

部署完成后浏览器访问：

- 入口页：`http://服务器IP:8080/`
- 单城报告：`http://服务器IP:8080/report.html`
- 全国地图：`http://服务器IP:8080/national_map.html`

> 外网访问需在 Windows 防火墙放行 8080 端口。微信推送需要在服务器上另建 `config.local.yaml`（填入你的 SendKey，该文件不入库、git clone 不会带上）。

### 手动步骤

1. **装 Python 3.9+**，`pip install -r requirements.txt`；
2. **上传项目**（`git clone https://github.com/shijianning/zhaoxia-prediction.git` 或直接打包上传）；
3. **配置**：
   - `config.yaml`：改 `city` 为你所在城市；如需全国地图，填 `map.tmap_key`（见下）；
   - `config.local.yaml`：填 `notify.sckey`（微信推送，可选）；
4. **跑一次验证**：`python daily_run.py --predict-only`，确认生成 `output/report.html`；
5. **托管网页**：用 IIS / nginx / Caddy 把 `output/` 目录挂出去，即可通过浏览器访问报告；
6. **定时自动跑**：Windows「任务计划程序」每天早晨跑 `daily_run.py`（会自动推微信）。

**全国地图在服务器上**：需要申请腾讯地图 key（免费）：

1. 到 <https://lbs.qq.com> 注册并完成实名/开发者认证；
2. 控制台「应用管理」创建应用 → 添加 key（勾选「WebServiceAPI / 地图 JavaScript API」）；
3. 把 key 填入 `config.yaml` 的 `map.tmap_key`；
4. 在 key 设置里把**服务器域名/IP 加进「域名白名单」**（否则地图被拒）；
5. 服务器上跑 `python national_map.py`，生成的 `output/national_map.html` 即用你自己的 key 加载地图。

> 本地开发时 `map.tmap_key` 留空即可，会自动走 WorkBuddy 代理模式（前端不带 key）；填入 key 后自动切换为标准模式。

## 局限

- 朝霞/晚霞受局地云况影响大，预报是概率性的，仅供参考。
- 模型冷启动阶段样本少，会先用规则评分兜底，随标注增加逐步切换到 ML。
