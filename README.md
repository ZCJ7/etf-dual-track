# 左右侧 ETF 决策看板

用 AKShare 的公开行情，按左右侧双轨买卖点，把 ETF 分成两栏对照。

- 右侧：顺势。比价线、这只 ETF 自己的周线 MA20 与 MACD、日线 BIAS 与 MACD、量比一起看。大盘周线 MA20、MA60 只作对照，不拦截开仓。
- 同一标的只留成交额最高的一只，例如中证1000只出现一家公司的 ETF。
- 左侧：超跌。先要有支撑，再看周线 BIAS6 低于 -4%、日线绿柱缩短、量能，以及该 ETF 自己的周线 MA60 是否拐头。
- 沪深300的周线 MA20、MA60 只显示在页头作对照，不决定能不能开仓。

货币和债券 ETF 默认不进扫描池。宽基、行业主题、商品、跨境都会进，按成交额从高到低取前 N 只。

## 运行

```powershell
cd C:\Users\92594\etf-dual-track
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

本机打开 http://127.0.0.1:8765 ，点「更新今日数据」。第一次会下载历史日线，之后只补最近的 K 线。交易日 16:00 之后如果当天还没更新过，打开页面会自动补一次。

手机在电脑关机时打开 GitHub Pages：https://zcj7.github.io/etf-dual-track/ 。页面只含 ETF。工作日北京时间 14:30 和 16:40 由 GitHub 自动拉新数据。14:30 没成功时，14:45 再补跑一次。

行情调用的是 AKShare 里的 `fund_etf_category_sina`、`fund_etf_hist_sina` 和 `stock_zh_index_daily`。新浪日线不做前复权。若本机开着失效的系统代理，程序会改走直连。

这是收盘后的决策对照，不是下单指令。
