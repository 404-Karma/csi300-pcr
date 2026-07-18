#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沪深300 PCR (Put-Call Ratio) 与指数趋势比较工具
================================================
数据源（按可靠性排序）：
1. 中金所 CFFEX 沪深300股指期权 (IO) —— 与指数直接挂钩
   - 日行情 XML: http://www.cffex.com.cn/sj/hqsj/rtj/{YYYYMM}/{DD}/index.xml
2. 上交所 510300 沪深300ETF 期权
   - akshare.option_daily_stats_sse(date)
3. 深交所 159919 沪深300ETF 期权
   - akshare.option_daily_stats_szse(date)
4. 沪深300指数日线
   - akshare.index_zh_a_hist(symbol='000300', ...)

输出：
- CSV: csi300_pcr_output/csi300_pcr_index.csv
- 交互式图表: csi300_pcr_output/csi300_pcr_chart.html（ECharts，无需安装额外库）
- 若已安装 matplotlib，额外生成: csi300_pcr_output/csi300_pcr_chart.png
- 统计摘要: csi300_pcr_output/csi300_pcr_summary.txt
"""

import os
import re
import time
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "csi300_pcr_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRADE_DAYS_CACHE_FILE = os.path.join(OUTPUT_DIR, ".trade_days_cache.json")
PCR_DAILY_CACHE_FILE = os.path.join(OUTPUT_DIR, ".pcr_daily_cache.json")

# 最近交易日偏移：盘中收盘前数据可能未更新，默认取前一个交易日
END_OFFSET_DAYS = 0

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def get_trade_days(start_date: str, end_date: str) -> List[str]:
    """获取区间内的所有自然日，后续按数据源实际有数据的天数过滤。"""
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    days = []
    while start <= end:
        days.append(start.strftime("%Y%m%d"))
        start += timedelta(days=1)
    return days


def load_cache() -> Dict:
    if os.path.exists(TRADE_DAYS_CACHE_FILE):
        with open(TRADE_DAYS_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict):
    with open(TRADE_DAYS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_pcr_cache() -> Dict[str, Dict]:
    if os.path.exists(PCR_DAILY_CACHE_FILE):
        with open(PCR_DAILY_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_pcr_cache(cache: Dict[str, Dict]):
    with open(PCR_DAILY_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, default=str)


def is_trading_day(date_str: str, cache: Dict) -> bool:
    """简单判断是否为交易日：指数数据接口有数据即为交易日。"""
    if date_str in cache:
        return cache[date_str]
    return None


# ---------------------------------------------------------------------------
# 1. 沪深300指数
# ---------------------------------------------------------------------------

def fetch_csi300_index(start_date: str, end_date: str) -> pd.DataFrame:
    """获取沪深300指数日线（前复权等同指数本身），带 fallback。"""
    import akshare as ak
    df = None
    errors = []
    # 1) Eastmoney
    try:
        df = ak.index_zh_a_hist(
            symbol="000300",
            period="daily",
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        errors.append(f"index_zh_a_hist: {e}")
    # 2) Sina 日线
    if df is None or df.empty:
        try:
            df = ak.stock_zh_index_daily(symbol="sh000300")
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        except Exception as e:
            errors.append(f"stock_zh_index_daily: {e}")
    if df is None or df.empty:
        raise RuntimeError(f"未能获取沪深300指数日线数据: {'; '.join(errors)}")
    df = df.copy()

    # 统一列名：东财来源为中文，新浪来源为英文
    rename_map = {
        "日期": "date", "date": "date",
        "开盘": "open", "open": "open",
        "最高": "high", "high": "high",
        "最低": "low", "low": "low",
        "收盘": "close", "close": "close",
        "成交量": "volume", "volume": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "change_pct",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    # 只重命名实际存在的列
    existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=existing_rename, inplace=True)

    # 确保必要列存在
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise RuntimeError(f"指数数据缺少必要列: {col}")
    for col in ["amount", "amplitude", "change_pct", "change", "turnover"]:
        if col not in df.columns:
            df[col] = np.nan

    # 补全可计算的字段
    if df["change"].isna().all():
        df["change"] = df["close"] - df["close"].shift(1)
    if df["change_pct"].isna().all():
        df["change_pct"] = df["close"].pct_change() * 100
    if df["amplitude"].isna().all():
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. 中金所 CFFEX IO 期权 PCR
# ---------------------------------------------------------------------------

def _extract_io_expiry(instrument: str) -> Optional[str]:
    """从合约代码提取到期月份，如 IO2507-C-3900 -> 2507。"""
    if not instrument or not instrument.startswith("IO"):
        return None
    # IO2507-C-3900 -> 2507
    expiry = instrument[2:6]
    if expiry.isdigit():
        return expiry
    return None


def fetch_cffex_option_daily(date_str: str, max_retries: int = 2) -> Optional[pd.DataFrame]:
    """从中金所日行情 XML 获取全部 IO 合约数据。"""
    yyyymm = date_str[:6]
    dd = date_str[6:8]
    url = f"http://www.cffex.com.cn/sj/hqsj/rtj/{yyyymm}/{dd}/index.xml"

    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200 or len(r.content) < 1000:
                time.sleep(0.5)
                continue
            root = ET.fromstring(r.content)
            rows = []
            for dd_el in root.findall("dailydata"):
                product = dd_el.find("productid")
                if product is None or product.text != "IO":
                    continue
                instrument = dd_el.find("instrumentid").text or ""
                # IO2507-C-3900
                parts = instrument.split("-")
                if len(parts) < 3:
                    continue
                cp = parts[1].upper()
                if cp not in ("C", "P"):
                    continue
                expiry = _extract_io_expiry(instrument)
                rows.append({
                    "date": date_str,
                    "instrument": instrument,
                    "expiry": expiry,
                    "call_put": "C" if cp == "C" else "P",
                    "volume": int(dd_el.find("volume").text or 0),
                    "turnover": float(dd_el.find("turnover").text or 0),
                    "open_interest": int(dd_el.find("openinterest").text or 0),
                    "closeprice": float(dd_el.find("closeprice").text or 0),
                })
            if rows:
                return pd.DataFrame(rows)
        except Exception as e:
            if attempt == max_retries:
                print(f"  [CFFEX] {date_str} 获取失败: {e}")
        time.sleep(0.3)
    return None


def compute_cffex_io_pcr(date_str: str) -> Optional[Dict]:
    df = fetch_cffex_option_daily(date_str)
    if df is None or df.empty:
        return None
    calls = df[df["call_put"] == "C"]
    puts = df[df["call_put"] == "P"]

    call_vol = calls["volume"].sum()
    put_vol = puts["volume"].sum()
    call_oi = calls["open_interest"].sum()
    put_oi = puts["open_interest"].sum()
    call_amt = calls["turnover"].sum()
    put_amt = puts["turnover"].sum()

    # 按到期月份拆分 CFFEX IO 成交量与持仓量，便于 tooltip 查验
    breakdown: Dict[str, Dict[str, int]] = {}
    for expiry, group in df.groupby("expiry"):
        c = group[group["call_put"] == "C"]
        p = group[group["call_put"] == "P"]
        breakdown[expiry] = {
            "call_vol": int(c["volume"].sum()),
            "put_vol": int(p["volume"].sum()),
            "call_oi": int(c["open_interest"].sum()),
            "put_oi": int(p["open_interest"].sum()),
        }

    return {
        "date": date_str,
        "cffex_io_volume_pcr": put_vol / call_vol if call_vol > 0 else np.nan,
        "cffex_io_oi_pcr": put_oi / call_oi if call_oi > 0 else np.nan,
        "cffex_io_amount_pcr": put_amt / call_amt if call_amt > 0 else np.nan,
        "cffex_io_call_vol": call_vol,
        "cffex_io_put_vol": put_vol,
        "cffex_io_call_oi": call_oi,
        "cffex_io_put_oi": put_oi,
        "cffex_io_breakdown": json.dumps(breakdown, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# 3. 上交所 510300 ETF 期权 PCR
# ---------------------------------------------------------------------------

def fetch_sse_option_daily(date_str: str, max_retries: int = 2) -> Optional[pd.DataFrame]:
    import akshare as ak
    for attempt in range(max_retries + 1):
        try:
            df = ak.option_daily_stats_sse(date=date_str)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            if attempt == max_retries:
                print(f"  [SSE] {date_str} 获取失败: {e}")
        time.sleep(0.3)
    return None


def compute_sse_300etf_pcr(date_str: str) -> Optional[Dict]:
    df = fetch_sse_option_daily(date_str)
    if df is None or df.empty:
        return None
    row = df[df["合约标的代码"] == "510300"]
    if row.empty:
        return None
    r = row.iloc[0]
    call_vol = int(r["认购成交量"])
    put_vol = int(r["认沽成交量"])
    call_oi = int(r["未平仓认购合约数"])
    put_oi = int(r["未平仓认沽合约数"])
    # 成交金额PCR 用成交额近似：认沽成交额/认购成交额
    total_amt = float(r["总成交额"]) * 10000  # 万元 -> 元
    # 上交所接口没有直接给出认购/认沽成交额，用均价近似
    # 这里先不计算 amount_pcr，因为缺乏明细；只计算 volume / OI
    return {
        "date": date_str,
        "sse_300etf_volume_pcr": put_vol / call_vol if call_vol > 0 else np.nan,
        "sse_300etf_oi_pcr": put_oi / call_oi if call_oi > 0 else np.nan,
        "sse_300etf_call_vol": call_vol,
        "sse_300etf_put_vol": put_vol,
        "sse_300etf_call_oi": call_oi,
        "sse_300etf_put_oi": put_oi,
    }


# ---------------------------------------------------------------------------
# 4. 深交所 159919 ETF 期权 PCR
# ---------------------------------------------------------------------------

def fetch_szse_option_daily(date_str: str, max_retries: int = 1) -> Optional[pd.DataFrame]:
    import akshare as ak
    for attempt in range(max_retries + 1):
        try:
            df = ak.option_daily_stats_szse(date=date_str)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            if attempt == max_retries:
                print(f"  [SZSE] {date_str} 获取失败: {str(e)[:80]}")
            # 连接被拒绝时快速失败，不重试
            if "Connection refused" in str(e):
                return None
        time.sleep(0.1)
    return None


def compute_szse_300etf_pcr(date_str: str) -> Optional[Dict]:
    df = fetch_szse_option_daily(date_str)
    if df is None or df.empty:
        return None
    row = df[df["合约标的代码"] == "159919"]
    if row.empty:
        return None
    r = row.iloc[0]
    call_vol = int(r["认购成交量"])
    put_vol = int(r["认沽成交量"])
    call_oi = int(r["未平仓认购合约数"])
    put_oi = int(r["未平仓认沽合约数"])
    return {
        "date": date_str,
        "szse_300etf_volume_pcr": put_vol / call_vol if call_vol > 0 else np.nan,
        "szse_300etf_oi_pcr": put_oi / call_oi if call_oi > 0 else np.nan,
        "szse_300etf_call_vol": call_vol,
        "szse_300etf_put_vol": put_vol,
        "szse_300etf_call_oi": call_oi,
        "szse_300etf_put_oi": put_oi,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_pcr_dataset(start_date: str, end_date: str) -> pd.DataFrame:
    """获取指定区间内的指数与三个交易所 PCR 数据，合并成日频表（带缓存）。"""
    pcr_cache = load_pcr_cache()

    print(f"\n[1/4] 获取沪深300指数日线 {start_date} ~ {end_date}")
    index_df = fetch_csi300_index(start_date, end_date)
    trade_days = index_df["date"].tolist()
    print(f"      共 {len(trade_days)} 个交易日")

    def get_or_fetch(label: str, date_str: str, fetch_fn):
        key = f"{label}::{date_str}"
        if key in pcr_cache and pcr_cache[key] is not None:
            return pcr_cache[key]
        try:
            row = fetch_fn(date_str)
            if row:
                pcr_cache[key] = row
            return row
        except Exception as e:
            print(f"\n  [{label}] {date_str} 异常: {e}")
            return None

    print("\n[2/4] 获取中金所 IO 期权 PCR（直接对应沪深300指数）")
    cffex_rows = []
    for i, d in enumerate(trade_days, 1):
        print(f"      CFFEX {d}  ({i}/{len(trade_days)})", end="\r")
        row = get_or_fetch("cffex", d, compute_cffex_io_pcr)
        if row:
            cffex_rows.append(row)
        time.sleep(0.05)
    print(f"\n      成功 {len(cffex_rows)}/{len(trade_days)} 天")

    print("\n[3/4] 获取上交所 510300 ETF 期权 PCR")
    sse_rows = []
    for i, d in enumerate(trade_days, 1):
        print(f"      SSE {d}  ({i}/{len(trade_days)})", end="\r")
        row = get_or_fetch("sse", d, compute_sse_300etf_pcr)
        if row:
            sse_rows.append(row)
        time.sleep(0.05)
    print(f"\n      成功 {len(sse_rows)}/{len(trade_days)} 天")

    print("\n[3.5/4] 获取深交所 159919 ETF 期权 PCR")
    szse_rows = []
    szse_consecutive_fail = 0
    for i, d in enumerate(trade_days, 1):
        print(f"      SZSE {d}  ({i}/{len(trade_days)})", end="\r")
        row = get_or_fetch("szse", d, compute_szse_300etf_pcr)
        if row:
            szse_rows.append(row)
            szse_consecutive_fail = 0
        else:
            szse_consecutive_fail += 1
            # 连续 5 次失败则判定为当前网络被深交所屏蔽，跳过剩余日期
            if szse_consecutive_fail >= 5:
                print(f"\n  [SZSE] 连续 {szse_consecutive_fail} 次获取失败，跳过剩余 {len(trade_days)-i} 天（CFFEX/SSE 仍为主数据源）")
                break
        time.sleep(0.05)
    print(f"\n      成功 {len(szse_rows)}/{len(trade_days)} 天")

    save_pcr_cache(pcr_cache)

    # 合并
    merged = index_df.copy()
    if cffex_rows:
        merged = merged.merge(pd.DataFrame(cffex_rows), on="date", how="left")
    if sse_rows:
        merged = merged.merge(pd.DataFrame(sse_rows), on="date", how="left")
    if szse_rows:
        merged = merged.merge(pd.DataFrame(szse_rows), on="date", how="left")

    # 确保 PCR 相关列为数值型（缓存 JSON 反序列化可能变为字符串）
    pcr_numeric_cols = [c for c in merged.columns if any(k in c for k in ["_pcr", "_vol", "_oi", "_amt", "total_call_", "total_put_"])]
    for col in pcr_numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # 合成全市场成交量/持仓量 PCR（三个交易所加总）
    # 注意：列名后缀是 _vol / _oi
    metric_map = {"vol": "volume", "oi": "oi"}
    for metric in ["vol", "oi"]:
        call_cols = [c for c in merged.columns if c.endswith(f"_call_{metric}")]
        put_cols = [c for c in merged.columns if c.endswith(f"_put_{metric}")]
        if call_cols and put_cols:
            label = metric_map[metric]
            merged[f"total_call_{label}"] = merged[call_cols].sum(axis=1, min_count=1)
            merged[f"total_put_{label}"] = merged[put_cols].sum(axis=1, min_count=1)
            merged[f"total_{label}_pcr"] = merged[f"total_put_{label}"] / merged[f"total_call_{label}"]

    # 合并 QVIX（300ETF 期权波动率指数）到主表，供叠图与交叉验证使用
    try:
        import akshare as ak
        qvix = ak.index_option_300etf_qvix()
        if qvix is not None and not qvix.empty:
            qvix = qvix.copy()
            qvix["date"] = pd.to_datetime(qvix["date"]).dt.strftime("%Y%m%d")
            qvix = qvix.rename(columns={
                "open": "qvix_open",
                "high": "qvix_high",
                "low": "qvix_low",
                "close": "qvix_close",
            })
            merged = merged.merge(qvix[["date", "qvix_open", "qvix_high", "qvix_low", "qvix_close"]],
                                  on="date", how="left")
    except Exception:
        pass

    # 滚动百分位（20/60日）用于判断高低点
    pcr_cols = [c for c in merged.columns if "_pcr" in c and "pct" not in c]
    for col in pcr_cols:
        merged[f"{col}_pct20"] = merged[col].rolling(20, min_periods=10).apply(
            lambda s: (s <= s.iloc[-1]).mean(), raw=False)
        merged[f"{col}_pct60"] = merged[col].rolling(60, min_periods=30).apply(
            lambda s: (s <= s.iloc[-1]).mean(), raw=False)

    return merged


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_html_chart(df: pd.DataFrame, output_path: str, validation_text: str = ""):
    """生成基于 ECharts 的交互式 HTML 图表（无需 matplotlib），支持日期范围筛选与数据查验。"""
    df = df.copy()
    df["date_show"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

    dates = df["date_show"].tolist()
    close = df["close"].round(2).tolist()
    ma20 = df["close"].rolling(20, min_periods=10).mean().round(2).tolist()
    ma60 = df["close"].rolling(60, min_periods=30).mean().round(2).tolist()
    date_min = dates[0] if dates else ""
    date_max = dates[-1] if dates else ""

    # 原始成交量/持仓量数据，用于 tooltip 查验
    raw_cols = {
        "cffex_call_vol": "cffex_io_call_vol",
        "cffex_put_vol": "cffex_io_put_vol",
        "cffex_call_oi": "cffex_io_call_oi",
        "cffex_put_oi": "cffex_io_put_oi",
        "sse_call_vol": "sse_300etf_call_vol",
        "sse_put_vol": "sse_300etf_put_vol",
        "sse_call_oi": "sse_300etf_call_oi",
        "sse_put_oi": "sse_300etf_put_oi",
        "total_call_volume": "total_call_volume",
        "total_put_volume": "total_put_volume",
        "total_call_oi": "total_call_oi",
        "total_put_oi": "total_put_oi",
        "qvix_close": "qvix_close",
    }
    raw_data = {}
    for key, col in raw_cols.items():
        raw_data[key] = (df[col].fillna(0).astype(int).tolist() if col in df.columns else [0] * len(df))

    # CFFEX IO 按到期月份的拆分数据（JSON 字符串），用于 tooltip 展示每一笔细分
    if "cffex_io_breakdown" in df.columns:
        breakdowns = []
        for val in df["cffex_io_breakdown"]:
            try:
                if isinstance(val, str) and val.strip():
                    breakdowns.append(json.loads(val))
                elif isinstance(val, dict):
                    breakdowns.append(val)
                else:
                    breakdowns.append({})
            except Exception:
                breakdowns.append({})
        raw_data["cffex_breakdown"] = breakdowns
    else:
        raw_data["cffex_breakdown"] = [{}] * len(df)

    # 序列构造：优先使用 CFFEX，否则用 total
    def series(name: str, col: str, color: str):
        if col not in df.columns:
            return None
        return {"name": name, "type": "line", "data": df[col].round(3).tolist(),
                "smooth": True, "symbol": "none", "lineStyle": {"width": 1.5, "color": color}}

    series_volume = []
    s = series("CFFEX IO 成交量PCR", "cffex_io_volume_pcr", "#2ca02c")
    if s:
        series_volume.append(s)
    s = series("SSE 300ETF 成交量PCR", "sse_300etf_volume_pcr", "#17becf")
    if s:
        series_volume.append(s)
    s = series("全市场成交量PCR", "total_volume_pcr", "#ff7f0e")
    if s:
        series_volume.append(s)

    series_oi = []
    s = series("CFFEX IO 持仓量PCR", "cffex_io_oi_pcr", "#9467bd")
    if s:
        series_oi.append(s)
    s = series("SSE 300ETF 持仓量PCR", "sse_300etf_oi_pcr", "#8c564b")
    if s:
        series_oi.append(s)
    s = series("全市场持仓量PCR", "total_oi_pcr", "#e377c2")
    if s:
        series_oi.append(s)

    # 叠图专用序列
    overlay_volume_pcr = "total_volume_pcr" if "total_volume_pcr" in df.columns else "cffex_io_volume_pcr"
    overlay_oi_pcr = "total_oi_pcr" if "total_oi_pcr" in df.columns else "cffex_io_oi_pcr"
    overlay_qvix = "qvix_close"

    # 交叉验证文本格式化进 HTML
    validation_html = ""
    if validation_text:
        escaped = validation_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        validation_html = '<div class="note">' + escaped.replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;") + '</div>'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>沪深300 PCR 与指数趋势</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
h2 {{ text-align: center; color: #333; }}
.controls {{ display: flex; align-items: center; justify-content: center; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; padding: 12px 16px; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
.controls label {{ font-size: 13px; color: #555; }}
.controls input[type="date"] {{ padding: 4px 8px; font-size: 13px; border: 1px solid #ccc; border-radius: 4px; }}
.controls button {{ padding: 5px 14px; font-size: 13px; border: none; border-radius: 4px; cursor: pointer; }}
.controls .btn-primary {{ background: #1f77b4; color: #fff; }}
.controls .btn-secondary {{ background: #e0e0e0; color: #333; }}
.chart {{ width: 100%; height: 260px; margin-bottom: 20px; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
.chart-overlay {{ width: 100%; height: 340px; margin-bottom: 20px; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
.note {{ color: #666; font-size: 13px; padding: 12px 20px; background: #fff; border-radius: 8px; line-height: 1.7; margin-bottom: 16px; }}
.note a {{ color: #1f77b4; text-decoration: none; }}
.note a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="container">
<h2>沪深300 指数与 PCR（认沽/认购比）趋势比较</h2>
<div class="controls">
  <label for="startDate">开始日期：</label>
  <input type="date" id="startDate" value="{date_min}" min="{date_min}" max="{date_max}">
  <label for="endDate">结束日期：</label>
  <input type="date" id="endDate" value="{date_max}" min="{date_min}" max="{date_max}">
  <button class="btn-primary" onclick="applyDateFilter()">应用筛选</button>
  <button class="btn-secondary" onclick="resetDateFilter()">重置</button>
  <span id="rangeInfo" style="font-size:12px;color:#888;margin-left:8px;">共 {len(dates)} 个交易日</span>
</div>
<div id="chart1" class="chart"></div>
<div id="chart2" class="chart"></div>
<div id="chart3" class="chart"></div>
<div id="chart4" class="chart-overlay"></div>
<div id="chart5" class="chart-overlay"></div>
<div id="chart6" class="chart-overlay"></div>
<div class="controls" style="justify-content: flex-start;">
  <b style="font-size:13px;color:#333;">自定义叠图指标（对应上方第6张图）：</b>
  <label><input type="checkbox" id="customCsi300" checked onchange="updateCustomChart(currentData)"> 沪深300指数</label>
  <label><input type="checkbox" id="customVolPCR" checked onchange="updateCustomChart(currentData)"> 成交量PCR</label>
  <label><input type="checkbox" id="customOIPCR" checked onchange="updateCustomChart(currentData)"> 持仓量PCR</label>
  <label><input type="checkbox" id="customQVIX" checked onchange="updateCustomChart(currentData)"> QVIX</label>
  <button class="btn-primary" onclick="updateCustomChart(currentData)">更新叠图</button>
</div>
<div class="note">
<b>计算公式：</b><br>
• 成交量 PCR = Σ 认沽期权成交量 / Σ 认购期权成交量<br>
• 持仓量 PCR = Σ 认沽期权持仓量 / Σ 认购期权持仓量<br>
• 全市场成交量 PCR = (CFFEX 认沽成交量 + SSE 认沽成交量 + SZSE 认沽成交量) / (CFFEX 认购成交量 + SSE 认购成交量 + SZSE 认购成交量)<br>
• 鼠标悬停在数据点上，可在 tooltip 中查看当日的原始成交量/持仓量，便于与交易所官方数据核对。<br>
<br>
<b>数据源与核对网址：</b><br>
• 沪深300指数：来自 akshare（底层 <a href="https://quote.eastmoney.com/zs000300.html" target="_blank">东方财富沪深300行情</a>）<br>
• CFFEX 沪深300股指期权（品种代码 <b>IO</b>）：<a href="http://www.cffex.com.cn/cn/hs300gzqq.html" target="_blank">中金所沪深300股指期权专区</a>；日行情原始 XML：http://www.cffex.com.cn/sj/hqsj/rtj/202507/17/index.xml。XML 中每个 IO 合约（不同行权价、到期月份）单独列出，脚本读取所有 productid=IO 的合约后，按认购/认沽分别<b>加总成交量与持仓量</b>，再计算 PCR。<br>
• SSE 300ETF 期权日统计：<a href="http://www.sse.com.cn/assortment/options/date/" target="_blank">上交所期权每日统计</a>（合约标的 510300）<br>
• 300ETF 期权波动率指数 QVIX：<a href="http://1.optbbs.com/s/vix.shtml?300ETF" target="_blank">http://1.optbbs.com/s/vix.shtml?300ETF</a>（第三方数据源：期权论坛 optbbs；原始 CSV：<a href="http://1.optbbs.com/d/csv/d/k.csv" target="_blank">http://1.optbbs.com/d/csv/d/k.csv</a>）<br>
• <span style="color:#999;">（深交所 159919 ETF 期权因网络受限未纳入计算，故不列出核对链接）</span><br>
</div>
{validation_html}
</div>
<script>
const fullDates = {json.dumps(dates, ensure_ascii=False)};
const fullClose = {json.dumps(close, ensure_ascii=False)};
const fullMA20 = {json.dumps(ma20, ensure_ascii=False)};
const fullMA60 = {json.dumps(ma60, ensure_ascii=False)};
const fullSeriesVolume = {json.dumps(series_volume, ensure_ascii=False)};
const fullSeriesOI = {json.dumps(series_oi, ensure_ascii=False)};
const fullOverlayVolumePCR = {json.dumps(df[overlay_volume_pcr].round(3).tolist(), ensure_ascii=False)};
const fullOverlayOIPCR = {json.dumps(df[overlay_oi_pcr].round(3).tolist(), ensure_ascii=False)};
const fullQVIX = {json.dumps(df[overlay_qvix].round(3).tolist() if overlay_qvix in df.columns else [], ensure_ascii=False)};
const rawData = {{
  cffexCallVol: {json.dumps(raw_data['cffex_call_vol'], ensure_ascii=False)},
  cffexPutVol: {json.dumps(raw_data['cffex_put_vol'], ensure_ascii=False)},
  cffexCallOI: {json.dumps(raw_data['cffex_call_oi'], ensure_ascii=False)},
  cffexPutOI: {json.dumps(raw_data['cffex_put_oi'], ensure_ascii=False)},
  sseCallVol: {json.dumps(raw_data['sse_call_vol'], ensure_ascii=False)},
  ssePutVol: {json.dumps(raw_data['sse_put_vol'], ensure_ascii=False)},
  sseCallOI: {json.dumps(raw_data['sse_call_oi'], ensure_ascii=False)},
  ssePutOI: {json.dumps(raw_data['sse_put_oi'], ensure_ascii=False)},
  totalCallVol: {json.dumps(raw_data['total_call_volume'], ensure_ascii=False)},
  totalPutVol: {json.dumps(raw_data['total_put_volume'], ensure_ascii=False)},
  totalCallOI: {json.dumps(raw_data['total_call_oi'], ensure_ascii=False)},
  totalPutOI: {json.dumps(raw_data['total_put_oi'], ensure_ascii=False)},
  qvixClose: {json.dumps(raw_data['qvix_close'], ensure_ascii=False)},
  cffexBreakdown: {json.dumps(raw_data['cffex_breakdown'], ensure_ascii=False)}
}};

function formatCffexBreakdown(bd) {{
  if (!bd) return '';
  const months = Object.keys(bd).filter(k => bd[k].call_vol + bd[k].put_vol + bd[k].call_oi + bd[k].put_oi > 0);
  if (months.length === 0) return '';
  // 按成交量合计排序
  months.sort((a, b) => (bd[b].call_vol + bd[b].put_vol) - (bd[a].call_vol + bd[a].put_vol));
  let html = '<br/><span style="color:#888;font-size:11px;">CFFEX IO 分月份：</span><br/>';
  months.forEach(m => {{
    html += `<span style="color:#888;font-size:11px;">IO${{m}}：认购 ${{fmtNum(bd[m].call_vol)}} / 认沽 ${{fmtNum(bd[m].put_vol)}}；持仓 认购 ${{fmtNum(bd[m].call_oi)}} / 认沽 ${{fmtNum(bd[m].put_oi)}}</span><br/>`;
  }});
  return html;
}}

function fmtNum(n) {{ return Number(n).toLocaleString(); }}

function sliceData(startIdx, endIdx) {{
  const idxEnd = endIdx === undefined ? fullDates.length : endIdx + 1;
  const d = fullDates.slice(startIdx, idxEnd);
  const slicedRaw = {{}};
  for (const k of Object.keys(rawData)) {{
    slicedRaw[k] = rawData[k].slice(startIdx, idxEnd);
  }}
  return {{
    dates: d,
    close: fullClose.slice(startIdx, idxEnd),
    ma20: fullMA20.slice(startIdx, idxEnd),
    ma60: fullMA60.slice(startIdx, idxEnd),
    overlayVolumePCR: fullOverlayVolumePCR.slice(startIdx, idxEnd),
    overlayOIPCR: fullOverlayOIPCR.slice(startIdx, idxEnd),
    qvix: fullQVIX.slice(startIdx, idxEnd),
    raw: slicedRaw,
    seriesVolume: fullSeriesVolume.map(s => ({{ ...s, data: s.data.slice(startIdx, idxEnd) }})).concat([
      {{ name: 'PCR=1.0（高位警戒）', type: 'line', data: d.map(() => 1.0), symbol: 'none', lineStyle: {{ type: 'dashed', color: '#d62728', width: 1 }} }},
      {{ name: 'PCR=0.6（低位警戒）', type: 'line', data: d.map(() => 0.6), symbol: 'none', lineStyle: {{ type: 'dashed', color: '#1f77b4', width: 1 }} }}
    ]),
    seriesOI: fullSeriesOI.map(s => ({{ ...s, data: s.data.slice(startIdx, idxEnd) }})).concat([
      {{ name: 'PCR=1.0', type: 'line', data: d.map(() => 1.0), symbol: 'none', lineStyle: {{ type: 'dashed', color: '#d62728', width: 1 }} }},
      {{ name: 'PCR=0.8', type: 'line', data: d.map(() => 0.8), symbol: 'none', lineStyle: {{ type: 'dashed', color: '#9467bd', width: 1 }} }}
    ])
  }};
}}

function getFilteredData() {{
  const startInput = document.getElementById('startDate').value;
  const endInput = document.getElementById('endDate').value;
  if (!startInput || !endInput) return sliceData(0, fullDates.length - 1);
  let startIdx = fullDates.findIndex(d => d >= startInput);
  let endIdx = fullDates.findIndex(d => d > endInput) - 1;
  if (startIdx === -1) startIdx = 0;
  if (endIdx === -2) endIdx = fullDates.length - 1;
  if (endIdx < startIdx) endIdx = startIdx;
  return sliceData(startIdx, endIdx);
}}

function indexTooltip(params) {{
  const idx = params[0].dataIndex;
  const d = params[0].axisValue;
  let html = `<b>${{d}}</b><br/>`;
  params.forEach(p => {{
    if (p.seriesName === '沪深300') html += `${{p.marker}} 沪深300: <b>${{p.value}}</b><br/>`;
    if (p.seriesName === 'MA20') html += `${{p.marker}} MA20: ${{p.value}}<br/>`;
    if (p.seriesName === 'MA60') html += `${{p.marker}} MA60: ${{p.value}}<br/>`;
  }});
  return html;
}}

function volumeTooltip(params, data) {{
  const idx = params[0].dataIndex;
  const d = params[0].axisValue;
  let html = `<b>${{d}}</b><br/>`;
  params.forEach(p => {{
    if (p.seriesName.startsWith('PCR=')) return;
    html += `${{p.marker}} ${{p.seriesName}}: <b>${{p.value}}</b><br/>`;
  }});
  html += '<hr style="border:0;border-top:1px solid #eee;margin:6px 0;"/>';
  html += '<span style="color:#666;font-size:12px;">成交量（查验用）：</span><br/>';
  html += `全市场成交量（CFFEX+SSE）：${{fmtNum(data.raw.totalCallVol[idx])}} / ${{fmtNum(data.raw.totalPutVol[idx])}}<br/>`;
  html += `CFFEX合计：${{fmtNum(data.raw.cffexCallVol[idx])}} / ${{fmtNum(data.raw.cffexPutVol[idx])}}`;
  html += formatCffexBreakdown(data.raw.cffexBreakdown[idx]);
  html += `<br/>SSE：${{fmtNum(data.raw.sseCallVol[idx])}} / ${{fmtNum(data.raw.ssePutVol[idx])}}`;
  return html;
}}

function oiTooltip(params, data) {{
  const idx = params[0].dataIndex;
  const d = params[0].axisValue;
  let html = `<b>${{d}}</b><br/>`;
  params.forEach(p => {{
    if (p.seriesName.startsWith('PCR=')) return;
    html += `${{p.marker}} ${{p.seriesName}}: <b>${{p.value}}</b><br/>`;
  }});
  html += '<hr style="border:0;border-top:1px solid #eee;margin:6px 0;"/>';
  html += '<span style="color:#666;font-size:12px;">持仓量（查验用）：</span><br/>';
  html += `全市场持仓量（CFFEX+SSE）：${{fmtNum(data.raw.totalCallOI[idx])}} / ${{fmtNum(data.raw.totalPutOI[idx])}}<br/>`;
  html += `CFFEX合计：${{fmtNum(data.raw.cffexCallOI[idx])}} / ${{fmtNum(data.raw.cffexPutOI[idx])}}`;
  html += formatCffexBreakdown(data.raw.cffexBreakdown[idx]);
  html += `<br/>SSE：${{fmtNum(data.raw.sseCallOI[idx])}} / ${{fmtNum(data.raw.ssePutOI[idx])}}`;
  return html;
}}

function overlayTooltip(params, data) {{
  const idx = params[0].dataIndex;
  const d = params[0].axisValue;
  let html = `<b>${{d}}</b><br/>`;
  params.forEach(p => {{
    if (p.seriesName === '沪深300指数') html += `${{p.marker}} 沪深300: <b>${{p.value}}</b><br/>`;
    if (p.seriesName === '成交量PCR') html += `${{p.marker}} 成交量PCR: <b>${{p.value}}</b><br/>`;
    if (p.seriesName === '持仓量PCR') html += `${{p.marker}} 持仓量PCR: <b>${{p.value}}</b><br/>`;
  }});
  html += '<hr style="border:0;border-top:1px solid #eee;margin:6px 0;"/>';
  html += '<span style="color:#666;font-size:12px;">成交量（查验用）：</span><br/>';
  html += `全市场成交量（CFFEX+SSE）：${{fmtNum(data.raw.totalCallVol[idx])}} / ${{fmtNum(data.raw.totalPutVol[idx])}}<br/>`;
  html += `CFFEX合计：${{fmtNum(data.raw.cffexCallVol[idx])}} / ${{fmtNum(data.raw.cffexPutVol[idx])}}`;
  html += formatCffexBreakdown(data.raw.cffexBreakdown[idx]);
  html += `<br/>SSE：${{fmtNum(data.raw.sseCallVol[idx])}} / ${{fmtNum(data.raw.ssePutVol[idx])}}<br/>`;
  html += '<span style="color:#666;font-size:12px;">持仓量（查验用）：</span><br/>';
  html += `全市场持仓量（CFFEX+SSE）：${{fmtNum(data.raw.totalCallOI[idx])}} / ${{fmtNum(data.raw.totalPutOI[idx])}}<br/>`;
  html += `CFFEX合计：${{fmtNum(data.raw.cffexCallOI[idx])}} / ${{fmtNum(data.raw.cffexPutOI[idx])}}`;
  html += formatCffexBreakdown(data.raw.cffexBreakdown[idx]);
  html += `<br/>SSE：${{fmtNum(data.raw.sseCallOI[idx])}} / ${{fmtNum(data.raw.ssePutOI[idx])}}`;
  return html;
}}

function qvixOverlayTooltip(params, data) {{
  const idx = params[0].dataIndex;
  const d = params[0].axisValue;
  let html = `<b>${{d}}</b><br/>`;
  params.forEach(p => {{
    if (p.seriesName === '沪深300指数') html += `${{p.marker}} 沪深300: <b>${{p.value}}</b><br/>`;
    if (p.seriesName === '成交量PCR') html += `${{p.marker}} 成交量PCR: <b>${{p.value}}</b><br/>`;
    if (p.seriesName === 'QVIX') html += `${{p.marker}} QVIX: <b>${{p.value}}</b><br/>`;
  }});
  html += '<hr style="border:0;border-top:1px solid #eee;margin:6px 0;"/>';
  html += '<span style="color:#666;font-size:12px;">成交量（查验用）：</span><br/>';
  html += `全市场成交量（CFFEX+SSE）：${{fmtNum(data.raw.totalCallVol[idx])}} / ${{fmtNum(data.raw.totalPutVol[idx])}}<br/>`;
  html += `CFFEX合计：${{fmtNum(data.raw.cffexCallVol[idx])}} / ${{fmtNum(data.raw.cffexPutVol[idx])}}`;
  html += formatCffexBreakdown(data.raw.cffexBreakdown[idx]);
  html += `<br/>SSE：${{fmtNum(data.raw.sseCallVol[idx])}} / ${{fmtNum(data.raw.ssePutVol[idx])}}<br/>`;
  html += `QVIX：${{data.raw.qvixClose[idx] ? data.raw.qvixClose[idx].toFixed(3) : 'N/A'}}`;
  return html;
}}

function buildCommonOption(dates) {{
  return {{
    tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }} }},
    legend: {{ top: 30, type: 'scroll', itemGap: 8, textStyle: {{ fontSize: 11 }} }},
    grid: {{ left: '4%', right: '4%', bottom: '12%', top: '20%', containLabel: true }},
    xAxis: {{ type: 'category', data: dates, boundaryGap: false, axisLabel: {{ rotate: 30 }} }},
    dataZoom: [
    {{ type: 'inside' }},
    {{
      type: 'slider',
      bottom: 0,
      height: 32,
      showDetail: true,
      backgroundColor: '#f5f5f5',
      fillerColor: 'rgba(31,119,180,0.25)',
      borderColor: '#bbb',
      handleIcon: 'path://M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
      handleSize: '80%',
      handleStyle: {{
        color: '#1f77b4',
        shadowBlur: 3,
        shadowColor: 'rgba(0,0,0,0.3)',
        shadowOffsetX: 0,
        shadowOffsetY: 0
      }},
      textStyle: {{ color: '#444' }}
    }}
  ]
  }};
}}

let chart1, chart2, chart3, chart4, chart5, chart6;
function renderCharts(data) {{
  if (!chart1) chart1 = echarts.init(document.getElementById('chart1'));
  if (!chart2) chart2 = echarts.init(document.getElementById('chart2'));
  if (!chart3) chart3 = echarts.init(document.getElementById('chart3'));
  if (!chart4) chart4 = echarts.init(document.getElementById('chart4'));
  if (!chart5) chart5 = echarts.init(document.getElementById('chart5'));

  chart1.setOption({{
    ...buildCommonOption(data.dates),
    title: {{ text: '沪深300指数', left: 'center', top: 5, textStyle: {{ fontSize: 14 }} }},
    tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }}, formatter: indexTooltip }},
    graphic: [{{
      type: 'group',
      left: 6,
      bottom: 34,
      children: [
        {{
          type: 'rect',
          shape: {{ width: 150, height: 24, r: 4 }},
          style: {{ fill: 'rgba(255, 243, 205, 0.95)', stroke: '#ffc107', lineWidth: 1 }}
        }},
        {{
          type: 'text',
          left: 7,
          top: 5,
          style: {{ text: '← 拉动调整时间范围', fill: '#856404', fontSize: 12, fontWeight: 'bold' }}
        }}
      ]
    }}],
    yAxis: [{{ type: 'value', name: '指数', scale: true }}],
    series: [
      {{ name: '沪深300', type: 'line', data: data.close, smooth: true, symbol: 'none', lineStyle: {{ width: 1.5, color: '#1f77b4' }} }},
      {{ name: 'MA20', type: 'line', data: data.ma20, smooth: true, symbol: 'none', lineStyle: {{ width: 1, color: '#ff7f0e' }} }},
      {{ name: 'MA60', type: 'line', data: data.ma60, smooth: true, symbol: 'none', lineStyle: {{ width: 1, color: '#d62728' }} }}
    ]
  }}, true);

  chart2.setOption({{
    ...buildCommonOption(data.dates),
    title: {{ text: '成交量 PCR', left: 'center', top: 5, textStyle: {{ fontSize: 14 }} }},
    tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }}, formatter: p => volumeTooltip(p, data) }},
    graphic: [{{
      type: 'group',
      left: 6,
      bottom: 34,
      children: [
        {{
          type: 'rect',
          shape: {{ width: 150, height: 24, r: 4 }},
          style: {{ fill: 'rgba(255, 243, 205, 0.95)', stroke: '#ffc107', lineWidth: 1 }}
        }},
        {{
          type: 'text',
          left: 7,
          top: 5,
          style: {{ text: '← 拉动调整时间范围', fill: '#856404', fontSize: 12, fontWeight: 'bold' }}
        }}
      ]
    }}],
    yAxis: [{{ type: 'value', name: 'PCR', scale: true, axisLabel: {{ formatter: '{{value}}' }} }}],
    series: data.seriesVolume
  }}, true);

  chart3.setOption({{
    ...buildCommonOption(data.dates),
    title: {{ text: '持仓量 PCR', left: 'center', top: 5, textStyle: {{ fontSize: 14 }} }},
    tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }}, formatter: p => oiTooltip(p, data) }},
    graphic: [{{
      type: 'group',
      left: 6,
      bottom: 34,
      children: [
        {{
          type: 'rect',
          shape: {{ width: 150, height: 24, r: 4 }},
          style: {{ fill: 'rgba(255, 243, 205, 0.95)', stroke: '#ffc107', lineWidth: 1 }}
        }},
        {{
          type: 'text',
          left: 7,
          top: 5,
          style: {{ text: '← 拉动调整时间范围', fill: '#856404', fontSize: 12, fontWeight: 'bold' }}
        }}
      ]
    }}],
    yAxis: [{{ type: 'value', name: 'PCR', scale: true, axisLabel: {{ formatter: '{{value}}' }} }}],
    series: data.seriesOI
  }}, true);

  chart4.setOption({{
    ...buildCommonOption(data.dates),
    title: {{ text: '沪深300指数 + 成交量PCR + 持仓量PCR（叠图，无均线）', left: 'center', top: 5, textStyle: {{ fontSize: 14 }} }},
    legend: {{ top: 30, type: 'scroll', itemGap: 8, textStyle: {{ fontSize: 11 }} }},
    tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }}, formatter: p => overlayTooltip(p, data) }},
    graphic: [{{
      type: 'group',
      left: 6,
      bottom: 34,
      children: [
        {{
          type: 'rect',
          shape: {{ width: 150, height: 24, r: 4 }},
          style: {{ fill: 'rgba(255, 243, 205, 0.95)', stroke: '#ffc107', lineWidth: 1 }}
        }},
        {{
          type: 'text',
          left: 7,
          top: 5,
          style: {{ text: '← 拉动调整时间范围', fill: '#856404', fontSize: 12, fontWeight: 'bold' }}
        }}
      ]
    }}],
    grid: {{ left: '5%', right: '5%', bottom: '12%', top: '18%', containLabel: true }},
    yAxis: [
      {{ type: 'value', name: '沪深300指数', position: 'left', scale: true, axisLine: {{ lineStyle: {{ color: '#1f77b4' }} }}, axisLabel: {{ color: '#1f77b4' }} }},
      {{ type: 'value', name: 'PCR', position: 'right', scale: true, min: 0, axisLine: {{ lineStyle: {{ color: '#d62728' }} }}, axisLabel: {{ color: '#d62728' }} }}
    ],
    series: [
      {{ name: '沪深300指数', type: 'line', yAxisIndex: 0, data: data.close, smooth: true, symbol: 'none', lineStyle: {{ width: 2, color: '#1f77b4' }} }},
      {{ name: '成交量PCR', type: 'line', yAxisIndex: 1, data: data.overlayVolumePCR, smooth: true, symbol: 'none', lineStyle: {{ width: 1.5, color: '#2ca02c' }} }},
      {{ name: '持仓量PCR', type: 'line', yAxisIndex: 1, data: data.overlayOIPCR, smooth: true, symbol: 'none', lineStyle: {{ width: 1.5, color: '#9467bd' }} }}
    ]
  }}, true);

  const hasQVIX = data.qvix && data.qvix.length > 0 && data.qvix.some(v => v !== null && v !== undefined);
  chart5.setOption({{
    ...buildCommonOption(data.dates),
    title: {{ text: '沪深300指数 + 成交量PCR + QVIX（三轴叠图）', left: 'center', top: 5, textStyle: {{ fontSize: 14 }} }},
    legend: {{ top: 30, type: 'scroll', itemGap: 8, textStyle: {{ fontSize: 11 }} }},
    tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }}, formatter: p => qvixOverlayTooltip(p, data) }},
    graphic: [{{
      type: 'group',
      left: 6,
      bottom: 34,
      children: [
        {{
          type: 'rect',
          shape: {{ width: 150, height: 24, r: 4 }},
          style: {{ fill: 'rgba(255, 243, 205, 0.95)', stroke: '#ffc107', lineWidth: 1 }}
        }},
        {{
          type: 'text',
          left: 7,
          top: 5,
          style: {{ text: '← 拉动调整时间范围', fill: '#856404', fontSize: 12, fontWeight: 'bold' }}
        }}
      ]
    }}],
    grid: {{ left: '6%', right: '12%', bottom: '12%', top: '18%', containLabel: true }},
    yAxis: [
      {{ type: 'value', name: '沪深300指数', position: 'left', scale: true, axisLine: {{ lineStyle: {{ color: '#1f77b4' }} }}, axisLabel: {{ color: '#1f77b4' }} }},
      {{ type: 'value', name: 'PCR', position: 'right', scale: true, min: 0, offset: 0, axisLine: {{ lineStyle: {{ color: '#2ca02c' }} }}, axisLabel: {{ color: '#2ca02c' }} }},
      {{ type: 'value', name: 'QVIX', position: 'right', scale: true, min: 0, offset: 60, axisLine: {{ lineStyle: {{ color: '#e377c2' }} }}, axisLabel: {{ color: '#e377c2' }} }}
    ],
    series: [
      {{ name: '沪深300指数', type: 'line', yAxisIndex: 0, data: data.close, smooth: true, symbol: 'none', lineStyle: {{ width: 2, color: '#1f77b4' }} }},
      {{ name: '成交量PCR', type: 'line', yAxisIndex: 1, data: data.overlayVolumePCR, smooth: true, symbol: 'none', lineStyle: {{ width: 1.5, color: '#2ca02c' }} }},
      {{ name: 'QVIX', type: 'line', yAxisIndex: 2, data: data.qvix, smooth: true, symbol: 'none', lineStyle: {{ width: 1.5, color: '#e377c2' }} }}
    ]
  }}, true);

  document.getElementById('rangeInfo').textContent = `共 ${{data.dates.length}} 个交易日`;
  currentData = data;
  updateCustomChart(data);
}}

let currentData = sliceData(0, fullDates.length - 1);

function customTooltip(params, data) {{
  const idx = params[0].dataIndex;
  const d = params[0].axisValue;
  let html = `<b>${{d}}</b><br/>`;
  params.forEach(p => {{
    if (p.seriesName === '沪深300指数') html += `${{p.marker}} 沪深300: <b>${{p.value}}</b><br/>`;
    if (p.seriesName === '成交量PCR') html += `${{p.marker}} 成交量PCR: <b>${{p.value}}</b><br/>`;
    if (p.seriesName === '持仓量PCR') html += `${{p.marker}} 持仓量PCR: <b>${{p.value}}</b><br/>`;
    if (p.seriesName === 'QVIX') html += `${{p.marker}} QVIX: <b>${{p.value}}</b><br/>`;
  }});
  html += '<hr style="border:0;border-top:1px solid #eee;margin:6px 0;"/>';
  html += '<span style="color:#666;font-size:12px;">成交量（查验用）：</span><br/>';
  html += `全市场成交量（CFFEX+SSE）：${{fmtNum(data.raw.totalCallVol[idx])}} / ${{fmtNum(data.raw.totalPutVol[idx])}}<br/>`;
  html += `CFFEX合计：${{fmtNum(data.raw.cffexCallVol[idx])}} / ${{fmtNum(data.raw.cffexPutVol[idx])}}`;
  html += formatCffexBreakdown(data.raw.cffexBreakdown[idx]);
  html += `<br/>SSE：${{fmtNum(data.raw.sseCallVol[idx])}} / ${{fmtNum(data.raw.ssePutVol[idx])}}<br/>`;
  html += '<span style="color:#666;font-size:12px;">持仓量（查验用）：</span><br/>';
  html += `全市场持仓量（CFFEX+SSE）：${{fmtNum(data.raw.totalCallOI[idx])}} / ${{fmtNum(data.raw.totalPutOI[idx])}}<br/>`;
  html += `CFFEX合计：${{fmtNum(data.raw.cffexCallOI[idx])}} / ${{fmtNum(data.raw.cffexPutOI[idx])}}`;
  html += formatCffexBreakdown(data.raw.cffexBreakdown[idx]);
  html += `<br/>SSE：${{fmtNum(data.raw.sseCallOI[idx])}} / ${{fmtNum(data.raw.ssePutOI[idx])}}<br/>`;
  html += `QVIX：${{data.raw.qvixClose[idx] ? data.raw.qvixClose[idx].toFixed(3) : 'N/A'}}`;
  return html;
}}

function updateCustomChart(data) {{
  if (!chart6) chart6 = echarts.init(document.getElementById('chart6'));
  const showIndex = document.getElementById('customCsi300').checked;
  const showVol = document.getElementById('customVolPCR').checked;
  const showOI = document.getElementById('customOIPCR').checked;
  const showQVIX = document.getElementById('customQVIX').checked;

  const yAxis = [];
  const series = [];
  let axisIdx = 0;

  if (showIndex) {{
    yAxis.push({{ type: 'value', name: '沪深300指数', position: 'left', scale: true, axisLine: {{ lineStyle: {{ color: '#1f77b4' }} }}, axisLabel: {{ color: '#1f77b4' }} }});
    series.push({{ name: '沪深300指数', type: 'line', yAxisIndex: axisIdx, data: data.close, smooth: true, symbol: 'none', lineStyle: {{ width: 2, color: '#1f77b4' }} }});
    axisIdx++;
  }}
  if (showVol || showOI) {{
    yAxis.push({{ type: 'value', name: 'PCR', position: 'right', scale: true, min: 0, offset: 0, axisLine: {{ lineStyle: {{ color: '#d62728' }} }}, axisLabel: {{ color: '#d62728' }} }});
    if (showVol) series.push({{ name: '成交量PCR', type: 'line', yAxisIndex: axisIdx, data: data.overlayVolumePCR, smooth: true, symbol: 'none', lineStyle: {{ width: 1.5, color: '#2ca02c' }} }});
    if (showOI) series.push({{ name: '持仓量PCR', type: 'line', yAxisIndex: axisIdx, data: data.overlayOIPCR, smooth: true, symbol: 'none', lineStyle: {{ width: 1.5, color: '#9467bd' }} }});
    axisIdx++;
  }}
  if (showQVIX) {{
    yAxis.push({{ type: 'value', name: 'QVIX', position: 'right', scale: true, min: 0, offset: showVol || showOI ? 60 : 0, axisLine: {{ lineStyle: {{ color: '#e377c2' }} }}, axisLabel: {{ color: '#e377c2' }} }});
    series.push({{ name: 'QVIX', type: 'line', yAxisIndex: axisIdx, data: data.qvix, smooth: true, symbol: 'none', lineStyle: {{ width: 1.5, color: '#e377c2' }} }});
    axisIdx++;
  }}

  if (series.length === 0) {{
    chart6.clear();
    return;
  }}

  chart6.setOption({{
    tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }}, formatter: p => customTooltip(p, data) }},
    legend: {{ top: 30, type: 'scroll', itemGap: 8, textStyle: {{ fontSize: 11 }} }},
    graphic: [{{
      type: 'group',
      left: 6,
      bottom: 34,
      children: [
        {{
          type: 'rect',
          shape: {{ width: 150, height: 24, r: 4 }},
          style: {{ fill: 'rgba(255, 243, 205, 0.95)', stroke: '#ffc107', lineWidth: 1 }}
        }},
        {{
          type: 'text',
          left: 7,
          top: 5,
          style: {{ text: '← 拉动调整时间范围', fill: '#856404', fontSize: 12, fontWeight: 'bold' }}
        }}
      ]
    }}],
    grid: {{ left: '6%', right: '12%', bottom: '12%', top: '18%', containLabel: true }},
    xAxis: {{ type: 'category', data: data.dates, boundaryGap: false, axisLabel: {{ rotate: 30 }} }},
    dataZoom: [
    {{ type: 'inside' }},
    {{
      type: 'slider',
      bottom: 0,
      height: 32,
      showDetail: true,
      backgroundColor: '#f5f5f5',
      fillerColor: 'rgba(31,119,180,0.25)',
      borderColor: '#bbb',
      handleIcon: 'path://M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
      handleSize: '80%',
      handleStyle: {{
        color: '#1f77b4',
        shadowBlur: 3,
        shadowColor: 'rgba(0,0,0,0.3)',
        shadowOffsetX: 0,
        shadowOffsetY: 0
      }},
      textStyle: {{ color: '#444' }}
    }}
  ],
    yAxis: yAxis,
    series: series
  }}, true);
}}

function applyDateFilter() {{
  currentData = getFilteredData();
  renderCharts(currentData);
  updateCustomChart(currentData);
}}

function resetDateFilter() {{
  document.getElementById('startDate').value = '{date_min}';
  document.getElementById('endDate').value = '{date_max}';
  renderCharts(sliceData(0, fullDates.length - 1));
}}

renderCharts(sliceData(0, fullDates.length - 1));
window.addEventListener('resize', () => {{ chart1.resize(); chart2.resize(); chart3.resize(); chart4.resize(); chart5.resize(); chart6.resize(); }});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[Chart] 已保存交互式图表: {output_path}")


def plot_chart(df: pd.DataFrame, output_path: str, validation_text: str = ""):
    """兼容旧接口：优先生成 HTML，若 matplotlib 可用则额外生成 PNG。"""
    plot_html_chart(df, output_path.replace(".png", ".html"), validation_text=validation_text)
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        _png_path = output_path
        df = df.copy()
        df["date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d")
        primary_pcr = "cffex_io_volume_pcr" if "cffex_io_volume_pcr" in df.columns else (
            "total_volume_pcr" if "total_volume_pcr" in df.columns else None)
        oi_pcr = "cffex_io_oi_pcr" if "cffex_io_oi_pcr" in df.columns else None

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1, 1]})
        ax1 = axes[0]
        ax1.plot(df["date_dt"], df["close"], label="沪深300", color="#1f77b4", linewidth=1.5)
        ax1.plot(df["date_dt"], df["close"].rolling(20).mean(), label="MA20", color="orange", alpha=0.8)
        ax1.plot(df["date_dt"], df["close"].rolling(60).mean(), label="MA60", color="red", alpha=0.7)
        ax1.set_ylabel("沪深300指数")
        ax1.set_title("沪深300 指数与 PCR 趋势比较")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        if primary_pcr:
            ax2.plot(df["date_dt"], df[primary_pcr], label=primary_pcr, color="green", linewidth=1.2)
            ax2.axhline(1.0, color="red", linestyle="--", alpha=0.5)
            ax2.axhline(0.6, color="blue", linestyle="--", alpha=0.5)
            ax2.set_ylabel("成交量 PCR")
            ax2.legend(loc="upper left")
            ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        if oi_pcr:
            ax3.plot(df["date_dt"], df[oi_pcr], label=oi_pcr, color="purple", linewidth=1.2)
            ax3.axhline(1.0, color="red", linestyle="--", alpha=0.5)
            ax3.axhline(0.8, color="blue", linestyle="--", alpha=0.5)
            ax3.set_ylabel("持仓量 PCR")
            ax3.set_xlabel("日期")
            ax3.legend(loc="upper left")
            ax3.grid(True, alpha=0.3)

        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(_png_path, dpi=150)
        print(f"[Chart] 已保存 PNG 图表: {_png_path}")
        plt.close(fig)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 交叉验证：多源数据相互核对
# ---------------------------------------------------------------------------

def cross_validate_pcr(df: pd.DataFrame, sample_days: int = 10) -> Tuple[str, Dict]:
    """
    1. 计算 CFFEX / SSE / total PCR 之间的相关性；
    2. 对最近 sample_days 个交易日，重新拉取上交所官方日统计，
       将我们算出的 SSE 300ETF PCR 与官方公布的「认沽/认购」及持仓 PCR 做比对。
    返回 (报告文本, 结果字典)。
    """
    import akshare as ak
    records: List[str] = []
    result: Dict = {"corrs": {}, "sse_spot_check": []}

    # 1. 内部相关性
    pcr_cols = [c for c in df.columns if "pcr" in c and "pct" not in c]
    if len(pcr_cols) >= 2:
        corr_mat = df[pcr_cols].corr()
        records.append("【内部交叉验证】各 PCR 序列相关性：")
        for i, c1 in enumerate(pcr_cols):
            for c2 in pcr_cols[i + 1:]:
                val = corr_mat.loc[c1, c2]
                if pd.notna(val):
                    key = f"{c1} vs {c2}"
                    result["corrs"][key] = round(val, 3)
                    records.append(f"  • {key}: {val:.3f}")
        records.append("")

    # 2. SSE 官方 spot check
    if "sse_300etf_volume_pcr" in df.columns:
        recent = df.dropna(subset=["sse_300etf_volume_pcr"]).tail(sample_days)
        discrepancies_vol = []
        discrepancies_oi = []
        for _, row in recent.iterrows():
            date_str = str(int(row["date"]))
            try:
                official = ak.option_daily_stats_sse(date=date_str)
                if official is None or official.empty:
                    continue
                # 校验返回的交易日是否匹配请求日期，防止非交易日返回默认/历史数据
                trade_date_col = "交易日"
                if trade_date_col in official.columns:
                    official_dates = official[trade_date_col].astype(str).str.replace("-", "")
                    if not (official_dates == date_str).any():
                        continue
                sse_row = official[official["合约标的代码"] == "510300"]
                if sse_row.empty:
                    continue
                r = sse_row.iloc[0]
                # 上交所「认沽/认购」列官方以百分比形式披露，需除以 100
                official_vol_pcr = float(r["认沽/认购"]) / 100.0
                official_oi_pcr = float(r["未平仓认沽合约数"]) / float(r["未平仓认购合约数"])
                our_vol_pcr = row["sse_300etf_volume_pcr"]
                our_oi_pcr = row["sse_300etf_oi_pcr"]
                dvol = our_vol_pcr - official_vol_pcr
                doi = our_oi_pcr - official_oi_pcr
                discrepancies_vol.append(abs(dvol))
                discrepancies_oi.append(abs(doi))
                result["sse_spot_check"].append({
                    "date": date_str,
                    "our_vol_pcr": round(our_vol_pcr, 4),
                    "official_vol_pcr": round(official_vol_pcr, 4),
                    "diff_vol_pcr": round(dvol, 4),
                    "our_oi_pcr": round(our_oi_pcr, 4),
                    "official_oi_pcr": round(official_oi_pcr, 4),
                    "diff_oi_pcr": round(doi, 4),
                })
            except Exception as e:
                # spot check 失败不影响主流程
                continue

        if discrepancies_vol:
            max_vol = max(discrepancies_vol)
            mean_vol = sum(discrepancies_vol) / len(discrepancies_vol)
            max_oi = max(discrepancies_oi)
            mean_oi = sum(discrepancies_oi) / len(discrepancies_oi)
            records.append("【上交所官方 spot check】")
            records.append(f"  最近尝试 {sample_days} 个交易日，成功比对 {len(discrepancies_vol)} 天：")
            records.append(f"  • 成交量 PCR 最大偏差: {max_vol:.4f}，平均偏差: {mean_vol:.4f}")
            records.append(f"  • 持仓量 PCR 最大偏差: {max_oi:.4f}，平均偏差: {mean_oi:.4f}")
            records.append("  （偏差≈0 表示我们计算与上交所官方披露一致）")
        else:
            records.append("【上交所官方 spot check】未能获取官方日统计进行比对（请求日期可能为非交易日或数据未更新）。")

    # 3. 与 QVIX（300ETF 期权波动率指数）做情绪交叉验证
    try:
        if "qvix_close" in df.columns:
            _df = df.dropna(subset=["qvix_close"]).copy()
        else:
            qvix = ak.index_option_300etf_qvix()
            qvix["date"] = pd.to_datetime(qvix["date"]).dt.strftime("%Y%m%d")
            qvix = qvix.rename(columns={"close": "qvix_close"})
            _df = df.merge(qvix[["date", "qvix_close"]], on="date", how="inner")
        if len(_df) >= 30:
            records.append("")
            records.append("【与 300ETF 期权波动率指数 QVIX 交叉验证】")
            for pcr_col in ["cffex_io_volume_pcr", "sse_300etf_volume_pcr", "total_volume_pcr"]:
                if pcr_col in _df.columns:
                    corr = _df[pcr_col].corr(_df["qvix_close"])
                    if pd.notna(corr):
                        result[f"qvix_corr_{pcr_col}"] = round(corr, 3)
                        records.append(f"  • {pcr_col} vs QVIX: {corr:.3f}（同向为正，PCR 与波动率同为情绪恐慌指标）")
    except Exception as e:
        records.append("")
        records.append(f"【QVIX 交叉验证】未能获取 QVIX 数据: {str(e)[:80]}")

    # 4. CFFEX 与 SSE 成交量 PCR 的日涨跌方向一致性（独立源交叉验证）
    if "cffex_io_volume_pcr" in df.columns and "sse_300etf_volume_pcr" in df.columns:
        _df = df[["cffex_io_volume_pcr", "sse_300etf_volume_pcr"]].dropna().copy()
        if len(_df) >= 30:
            _df["cffex_diff"] = _df["cffex_io_volume_pcr"].diff()
            _df["sse_diff"] = _df["sse_300etf_volume_pcr"].diff()
            _df["same_direction"] = (_df["cffex_diff"] * _df["sse_diff"]) > 0
            agreement = _df["same_direction"].mean()
            result["cffex_sse_direction_agreement"] = round(agreement, 3)
            records.append("")
            records.append("【CFFEX vs SSE 方向一致性验证】")
            records.append(f"  • CFFEX IO 成交量 PCR 与 SSE 300ETF 成交量 PCR 日涨跌同向比例: {agreement:.1%}")
            records.append("  （比例>50% 说明两个独立交易所的情绪指标趋势一致）")

    # 5. 与沪深300已实现波动率做交叉验证（用指数日线独立计算）
    if "close" in df.columns and len(df) >= 30:
        _df = df.copy()
        _df["ret"] = _df["close"].pct_change()
        # 5 日年化已实现波动率
        _df["realized_vol_5d"] = _df["ret"].rolling(5).std() * (252 ** 0.5)
        # 20 日年化已实现波动率
        _df["realized_vol_20d"] = _df["ret"].rolling(20).std() * (252 ** 0.5)
        records.append("")
        records.append("【与沪深300已实现波动率交叉验证】（完全独立于期权数据）")
        for pcr_col in ["cffex_io_volume_pcr", "sse_300etf_volume_pcr", "total_volume_pcr"]:
            if pcr_col in _df.columns:
                corr_5 = _df[pcr_col].corr(_df["realized_vol_5d"])
                corr_20 = _df[pcr_col].corr(_df["realized_vol_20d"])
                if pd.notna(corr_5):
                    result[f"rv5_corr_{pcr_col}"] = round(corr_5, 3)
                    result[f"rv20_corr_{pcr_col}"] = round(corr_20, 3)
                    records.append(f"  • {pcr_col} vs 5日已实现波动率: {corr_5:.3f}")
                    records.append(f"  • {pcr_col} vs 20日已实现波动率: {corr_20:.3f}")

    return "\n".join(records), result


# ---------------------------------------------------------------------------
# 简单规律统计
# ---------------------------------------------------------------------------

def summarize_signals(df: pd.DataFrame, validation_text: str = "") -> str:
    # 把验证报告并入摘要
    validation_section = validation_text + "\n\n" if validation_text else ""
    df = df.copy()
    # 使用 CFFEX IO volume PCR 作为情绪指标
    col = "cffex_io_volume_pcr"
    if col not in df.columns:
        col = "total_volume_pcr"
    if col not in df.columns:
        return "未找到可用 PCR 列，无法生成规律摘要。"

    s = df[col].dropna()
    if s.empty:
        return "PCR 数据为空。"

    # 阈值参考：基于历史 20 日滚动极值
    df["high_pcr"] = df[col] > s.quantile(0.85)
    df["low_pcr"] = df[col] < s.quantile(0.15)

    # 未来 5 日涨跌
    df["future_5d_return"] = df["close"].shift(-5) / df["close"] - 1

    high_avg = df.loc[df["high_pcr"], "future_5d_return"].mean()
    low_avg = df.loc[df["low_pcr"], "future_5d_return"].mean()

    corr = df[col].corr(df["close"].pct_change().rolling(5).mean().shift(-5))

    lines = [
        "=" * 60,
        "PCR 与指数高低点规律初探（统计口径：CFFEX 沪深300股指期权成交量 PCR）",
        "=" * 60,
        f"区间: {df['date'].min()} ~ {df['date'].max()}",
        f"PCR 均值: {s.mean():.3f}  |  中位数: {s.median():.3f}",
        f"PCR 最小值: {s.min():.3f}  |  最大值: {s.max():.3f}",
        "",
        "成交量 PCR 与指数通常呈负相关：",
        "  - PCR 极高（>85% 分位）往往对应市场情绪恐慌，指数处于低位区域；",
        "  - PCR 极低（<15% 分位）往往对应市场情绪贪婪，指数处于高位区域。",
        "",
        f"PCR 高位后 5 日平均涨跌幅: {high_avg*100:+.2f}%",
        f"PCR 低位后 5 日平均涨跌幅: {low_avg*100:+.2f}%",
        f"PCR 与未来5日收益率相关性: {corr:.3f}",
        "",
        "注意：PCR 仅为情绪指标，需结合趋势、波动率、资金流综合判断。",
        "=" * 60,
    ]
    return validation_section + "\n".join(lines)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    import sys
    # 默认近两年；可接受命令行参数：python csi300_pcr_analyzer.py [days]
    days = 365 * 2
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            print("用法: python csi300_pcr_analyzer.py [近N天]")
            return

    end = datetime.now() - timedelta(days=END_OFFSET_DAYS)
    start = end - timedelta(days=days)
    end_date = end.strftime("%Y%m%d")
    start_date = start.strftime("%Y%m%d")

    print("=" * 60)
    print("沪深300 PCR 与指数趋势比较工具")
    print("=" * 60)
    print(f"区间：{start_date} ~ {end_date}（近{days}天）")

    df = build_pcr_dataset(start_date, end_date)

    csv_path = os.path.join(OUTPUT_DIR, "csi300_pcr_index.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[CSV] 已保存数据: {csv_path}")

    print("\n" + "-" * 60)
    print("正在进行多源交叉验证...")
    validation_text, validation_result = cross_validate_pcr(df, sample_days=10)
    print("交叉验证完成，结果将写入摘要与 HTML。")
    print("-" * 60)

    chart_path = os.path.join(OUTPUT_DIR, "csi300_pcr_chart.png")
    plot_chart(df, chart_path, validation_text=validation_text)
    html_path = os.path.join(OUTPUT_DIR, "csi300_pcr_chart.html")
    print(f"[HTML] 用浏览器打开即可交互查看: {html_path}")

    summary = summarize_signals(df, validation_text=validation_text)
    print("\n" + summary)

    summary_path = os.path.join(OUTPUT_DIR, "csi300_pcr_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\n[TXT] 已保存摘要: {summary_path}")


if __name__ == "__main__":
    main()
