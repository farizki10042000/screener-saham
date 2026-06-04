#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screener Saham BEI - 3 STRATEGI (Streamlit) - v3
=================================================
Menu strategi:
  🌙 Beli Sore - Jual Pagi  : beli di Close, jual di Open besok (overnight gap)
  ☀️ Beli Pagi - Jual Sore  : beli di Open, jual di Close hari yang sama (intraday)
  📈 Swing                  : beli & tahan N hari bursa (tren MA20/MA50)

Tiap strategi punya perhitungan, verdict (Layak/Hampir/Tidak + t-stat),
dan CATATAN ANALISA dengan alasan eksplisit per saham.

ALAT BANTU RISET - bukan nasihat keuangan / sinyal beli-jual.
"""

import io
import math
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Screener Saham BEI - 3 Strategi",
                   page_icon="📈", layout="wide")

# ====================== UNIVERSE ~140 SAHAM LIKUID BEI =======================
DEFAULT_TICKERS = [
    "BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK","BRIS.JK","BBTN.JK","ARTO.JK",
    "BJBR.JK","BJTM.JK","BNGA.JK","PNBN.JK","NISP.JK","BTPS.JK","BANK.JK",
    "BBHI.JK","BBYB.JK","SDRA.JK","AGRO.JK",
    "TLKM.JK","ISAT.JK","EXCL.JK","TOWR.JK","TBIG.JK","MTEL.JK","LINK.JK",
    "GOTO.JK","BUKA.JK","EMTK.JK","BELI.JK","MTDL.JK","DMMX.JK",
    "ASII.JK","UNTR.JK","AUTO.JK","DRMA.JK","SMSM.JK","GJTL.JK",
    "ADRO.JK","AADI.JK","ADMR.JK","PTBA.JK","ITMG.JK","INDY.JK","HRUM.JK",
    "BUMI.JK","DOID.JK","BYAN.JK","GEMS.JK","PTRO.JK","CUAN.JK",
    "MEDC.JK","PGAS.JK","PGEO.JK","ELSA.JK","ENRG.JK","AKRA.JK","RAJA.JK",
    "BREN.JK","CDIA.JK","RATU.JK",
    "ANTM.JK","INCO.JK","MDKA.JK","TINS.JK","NCKL.JK","MBMA.JK","BRMS.JK",
    "PSAB.JK","HRTA.JK","AMMN.JK","NIKL.JK",
    "UNVR.JK","ICBP.JK","INDF.JK","MYOR.JK","KLBF.JK","SIDO.JK","GGRM.JK",
    "HMSP.JK","WIIM.JK","CMRY.JK","ULTJ.JK","GOOD.JK","CLEO.JK",
    "AMRT.JK","MIDI.JK","MAPI.JK","MAPA.JK","ACES.JK","ERAA.JK","RALS.JK","LPPF.JK",
    "CPIN.JK","JPFA.JK","MAIN.JK",
    "SMGR.JK","INTP.JK","WIKA.JK","PTPP.JK","ADHI.JK",
    "CTRA.JK","BSDE.JK","PWON.JK","SMRA.JK","ASRI.JK","LPKR.JK","DMAS.JK",
    "PANI.JK","KIJA.JK","SSIA.JK",
    "JSMR.JK","POWR.JK",
    "MIKA.JK","HEAL.JK","SILO.JK","PRDA.JK",
    "TPIA.JK","BRPT.JK","ESSA.JK","AGII.JK",
    "INKP.JK","TKIM.JK",
    "AALI.JK","LSIP.JK","DSNG.JK","TAPG.JK","SSMS.JK",
    "SMDR.JK","TMAS.JK","BIRD.JK","ASSA.JK",
    "MNCN.JK","SCMA.JK","FILM.JK",
    "BFIN.JK","ADMF.JK","PNLF.JK",
    "SRTG.JK","ISSP.JK","WIFI.JK",
]

STRAT_LABELS = {
    "🌙 Beli Sore – Jual Pagi": "overnight",
    "☀️ Beli Pagi – Jual Sore": "intraday",
    "📈 Swing (tahan beberapa hari)": "swing",
}
DESKRIPSI = {
    "overnight": "Mengukur **gap semalam**: beli di harga penutupan (Close), jual di "
                 "pembukaan (Open) besok pagi. Cocok dieksekusi menjelang closing sore.",
    "intraday": "Mengukur **gerak satu hari penuh**: beli di pembukaan (Open) pagi, jual "
                "di penutupan (Close) sore di hari yang sama.",
    "swing": "Mengukur **return menahan N hari bursa** (Close→Close, blok tidak tumpang-tindih) "
             "plus struktur tren MA20/MA50 saat ini. Cocok untuk posisi beberapa hari–minggu.",
}

RINGKAS_COLS = ["Peringkat","Kode","Harga Terakhir","Net stlh Biaya %","Win Rate %",
                "t-Stat","Momentum 20h %","Tren MA","Likuiditas (Rp Jt/hr)","Skor","Layak?"]


# ============================ PERHITUNGAN ====================================
def compute_metrics(data, lookback, cost_pct, mode, hold_n=5):
    cost = cost_pct / 100.0
    rows = []
    for t, df in data.items():
        df = df.sort_index().copy()
        close, opn = df["Close"], df["Open"]
        has_vol = "Volume" in df.columns and df["Volume"].notna().any()
        df["turnover"] = (close * df["Volume"]) if has_vol else np.nan

        # tren MA (dari seluruh riwayat)
        ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else np.nan
        ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
        last_close = float(close.iloc[-1])
        if not np.isnan(ma20) and not np.isnan(ma50):
            tren = ("Naik kuat" if last_close > ma20 > ma50
                    else ("Naik" if last_close > ma20 else "Turun"))
        elif not np.isnan(ma20):
            tren = "Naik" if last_close > ma20 else "Turun"
        else:
            tren = "-"

        recent = df.tail(lookback)
        if mode == "overnight":
            ser = (opn / close.shift(1) - 1).tail(lookback).dropna()
            ser = ser[ser.abs() <= 0.20]
            min_p = max(20, min(30, lookback - 5))
        elif mode == "intraday":
            ser = (close / opn - 1).tail(lookback).dropna()
            ser = ser[ser.abs() <= 0.20]
            min_p = max(20, min(30, lookback - 5))
        else:  # swing: blok N-hari TIDAK tumpang tindih (statistik jujur)
            c = recent["Close"].dropna().values
            rets, i = [], len(c) - 1
            while i - hold_n >= 0:
                rets.append(c[i] / c[i - hold_n] - 1.0)
                i -= hold_n
            ser = pd.Series(rets)
            ser = ser[ser.abs() <= 0.60]
            min_p = 8
        if len(ser) < min_p:
            continue

        mean_r = ser.mean(); std = ser.std(); win = (ser > 0).mean()
        net = mean_r - cost
        n = len(ser)
        t_stat = (mean_r / (std / math.sqrt(n))) if std and std > 0 else 0.0
        avg_turn = recent["turnover"].mean() if has_vol else np.nan
        mom5 = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) > 6 else np.nan
        mom20 = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else np.nan

        rows.append({
            "Kode": t.replace(".JK", ""),
            "Harga Terakhir": round(last_close, 2),
            "Avg Return %": round(mean_r * 100, 3),
            "Net stlh Biaya %": round(net * 100, 3),
            "Win Rate %": round(win * 100, 1),
            "t-Stat": round(t_stat, 2),
            "Volatilitas %": round(std * 100, 3),
            "Tren MA": tren,
            "Likuiditas (Rp Jt/hr)": (round(avg_turn / 1e6, 0)
                                      if not np.isnan(avg_turn) else None),
            "Momentum 5h %": round(mom5 * 100, 2) if not np.isnan(mom5) else None,
            "Momentum 20h %": round(mom20 * 100, 2) if not np.isnan(mom20) else None,
            "Periode": int(n),
            "Skor": round((net * 100.0) * win, 4),
        })
    return pd.DataFrame(rows)


def tetapkan_verdict(df, minliq_jt, mode, per_label):
    if df.empty:
        df["Layak?"] = []; df["Catatan Analisa"] = []
        return df

    def _bersih(v):
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    verdicts, catatans = [], []
    for _, r in df.iterrows():
        net, win, t = r["Net stlh Biaya %"], r["Win Rate %"], r["t-Stat"]
        mom = _bersih(r["Momentum 20h %"])
        liq = _bersih(r["Likuiditas (Rp Jt/hr)"])
        tren = r["Tren MA"]; hari = r["Periode"]
        liq_ok = (liq is None) or (liq >= minliq_jt)

        if mode == "swing":
            tren_ok = tren in ("Naik", "Naik kuat")
            gerbang = {
                f"net {net:+.3f}%{per_label}": net > 0,
                f"win rate {win:.0f}%": win >= 55,
                f"tren MA ({tren})": tren_ok,
            }
        else:
            mom_ok = (mom is None) or (mom > 0)
            gerbang = {
                f"net {net:+.3f}%{per_label}": net > 0,
                f"win rate {win:.0f}%": win >= 55,
                ("momentum 20h " + (f"{mom:+.1f}%" if mom is not None else "n/a")): mom_ok,
            }
        lolos = sum(gerbang.values())
        signifikan = t >= 1.5

        liq_txt = (f"; likuid Rp {liq/1000:.1f} M/hr → mudah keluar-masuk"
                   if liq is not None else "")
        tren_txt = f"; struktur tren {tren} (MA20/MA50)" if tren != "-" else ""

        if lolos == 3 and signifikan and liq_ok:
            v = "Layak"
            kekuatan = "kuat" if t >= 2 else "cukup meyakinkan"
            c = (f"LAYAK diriset: untung bersih {net:+.3f}%{per_label} stabil dari "
                 f"{hari} periode (menang {win:.0f}%; t={t:.1f} → edge {kekuatan}, "
                 f"bukan kebetulan){tren_txt}{liq_txt}.")
        elif lolos == 3 and liq_ok and not signifikan:
            v = "Hampir"
            c = (f"Hampir: semua syarat lolos TAPI edge belum signifikan (t={t:.1f} < 1.5) "
                 f"— bisa kebetulan. Pantau beberapa minggu lagi.")
        elif lolos == 2 and net > -0.05:
            v = "Hampir"
            gagal = [k for k, ok in gerbang.items() if not ok]
            c = f"Hampir layak — hanya gagal di: {gagal[0]}. Pantau."
        else:
            v = "Tidak"
            gagal = [k for k, ok in gerbang.items() if not ok]
            c = "Tidak lolos: " + "; ".join(gagal) + "."
        if not liq_ok and v != "Tidak":
            v = "Hampir"
            c += f" (Likuiditas Rp {liq/1000:.1f} M/hr di bawah ambang — hati-hati slippage.)"
        verdicts.append(v); catatans.append(c)

    df = df.copy()
    df["Layak?"] = verdicts
    df["Catatan Analisa"] = catatans
    return df


def rank(df, minliq_jt):
    if df.empty:
        return df
    liq = pd.to_numeric(df["Likuiditas (Rp Jt/hr)"], errors="coerce")
    mask = liq.isna() | (liq >= minliq_jt)
    out = df[mask].copy()
    if len(out) < 5:
        out = df.copy()
    out = out.sort_values(["Skor", "t-Stat"], ascending=False).reset_index(drop=True)
    out.insert(0, "Peringkat", range(1, len(out) + 1))
    return out


# ============================ SUMBER DATA ====================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live(tickers_tuple, lookback):
    import yfinance as yf
    period = f"{max(lookback + 60, 140)}d"
    tickers = list(tickers_tuple)
    data, gagal = {}, []
    CHUNK = 40
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        try:
            raw = yf.download(chunk, period=period, interval="1d",
                              auto_adjust=True, group_by="ticker",
                              threads=True, progress=False)
        except Exception:
            gagal.extend(chunk); continue
        if raw is None or raw.empty:
            gagal.extend(chunk); continue
        for t in chunk:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(
                    subset=["Open", "Close"])
                if len(df) >= 5:
                    data[t] = df
                else:
                    gagal.append(t)
            except Exception:
                gagal.append(t)
    return data, gagal


def parse_csv(file):
    raw = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in raw.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    c_date = pick("date","tanggal","datetime"); c_tick = pick("ticker","kode","symbol","saham","stock")
    c_open = pick("open","buka","pembukaan"); c_close = pick("close","tutup","penutupan","last")
    c_high = pick("high","tertinggi"); c_low = pick("low","terendah"); c_vol = pick("volume","vol")
    missing = [l for l, c in [("Date",c_date),("Ticker",c_tick),("Open",c_open),("Close",c_close)] if c is None]
    if missing:
        raise ValueError("Kolom wajib tidak ada: " + ", ".join(missing))
    df = pd.DataFrame({
        "Date": pd.to_datetime(raw[c_date], errors="coerce"),
        "Ticker": raw[c_tick].astype(str).str.upper().str.replace(".JK","",regex=False),
        "Open": pd.to_numeric(raw[c_open], errors="coerce"),
        "Close": pd.to_numeric(raw[c_close], errors="coerce"),
        "High": pd.to_numeric(raw[c_high], errors="coerce") if c_high else np.nan,
        "Low": pd.to_numeric(raw[c_low], errors="coerce") if c_low else np.nan,
        "Volume": pd.to_numeric(raw[c_vol], errors="coerce") if c_vol else np.nan,
    }).dropna(subset=["Date","Ticker","Open","Close"])
    data = {}
    for tk, g in df.groupby("Ticker"):
        g = g.sort_values("Date").set_index("Date")[["Open","High","Low","Close","Volume"]]
        if len(g) >= 5:
            data[tk + ".JK"] = g
    return data


def data_ke_long_csv(data):
    """Gabungkan dict harga -> CSV long (cadangan utk fitur Upload CSV)."""
    frames = []
    for t, df in data.items():
        g = df.copy()
        g.index.name = "Date"
        g = g.reset_index()
        g["Ticker"] = t.replace(".JK", "")
        g["Date"] = pd.to_datetime(g["Date"]).dt.date
        frames.append(g[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]])
    out = pd.concat(frames, ignore_index=True)
    return out.to_csv(index=False).encode("utf-8")


def demo_data(tickers, lookback, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=dt.date.today(), periods=lookback + 70)
    data = {}
    for i, t in enumerate(tickers):
        price = 500.0 * (1 + (i % 10) * 0.4)
        drift = rng.normal(0.0015, 0.0022)
        rows = []
        for d in dates:
            prev = price
            op = prev * (1 + rng.normal(drift, 0.010))
            cp = op * (1 + rng.normal(drift * 0.6, 0.013))
            hi = max(op, cp) * (1 + abs(rng.normal(0, 0.004)))
            lo = min(op, cp) * (1 - abs(rng.normal(0, 0.004)))
            vol = int(rng.integers(3_000_000, 90_000_000))
            rows.append((d, op, hi, lo, cp, vol)); price = cp
        data[t] = pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Volume"]).set_index("Date")
    return data


# ============================ TAMPILAN =======================================
def style_table(df):
    def warna(v):
        if v == "Layak":  return "background-color:#C6EFCE;color:#006100;font-weight:bold"
        if v == "Hampir": return "background-color:#FFEB9C;color:#9C6500;font-weight:bold"
        if v == "Tidak":  return "background-color:#FFC7CE;color:#9C0006"
        return ""
    def warna_tren(v):
        if v == "Naik kuat": return "color:#067d37;font-weight:700"
        if v == "Naik":      return "color:#067d37"
        if v == "Turun":     return "color:#c00000"
        return ""
    base = df.style
    _map = getattr(base, "map", None) or base.applymap
    sty = base
    if "Layak?" in df.columns:
        sty = _map(warna, subset=["Layak?"])
        _map = getattr(sty, "map", None) or sty.applymap
    if "Tren MA" in df.columns:
        sty = _map(warna_tren, subset=["Tren MA"])
    num = {"Avg Return %":"{:+.3f}","Net stlh Biaya %":"{:+.3f}","Volatilitas %":"{:.3f}",
           "Win Rate %":"{:.1f}","t-Stat":"{:.2f}","Momentum 5h %":"{:+.2f}",
           "Momentum 20h %":"{:+.2f}","Skor":"{:.4f}","Harga Terakhir":"{:,.0f}",
           "Likuiditas (Rp Jt/hr)":"{:,.0f}"}
    return sty.format({k: v for k, v in num.items() if k in df.columns}, na_rep="-")


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Ranking")
    return buf.getvalue()


def main():
    st.title("📈 Screener Saham BEI — 3 Strategi")

    pilih = st.radio("Pilih strategi:", list(STRAT_LABELS.keys()), horizontal=True)
    mode = STRAT_LABELS[pilih]
    st.caption(DESKRIPSI[mode] + " **Alat bantu riset — bukan rekomendasi/sinyal beli-jual.**")

    with st.expander("⚠️ Cara baca & arti 'Layak' (klik)"):
        st.markdown(
            "- **Net stlh Biaya %** — rata-rata untung per transaksi SETELAH biaya. "
            "Untuk swing, biaya hanya dihitung sekali per siklus tahan.\n"
            "- **t-Stat** — uji statistik: ≥2 edge kuat, 1.5–2 cukup, <1.5 bisa kebetulan.\n"
            "- **Tren MA** — posisi harga vs rata-rata 20 & 50 hari: *Naik kuat* = harga > MA20 > MA50.\n"
            "- **✅ Layak** = net>0 + win rate ≥55% + tren mendukung + t≥1.5 + cukup likuid. "
            "Syarat ketat — wajar bila hanya 0–5 saham.\n"
            "- **🟡 Hampir** = gagal tipis di satu syarat → kandidat pantauan.\n"
            "- **'Layak: 0' bukan error** — artinya strategi ini sedang tidak menguntungkan "
            "setelah biaya. Itu informasi berharga.\n"
            "- Kinerja masa lalu tidak menjamin hasil ke depan. Risiko di tangan Anda."
        )

    sb = st.sidebar
    sb.header("⚙️ Pengaturan")
    sumber = sb.radio("Sumber data", ["Live (Yahoo Finance)", "Upload CSV", "Data contoh (demo)"])

    hold_n = 5
    if mode == "swing":
        hold_n = sb.slider("Lama tahan (hari bursa)", 3, 20, 5,
                           help="Berapa hari posisi ditahan sebelum dijual.")
    default_lb = 120 if mode == "swing" else 60
    lookback = sb.slider("Hari bursa dianalisis", 20, 240, default_lb, 5,
                         help="Swing butuh riwayat lebih panjang (disarankan ≥120).")
    if mode == "swing" and lookback < hold_n * 10:
        sb.warning(f"Untuk tahan {hold_n} hr, sebaiknya 'Hari dianalisis' ≥ {hold_n*10} "
                   f"agar sampel cukup.")
    cost = sb.number_input("Biaya bolak-balik (%)", 0.0, 2.0, 0.40, 0.05,
                           help="Fee beli + jual broker Anda. Contoh 0.15+0.25 = 0.40")
    minliq = sb.number_input("Likuiditas min (Rp Juta/hari)", 0, 500_000, 5_000, 500)
    top_n = sb.slider("Tampilkan Top-N", 5, 150, 25)
    tampil = sb.radio("Tampilkan", ["Semua", "Layak + Hampir", "Hanya Layak"])
    semua_kolom = sb.checkbox("Tampilkan semua kolom")

    uploaded, tickers = None, DEFAULT_TICKERS
    if sumber == "Upload CSV":
        uploaded = sb.file_uploader("Upload CSV harga", type=["csv"])
        sb.caption("Format: Date, Ticker, Open, Close (wajib) + High, Low, Volume (opsional).")
    elif sumber == "Live (Yahoo Finance)":
        teks = sb.text_area("Daftar saham (pisah koma)",
                            ", ".join(t.replace(".JK","") for t in DEFAULT_TICKERS), height=140)
        tickers = [x.strip().upper() + (".JK" if not x.strip().upper().endswith(".JK") else "")
                   for x in teks.split(",") if x.strip()]
        sb.caption(f"{len(tickers)} saham — penarikan batch ±1–2 menit (cache 30 menit).")
        if sb.button("🔄 Refresh data (hapus cache)"):
            fetch_live.clear()

    # ---------- data ----------
    if sumber == "Live (Yahoo Finance)":
        with st.spinner(f"Menarik {len(tickers)} saham dari Yahoo (batch)..."):
            data, gagal = fetch_live(tuple(tickers), lookback)
        if not data:
            st.error("Gagal menarik data live. Coba Refresh, atau pakai Upload CSV / Data contoh.")
            return
        if gagal:
            st.caption(f"ℹ️ {len(gagal)} kode dilewati: "
                       + ", ".join(g.replace(".JK","") for g in gagal[:15])
                       + (" ..." if len(gagal) > 15 else ""))
    elif sumber == "Upload CSV":
        if uploaded is None:
            st.info("⬅️ Upload file CSV di panel kiri, atau pilih Data contoh.")
            return
        try:
            data = parse_csv(uploaded)
        except Exception as e:
            st.error(f"CSV tidak terbaca: {e}"); return
        if not data:
            st.error("Tidak ada saham dengan data cukup."); return
    else:
        data = demo_data(DEFAULT_TICKERS[:60], max(lookback, 120))
        st.warning("Mode **DATA CONTOH (simulasi)** — angka PALSU, hanya untuk lihat tampilan.")

    # ---------- hitung ----------
    per_label = {"overnight": "/malam", "intraday": "/hari",
                 "swing": f"/{hold_n} hr"}[mode]
    met = compute_metrics(data, lookback, cost, mode, hold_n)
    if met.empty:
        st.error("Data terlalu pendek untuk strategi & periode ini. "
                 "Perbesar 'Hari dianalisis' atau perkecil 'Lama tahan'.")
        return
    met = tetapkan_verdict(met, minliq, mode, per_label)
    ranked = rank(met, minliq)

    n_layak = int((met["Layak?"] == "Layak").sum())
    n_hampir = int((met["Layak?"] == "Hampir").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saham dihitung", len(met))
    c2.metric("✅ Layak", n_layak)
    c3.metric("🟡 Hampir", n_hampir)
    c4.metric("Tanggal", dt.date.today().isoformat())

    if n_layak == 0:
        st.info(f"**Tidak ada yang 'Layak' untuk strategi {pilih} saat ini — itu temuan, "
                f"bukan error.** Lihat tingkat 🟡 Hampir dan Catatan Analisa-nya.")

    if tampil == "Hanya Layak":
        ranked = ranked[ranked["Layak?"] == "Layak"].reset_index(drop=True)
    elif tampil == "Layak + Hampir":
        ranked = ranked[ranked["Layak?"].isin(["Layak", "Hampir"])].reset_index(drop=True)

    show = ranked.head(top_n)
    tampil_cols = (list(show.columns) if semua_kolom
                   else [c for c in RINGKAS_COLS if c in show.columns])
    st.dataframe(style_table(show[tampil_cols]), use_container_width=True, height=520)

    st.subheader("📝 Catatan Analisa")
    st.caption(f"Strategi: {pilih} — alasan eksplisit per saham (Top 10 dari tampilan di atas).")
    ikon = {"Layak": "✅", "Hampir": "🟡", "Tidak": "❌"}
    for _, r in show.head(10).iterrows():
        st.markdown(f"{ikon.get(r['Layak?'],'')} **{int(r['Peringkat'])}. {r['Kode']}** — {r['Catatan Analisa']}")

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Hasil CSV", ranked.to_csv(index=False).encode("utf-8"),
                       f"hasil_{mode}_{dt.date.today()}.csv", "text/csv",
                       use_container_width=True)
    d2.download_button("⬇️ Hasil Excel", to_excel_bytes(ranked),
                       f"hasil_{mode}_{dt.date.today()}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
    if sumber == "Live (Yahoo Finance)":
        d3.download_button("💾 Data mentah (cadangan)", data_ke_long_csv(data),
                           f"data_saham_{dt.date.today()}.csv", "text/csv",
                           use_container_width=True,
                           help="Simpan sebagai cadangan. Bisa di-upload lagi lewat menu "
                                "'Upload CSV' kalau suatu saat Live gagal.")

    st.caption("Urutan berdasar **Skor = Net% × Win Rate** (seri: t-Stat). Walau 'Layak', "
               "tetap cek berita & bid-offer sebelum ambil posisi.")


if __name__ == "__main__":
    main()
