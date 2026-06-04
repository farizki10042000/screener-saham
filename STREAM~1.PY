#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screener Saham "Beli Sore - Jual Pagi" (Overnight) - versi Web / HP (Streamlit)

Jalankan lokal:  streamlit run streamlit_app.py
Deploy ke HP  :  lihat PANDUAN_STREAMLIT.md (share.streamlit.io)

Sumber data: Live (Yahoo Finance) ATAU Upload CSV ATAU Data contoh (demo).
Ini ALAT BANTU RISET, bukan nasihat keuangan / sinyal beli-jual.
"""

import io
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
#  KONFIGURASI
# =============================================================================
st.set_page_config(page_title="Screener Beli Sore - Jual Pagi",
                   page_icon="📈", layout="wide")

DEFAULT_TICKERS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "BBTN.JK", "ARTO.JK",
    "TLKM.JK", "ISAT.JK", "EXCL.JK", "TOWR.JK", "TBIG.JK", "MTEL.JK",
    "ASII.JK", "UNTR.JK",
    "ADRO.JK", "PTBA.JK", "ITMG.JK", "INDY.JK", "HRUM.JK", "MEDC.JK", "PGAS.JK",
    "AKRA.JK", "ELSA.JK",
    "ANTM.JK", "INCO.JK", "MDKA.JK", "TINS.JK",
    "UNVR.JK", "ICBP.JK", "INDF.JK", "MYOR.JK", "KLBF.JK", "SIDO.JK",
    "GGRM.JK", "HMSP.JK", "CPIN.JK", "JPFA.JK", "SMGR.JK", "INTP.JK", "JSMR.JK",
    "AMRT.JK", "MAPI.JK", "ACES.JK", "ERAA.JK",
    "GOTO.JK", "BUKA.JK", "EMTK.JK",
    "TPIA.JK", "BRPT.JK", "ESSA.JK",
    "CTRA.JK", "BSDE.JK", "PWON.JK", "SMRA.JK", "MIKA.JK", "HEAL.JK",
]

LAYAK_MIN_NET_PCT = 0.0
LAYAK_MIN_WINRATE = 55.0
LAYAK_MIN_MOM20 = 0.0

RINGKAS_COLS = ["Peringkat", "Kode", "Harga Terakhir", "Net stlh Biaya %",
                "Win Rate %", "Momentum 20h %", "Likuiditas (Rp Jt/hr)",
                "Skor", "Layak?"]


# =============================================================================
#  PERHITUNGAN  (sama dengan versi CLI)
# =============================================================================
def compute_metrics(data, lookback, cost_pct, min_days):
    cost = cost_pct / 100.0
    rows = []
    for t, df in data.items():
        df = df.sort_index().copy()
        df["prev_close"] = df["Close"].shift(1)
        df["overnight"] = df["Open"] / df["prev_close"] - 1.0
        has_vol = "Volume" in df.columns and df["Volume"].notna().any()
        df["turnover"] = (df["Close"] * df["Volume"]) if has_vol else np.nan

        recent = df.tail(lookback)
        on = recent["overnight"].dropna()
        if len(on) < min_days:
            continue

        mean_on = on.mean()
        median_on = on.median()
        win = (on > 0).mean()
        std = on.std()
        net = mean_on - cost
        sharpe = mean_on / std if std and std > 0 else 0.0
        avg_turnover = recent["turnover"].mean() if has_vol else np.nan
        last_close = df["Close"].iloc[-1]

        mom5 = (df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) if len(df) > 6 else np.nan
        mom20 = (df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) if len(df) > 21 else np.nan

        skor = (net * 100.0) * win
        layak = ("Ya" if (net * 100.0 > LAYAK_MIN_NET_PCT
                          and win * 100.0 >= LAYAK_MIN_WINRATE
                          and (np.isnan(mom20) or mom20 * 100.0 >= LAYAK_MIN_MOM20))
                 else "Tidak")

        rows.append({
            "Kode": t.replace(".JK", ""),
            "Harga Terakhir": round(float(last_close), 2),
            "Overnight Avg %": round(mean_on * 100, 3),
            "Net stlh Biaya %": round(net * 100, 3),
            "Win Rate %": round(win * 100, 1),
            "Volatilitas %": round(std * 100, 3),
            "Sharpe/Trade": round(sharpe, 3),
            "Likuiditas (Rp Jt/hr)": (round(avg_turnover / 1e6, 0)
                                      if not np.isnan(avg_turnover) else None),
            "Momentum 5h %": round(mom5 * 100, 2) if not np.isnan(mom5) else None,
            "Momentum 20h %": round(mom20 * 100, 2) if not np.isnan(mom20) else None,
            "Jml Hari": int(len(on)),
            "Skor": round(skor, 4),
            "Layak?": layak,
        })
    return pd.DataFrame(rows)


def apply_filters_and_rank(df, min_turnover):
    if df.empty:
        return df
    df = df.copy()
    liq = df["Likuiditas (Rp Jt/hr)"] * 1e6
    # NaN (tanpa volume) dianggap lolos supaya tetap tampil
    mask = liq.isna() | (liq >= min_turnover)
    out = df[mask].copy()
    if len(out) < 5:
        out = df
    out = out.sort_values("Skor", ascending=False).reset_index(drop=True)
    out.insert(0, "Peringkat", range(1, len(out) + 1))
    return out


# =============================================================================
#  SUMBER DATA
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live(tickers_tuple, lookback):
    """Tarik dari Yahoo Finance. Hasil di-cache 30 menit."""
    import yfinance as yf
    period_days = max(lookback + 50, 130)
    data, gagal = {}, []
    for t in tickers_tuple:
        try:
            df = yf.download(t, period=f"{period_days}d", interval="1d",
                             progress=False, auto_adjust=False)
            if df is None or df.empty:
                gagal.append(t); continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) >= 5:
                data[t] = df
            else:
                gagal.append(t)
        except Exception:
            gagal.append(t)
    return data, gagal


def parse_csv(file):
    """Baca CSV format long. Kolom (tidak peka huruf besar/kecil):
    date, ticker/kode, open, close  (wajib); high, low, volume (opsional)."""
    raw = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in raw.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_date = pick("date", "tanggal", "datetime")
    c_tick = pick("ticker", "kode", "symbol", "saham", "stock")
    c_open = pick("open", "buka", "pembukaan")
    c_close = pick("close", "tutup", "penutupan", "last")
    c_high = pick("high", "tertinggi")
    c_low = pick("low", "terendah")
    c_vol = pick("volume", "vol", "volume_lembar")

    missing = [lbl for lbl, c in [("Date", c_date), ("Ticker/Kode", c_tick),
                                  ("Open", c_open), ("Close", c_close)] if c is None]
    if missing:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(missing)
                         + ". Minimal harus ada: Date, Ticker, Open, Close.")

    df = pd.DataFrame({
        "Date": pd.to_datetime(raw[c_date], errors="coerce"),
        "Ticker": raw[c_tick].astype(str).str.upper().str.replace(".JK", "", regex=False),
        "Open": pd.to_numeric(raw[c_open], errors="coerce"),
        "Close": pd.to_numeric(raw[c_close], errors="coerce"),
    })
    df["High"] = pd.to_numeric(raw[c_high], errors="coerce") if c_high else np.nan
    df["Low"] = pd.to_numeric(raw[c_low], errors="coerce") if c_low else np.nan
    df["Volume"] = pd.to_numeric(raw[c_vol], errors="coerce") if c_vol else np.nan
    df = df.dropna(subset=["Date", "Ticker", "Open", "Close"])

    data = {}
    for tk, g in df.groupby("Ticker"):
        g = g.sort_values("Date").set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        if len(g) >= 5:
            data[tk + ".JK"] = g
    return data


def demo_data(tickers, lookback, seed=7):
    rng = np.random.default_rng(seed)
    n = lookback + 60
    dates = pd.bdate_range(end=dt.date.today(), periods=n)
    data = {}
    for i, t in enumerate(tickers):
        price = 500.0 * (1 + (i % 10) * 0.4)
        on_drift = rng.normal(0.0015, 0.0022)
        rows = []
        for d in dates:
            prev = price
            overnight = rng.normal(on_drift, 0.010)
            op = prev * (1 + overnight)
            cp = op * (1 + rng.normal(-0.0002, 0.013))
            hi = max(op, cp) * (1 + abs(rng.normal(0, 0.004)))
            lo = min(op, cp) * (1 - abs(rng.normal(0, 0.004)))
            vol = int(rng.integers(3_000_000, 90_000_000))
            rows.append((d, op, hi, lo, cp, vol)); price = cp
        data[t] = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low",
                                              "Close", "Volume"]).set_index("Date")
    return data


# =============================================================================
#  TAMPILAN
# =============================================================================
def style_table(df):
    def warna_layak(v):
        if v == "Ya":
            return "background-color:#C6EFCE;color:#006100;font-weight:bold"
        if v == "Tidak":
            return "background-color:#FFC7CE;color:#9C0006"
        return ""
    sty = df.style.applymap(warna_layak, subset=["Layak?"])
    num = {c: "{:.3f}" for c in ["Overnight Avg %", "Net stlh Biaya %", "Volatilitas %"]}
    num.update({"Win Rate %": "{:.1f}", "Momentum 5h %": "{:.2f}",
                "Momentum 20h %": "{:.2f}", "Skor": "{:.4f}",
                "Harga Terakhir": "{:,.0f}", "Likuiditas (Rp Jt/hr)": "{:,.0f}"})
    return sty.format({k: v for k, v in num.items() if k in df.columns})


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Ranking")
    return buf.getvalue()


def main():
    st.title("📈 Screener Beli Sore – Jual Pagi")
    st.caption("Cari saham yang historisnya cenderung *gap-up* semalam "
               "(beli di Close sore, jual di Open pagi). **Alat bantu riset — "
               "bukan rekomendasi / sinyal beli-jual.**")

    with st.expander("⚠️ Baca dulu — disclaimer & cara baca"):
        st.markdown(
            "- Kinerja masa lalu **tidak menjamin** hasil ke depan.\n"
            "- Biaya transaksi (~0,4–0,5% bolak-balik) sering memakan untung gap. "
            "Kolom **Net stlh Biaya %** sudah memperhitungkannya.\n"
            "- **Win Rate %**: persen hari yang gap-nya positif (>55% relatif konsisten).\n"
            "- **Likuiditas**: rata-rata nilai transaksi/hari — makin besar makin "
            "mudah masuk/keluar. Hindari saham tidak likuid untuk ditahan semalam.\n"
            "- **Layak?**: penanda otomatis (Net>0, WinRate≥55%, Momentum20>0). "
            "Tetap **bukan rekomendasi** — cek lagi berita & kondisi saham."
        )

    sb = st.sidebar
    sb.header("⚙️ Pengaturan")
    sumber = sb.radio("Sumber data", ["Live (Yahoo Finance)", "Upload CSV",
                                      "Data contoh (demo)"])
    lookback = sb.slider("Hari bursa dianalisis", 20, 180, 60, 5)
    cost = sb.number_input("Biaya bolak-balik (%)", 0.0, 2.0, 0.40, 0.05,
                           help="Sesuaikan dengan fee broker Anda (beli+jual).")
    min_turn_m = sb.number_input("Likuiditas min (Rp Juta/hari)", 0, 200000, 5000, 500)
    top_n = sb.slider("Tampilkan Top-N", 5, 60, 20)
    hanya_layak = sb.checkbox("Hanya tampilkan 'Layak'")
    semua_kolom = sb.checkbox("Tampilkan semua kolom")

    uploaded, tickers = None, DEFAULT_TICKERS
    if sumber == "Upload CSV":
        uploaded = sb.file_uploader("Upload CSV harga", type=["csv"])
        sb.caption("Format: Date, Ticker, Open, Close (wajib) + High, Low, Volume "
                   "(opsional). Lihat contoh_format_data.csv.")
    elif sumber == "Live (Yahoo Finance)":
        teks = sb.text_area("Daftar saham (pisah koma)",
                            ", ".join(t.replace(".JK", "") for t in DEFAULT_TICKERS),
                            height=90)
        tickers = [x.strip().upper() + (".JK" if not x.strip().upper().endswith(".JK")
                   else "") for x in teks.split(",") if x.strip()]
        if sb.button("🔄 Refresh data (hapus cache)"):
            fetch_live.clear()

    # ---- ambil data ----
    data, catatan = {}, ""
    if sumber == "Live (Yahoo Finance)":
        with st.spinner("Menarik data dari Yahoo Finance..."):
            data, gagal = fetch_live(tuple(tickers), lookback)
        if not data:
            st.error("Gagal menarik data live (mungkin host dibatasi / Yahoo "
                     "rate-limit). Coba **Upload CSV** atau **Data contoh**.")
            return
        if gagal:
            catatan = f"{len(gagal)} kode gagal/dilewati: {', '.join(g.replace('.JK','') for g in gagal[:12])}" + (" ..." if len(gagal) > 12 else "")
    elif sumber == "Upload CSV":
        if uploaded is None:
            st.info("⬅️ Upload file CSV dulu di panel kiri, atau pilih **Data contoh**.")
            return
        try:
            data = parse_csv(uploaded)
        except Exception as e:
            st.error(f"CSV tidak terbaca: {e}")
            return
        if not data:
            st.error("Tidak ada saham dengan data cukup (min 5 baris/saham).")
            return
    else:
        data = demo_data(DEFAULT_TICKERS, lookback)
        st.warning("Mode **DATA CONTOH (simulasi)** — angka ini PALSU, hanya untuk "
                   "melihat tampilan. Pilih Live/Upload CSV untuk data nyata.")

    # ---- hitung & ranking ----
    metrics = compute_metrics(data, lookback, cost, min_days=min(30, lookback))
    ranked = apply_filters_and_rank(metrics, min_turn_m * 1e6)
    if ranked.empty:
        st.error("Tidak ada saham yang lolos perhitungan.")
        return
    if hanya_layak:
        ranked = ranked[ranked["Layak?"] == "Ya"].reset_index(drop=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Saham dihitung", len(metrics))
    c2.metric("Ditandai 'Layak'", int((metrics["Layak?"] == "Ya").sum()))
    c3.metric("Tanggal", dt.date.today().isoformat())
    if catatan:
        st.caption("ℹ️ " + catatan)

    show = ranked.head(top_n)
    if not semua_kolom:
        show = show[[c for c in RINGKAS_COLS if c in show.columns]]
    st.dataframe(style_table(show), use_container_width=True, height=560)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Unduh CSV", ranked.to_csv(index=False).encode("utf-8"),
                       f"hasil_screener_{dt.date.today()}.csv", "text/csv",
                       use_container_width=True)
    d2.download_button("⬇️ Unduh Excel", to_excel_bytes(ranked),
                       f"hasil_screener_{dt.date.today()}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

    st.caption("Diurutkan berdasarkan **Skor** = Net% × Win Rate. "
               "Fokus: Net positif + Win Rate tinggi + Likuiditas besar + Momentum positif.")


if __name__ == "__main__":
    main()
