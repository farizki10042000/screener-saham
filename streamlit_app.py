#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screener "Beli Sore - Jual Pagi" (Overnight) v2 - Streamlit
============================================================
Baru di v2:
- Universe ~140 saham likuid BEI (bisa diedit).
- Penarikan data BATCH + paralel -> jauh lebih cepat.
- Harga terkoreksi split & dividen (auto_adjust) -> gap palsu hilang.
- Uji statistik t-stat -> membedakan edge nyata vs kebetulan.
- Verdict 3 tingkat: Layak / Hampir / Tidak + CATATAN ANALISA per saham.

ALAT BANTU RISET - bukan nasihat keuangan / sinyal beli-jual.
"""

import io
import datetime as dt
import math

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Screener Beli Sore - Jual Pagi",
                   page_icon="📈", layout="wide")

# ====================== UNIVERSE ~140 SAHAM LIKUID BEI =======================
DEFAULT_TICKERS = [
    # Bank
    "BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK","BRIS.JK","BBTN.JK","ARTO.JK",
    "BJBR.JK","BJTM.JK","BNGA.JK","PNBN.JK","NISP.JK","BTPS.JK","BANK.JK",
    "BBHI.JK","BBYB.JK","SDRA.JK","AGRO.JK",
    # Telko & menara
    "TLKM.JK","ISAT.JK","EXCL.JK","TOWR.JK","TBIG.JK","MTEL.JK","LINK.JK",
    # Teknologi
    "GOTO.JK","BUKA.JK","EMTK.JK","BELI.JK","MTDL.JK","DMMX.JK",
    # Otomotif & komponen
    "ASII.JK","UNTR.JK","AUTO.JK","DRMA.JK","SMSM.JK","GJTL.JK",
    # Batu bara & kontraktor tambang
    "ADRO.JK","AADI.JK","ADMR.JK","PTBA.JK","ITMG.JK","INDY.JK","HRUM.JK",
    "BUMI.JK","DOID.JK","BYAN.JK","GEMS.JK","PTRO.JK","CUAN.JK",
    # Migas & energi baru
    "MEDC.JK","PGAS.JK","PGEO.JK","ELSA.JK","ENRG.JK","AKRA.JK","RAJA.JK",
    "BREN.JK","CDIA.JK","RATU.JK",
    # Logam & mineral
    "ANTM.JK","INCO.JK","MDKA.JK","TINS.JK","NCKL.JK","MBMA.JK","BRMS.JK",
    "PSAB.JK","HRTA.JK","AMMN.JK","NIKL.JK",
    # Konsumer
    "UNVR.JK","ICBP.JK","INDF.JK","MYOR.JK","KLBF.JK","SIDO.JK","GGRM.JK",
    "HMSP.JK","WIIM.JK","CMRY.JK","ULTJ.JK","GOOD.JK","CLEO.JK",
    # Ritel
    "AMRT.JK","MIDI.JK","MAPI.JK","MAPA.JK","ACES.JK","ERAA.JK","RALS.JK","LPPF.JK",
    # Unggas
    "CPIN.JK","JPFA.JK","MAIN.JK",
    # Semen & konstruksi
    "SMGR.JK","INTP.JK","WIKA.JK","PTPP.JK","ADHI.JK",
    # Properti
    "CTRA.JK","BSDE.JK","PWON.JK","SMRA.JK","ASRI.JK","LPKR.JK","DMAS.JK",
    "PANI.JK","KIJA.JK","SSIA.JK",
    # Infrastruktur & utilitas
    "JSMR.JK","POWR.JK",
    # Kesehatan
    "MIKA.JK","HEAL.JK","SILO.JK","PRDA.JK",
    # Petrokimia & gas industri
    "TPIA.JK","BRPT.JK","ESSA.JK","AGII.JK",
    # Kertas
    "INKP.JK","TKIM.JK",
    # Perkebunan
    "AALI.JK","LSIP.JK","DSNG.JK","TAPG.JK","SSMS.JK",
    # Pelayaran & logistik
    "SMDR.JK","TMAS.JK","BIRD.JK","ASSA.JK",
    # Media
    "MNCN.JK","SCMA.JK","FILM.JK",
    # Pembiayaan & asuransi
    "BFIN.JK","ADMF.JK","PNLF.JK",
    # Lain-lain likuid
    "SRTG.JK","ISSP.JK","WIFI.JK",
]

RINGKAS_COLS = ["Peringkat","Kode","Harga Terakhir","Net stlh Biaya %","Win Rate %",
                "t-Stat","Momentum 20h %","Likuiditas (Rp Jt/hr)","Skor","Layak?"]


# ============================ PERHITUNGAN ====================================
def compute_metrics(data, lookback, cost_pct):
    """Hitung metrik overnight mentah per saham (verdict ditetapkan terpisah)."""
    cost = cost_pct / 100.0
    min_days = max(20, min(30, lookback - 5))
    rows = []
    for t, df in data.items():
        df = df.sort_index().copy()
        df["prev_close"] = df["Close"].shift(1)
        df["overnight"] = df["Open"] / df["prev_close"] - 1.0
        has_vol = "Volume" in df.columns and df["Volume"].notna().any()
        df["turnover"] = (df["Close"] * df["Volume"]) if has_vol else np.nan

        recent = df.tail(lookback)
        on = recent["overnight"].dropna()
        # buang outlier ekstrem >20% (data error / suspend) supaya presisi
        on = on[on.abs() <= 0.20]
        if len(on) < min_days:
            continue

        mean_on = on.mean(); std = on.std(); win = (on > 0).mean()
        net = mean_on - cost
        n = len(on)
        t_stat = (mean_on / (std / math.sqrt(n))) if std and std > 0 else 0.0
        avg_turn = recent["turnover"].mean() if has_vol else np.nan
        last_close = df["Close"].iloc[-1]
        mom5 = (df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) if len(df) > 6 else np.nan
        mom20 = (df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) if len(df) > 21 else np.nan

        rows.append({
            "Kode": t.replace(".JK", ""),
            "Harga Terakhir": round(float(last_close), 2),
            "Overnight Avg %": round(mean_on * 100, 3),
            "Net stlh Biaya %": round(net * 100, 3),
            "Win Rate %": round(win * 100, 1),
            "t-Stat": round(t_stat, 2),
            "Volatilitas %": round(std * 100, 3),
            "Likuiditas (Rp Jt/hr)": (round(avg_turn / 1e6, 0)
                                      if not np.isnan(avg_turn) else None),
            "Momentum 5h %": round(mom5 * 100, 2) if not np.isnan(mom5) else None,
            "Momentum 20h %": round(mom20 * 100, 2) if not np.isnan(mom20) else None,
            "Jml Hari": int(n),
            "Skor": round((net * 100.0) * win, 4),
        })
    return pd.DataFrame(rows)


def tetapkan_verdict(df, minliq_jt):
    """Verdict 3 tingkat + catatan analisa per saham (alasan eksplisit)."""
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
        hari = r["Jml Hari"]
        liq_ok = (liq is None) or (liq >= minliq_jt)   # None = data tanpa volume
        mom_ok = (mom is None) or (mom > 0)

        gerbang = {
            f"net {net:+.3f}%/malam stlh biaya": net > 0,
            f"win rate {win:.0f}%": win >= 55,
            ("tren 20h " + (f"{mom:+.1f}%" if mom is not None else "n/a")): mom_ok,
        }
        lolos = sum(gerbang.values())
        signifikan = t >= 1.5

        if lolos == 3 and signifikan and liq_ok:
            v = "Layak"
            kekuatan = "kuat" if t >= 2 else "cukup meyakinkan"
            c = (f"LAYAK diriset: untung bersih {net:+.3f}%/malam stabil selama {hari} hr "
                 f"(menang {win:.0f}% hari; t={t:.1f} → edge {kekuatan}, bukan kebetulan); "
                 f"tren 20h {mom:+.1f}%" + 
                 (f"; likuid Rp {liq/1000:.1f} M/hr → mudah keluar di opening." if liq is not None else "."))
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
            c += f" (Catatan: likuiditas Rp {liq/1000:.1f} M/hr di bawah ambang — hati-hati slippage.)"
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
    """Tarik batch dari Yahoo (paralel, harga terkoreksi split/dividen)."""
    import yfinance as yf
    period = f"{max(lookback + 50, 130)}d"
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


def demo_data(tickers, lookback, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=dt.date.today(), periods=lookback + 60)
    data = {}
    for i, t in enumerate(tickers):
        price = 500.0 * (1 + (i % 10) * 0.4)
        drift = rng.normal(0.0015, 0.0022)
        rows = []
        for d in dates:
            prev = price
            op = prev * (1 + rng.normal(drift, 0.010))
            cp = op * (1 + rng.normal(-0.0002, 0.013))
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
    base = df.style
    _map = getattr(base, "map", None) or base.applymap
    sty = _map(warna, subset=["Layak?"]) if "Layak?" in df.columns else base
    num = {"Overnight Avg %":"{:+.3f}","Net stlh Biaya %":"{:+.3f}","Volatilitas %":"{:.3f}",
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
    st.title("📈 Screener Beli Sore – Jual Pagi")
    st.caption("v2 — universe ~140 saham likuid, harga terkoreksi split/dividen, "
               "uji statistik t-stat, dan catatan analisa per saham. "
               "**Alat bantu riset — bukan rekomendasi / sinyal beli-jual.**")

    with st.expander("⚠️ Cara baca & arti 'Layak' (klik)"):
        st.markdown(
            "- **Net stlh Biaya %** — rata-rata untung per malam SETELAH biaya transaksi. "
            "Inilah angka realistisnya.\n"
            "- **t-Stat** — uji statistik: ≥2 edge kuat, 1.5–2 cukup, <1.5 bisa cuma kebetulan.\n"
            "- **✅ Layak** = net>0 **dan** win rate ≥55% **dan** tren 20h naik **dan** t≥1.5 "
            "**dan** cukup likuid. Syarat sengaja ketat — wajar kalau sering hanya 0–5 saham.\n"
            "- **🟡 Hampir** = gagal tipis di satu syarat → kandidat pantauan.\n"
            "- **'Layak: 0' bukan error** — artinya saat ini gap semalam rata-rata memang "
            "lebih kecil dari biaya. Itu informasi berharga: jangan dipaksakan.\n"
            "- Kinerja masa lalu tidak menjamin hasil ke depan. Risiko di tangan Anda."
        )

    sb = st.sidebar
    sb.header("⚙️ Pengaturan")
    sumber = sb.radio("Sumber data", ["Live (Yahoo Finance)", "Upload CSV", "Data contoh (demo)"])
    lookback = sb.slider("Hari bursa dianalisis", 20, 180, 60, 5,
                         help="60 disarankan. Terlalu panjang = pola lama yang sudah basi ikut terhitung.")
    cost = sb.number_input("Biaya bolak-balik (%)", 0.0, 2.0, 0.40, 0.05,
                           help="Fee beli + fee jual broker Anda. Contoh 0.15+0.25 = 0.40")
    minliq = sb.number_input("Likuiditas min (Rp Juta/hari)", 0, 500_000, 5_000, 500,
                             help="5000 = Rp 5 miliar/hari. Jangan terlalu kecil untuk strategi overnight.")
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
        sb.caption(f"{len(tickers)} saham — penarikan batch ±1–2 menit (di-cache 30 menit).")
        if sb.button("🔄 Refresh data (hapus cache)"):
            fetch_live.clear()

    # ---------- data ----------
    if sumber == "Live (Yahoo Finance)":
        with st.spinner(f"Menarik {len(tickers)} saham dari Yahoo (batch)..."):
            data, gagal = fetch_live(tuple(tickers), lookback)
        if not data:
            st.error("Gagal menarik data live. Coba tombol Refresh, atau pakai Upload CSV / Data contoh.")
            return
        if gagal:
            st.caption(f"ℹ️ {len(gagal)} kode dilewati (tidak ada data): "
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
        data = demo_data(DEFAULT_TICKERS[:60], lookback)
        st.warning("Mode **DATA CONTOH (simulasi)** — angka PALSU, hanya untuk lihat tampilan.")

    # ---------- hitung ----------
    met = compute_metrics(data, lookback, cost)
    if met.empty:
        st.error("Data terlalu sedikit untuk dihitung (butuh ±20 hari per saham)."); return
    met = tetapkan_verdict(met, minliq)
    ranked = rank(met, minliq)

    n_layak = int((met["Layak?"] == "Layak").sum())
    n_hampir = int((met["Layak?"] == "Hampir").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saham dihitung", len(met))
    c2.metric("✅ Layak", n_layak)
    c3.metric("🟡 Hampir", n_hampir)
    c4.metric("Tanggal", dt.date.today().isoformat())

    if n_layak == 0:
        st.info("**Tidak ada yang 'Layak' hari ini — itu temuan, bukan error.** Artinya dalam "
                f"{lookback} hari terakhir tidak ada saham yang gap malamnya konsisten menutup "
                f"biaya {cost:.2f}%. Lihat tingkat **🟡 Hampir** di bawah untuk kandidat pantauan, "
                "dan baca Catatan Analisa-nya kenapa mereka belum lolos.")

    if tampil == "Hanya Layak":
        ranked = ranked[ranked["Layak?"] == "Layak"].reset_index(drop=True)
    elif tampil == "Layak + Hampir":
        ranked = ranked[ranked["Layak?"].isin(["Layak", "Hampir"])].reset_index(drop=True)

    show = ranked.head(top_n)
    tampil_cols = (list(show.columns) if semua_kolom
                   else [c for c in RINGKAS_COLS if c in show.columns])
    st.dataframe(style_table(show[tampil_cols]), use_container_width=True, height=520)

    # ---------- catatan analisa ----------
    st.subheader("📝 Catatan Analisa")
    st.caption("Alasan eksplisit per saham — kenapa layak / hampir / tidak. (Top 10 dari tampilan di atas)")
    ikon = {"Layak": "✅", "Hampir": "🟡", "Tidak": "❌"}
    for _, r in show.head(10).iterrows():
        st.markdown(f"{ikon.get(r['Layak?'],'')} **{int(r['Peringkat'])}. {r['Kode']}** — {r['Catatan Analisa']}")

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Unduh CSV (lengkap + catatan)", ranked.to_csv(index=False).encode("utf-8"),
                       f"hasil_screener_{dt.date.today()}.csv", "text/csv", use_container_width=True)
    d2.download_button("⬇️ Unduh Excel", to_excel_bytes(ranked),
                       f"hasil_screener_{dt.date.today()}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

    st.caption("Urutan berdasar **Skor = Net% × Win Rate** (seri: t-Stat). Walau 'Layak', tetap cek "
               "berita & antrian bid-offer sebelum ambil posisi — dan jangan lupa risiko gap turun.")


if __name__ == "__main__":
    main()
