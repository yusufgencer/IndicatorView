# 📊 IndicatorView - Teknik Analiz Tabanlı Strateji Simülatörü

**IndicatorView**, kullanıcıların teknik analiz göstergeleriyle hisse senedi al/sat stratejileri oluşturmasına, test etmesine ve görselleştirmesine olanak tanıyan interaktif bir Streamlit uygulamasıdır.

GitHub Repo: [https://github.com/yusufgencer/IndicatorView](https://github.com/yusufgencer/IndicatorView)

---

## 🚀 Özellikler

- ✅ Parametreli göstergeler: RSI, MACD, Bollinger Bands, MFI
- ✅ Sabit göstergeler: ATR, OBV, SuperTrend, VWAP, ADX, CCI, vb.
- ✅ Gelişmiş stratejiler: Triple Confirmation, Volatility Breakout, Kalman Trend
- ✅ Ortak sinyal stratejisi: Belirli bir eşik oranına göre dinamik sinyal üretimi
- ✅ Al/Sat sinyallerinin mum grafiği üzerinde gösterimi
- ✅ Portföy performanslarını tablo ile karşılaştırma

---

## 🧰 Kullanılan Teknolojiler

- Python 3.10+
- [Streamlit](https://streamlit.io/)
- `pandas`, `plotly`, `yfinance` gibi veri işleme ve görselleştirme kütüphaneleri

---

## ⚙️ Kurulum

```bash
git clone https://github.com/yusufgencer/IndicatorView.git
cd IndicatorView
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt
