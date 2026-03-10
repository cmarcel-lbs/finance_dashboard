# Financial Markets Dashboard — HW3

An interactive Dash dashboard covering equity prices, risk-return analysis, and the US Treasury yield curve.

## Features

| Section | What you can explore |
|---|---|
| **Normalised Price Performance** | Compare multiple stocks rebased to 100 over any time window |
| **Risk–Return Scatter** | Annualised return vs volatility; bubble size = Sharpe ratio |
| **Daily Returns Distribution** | Violin plots showing return distribution per stock |
| **Correlation Matrix** | Heatmap of pairwise return correlations |
| **Monthly Returns Heatmap** | Calendar-style grid of monthly % returns per stock |
| **Yield Curve Snapshot** | Compare the current curve to 1, 2, or 5 years ago |
| **10Y–2Y Spread** | Inversion indicator — red bars signal potential recession |

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open http://localhost:8050
```

## Deploying to Render.com

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial dashboard"
   git remote add origin https://github.com/YOUR_USERNAME/finance-dashboard.git
   git push -u origin main
   ```

2. **Create a new Web Service on Render**
   - Go to [render.com](https://render.com) → New → Web Service
   - Connect your GitHub repo
   - Set the following:
     - **Runtime**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - Click **Deploy**

3. Your dashboard will be live at `https://your-app-name.onrender.com` in ~2 minutes.

## Data Sources

- **Stock prices**: [Yahoo Finance](https://finance.yahoo.com) via `yfinance`
- **Treasury yields**: [FRED (Federal Reserve)](https://fred.stlouisfed.org) via `pandas-datareader`

## Files

```
finance_dashboard/
├── app.py            ← main dashboard application
├── requirements.txt  ← Python dependencies
├── Procfile          ← Gunicorn start command for Render
└── README.md         ← this file
```
# finance_dashboard
# finance_dashboard
# finance_dashboard
