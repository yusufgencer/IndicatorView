import plotly.graph_objects as go
import pandas as pd

def plot_price_with_signals(df: pd.DataFrame, title: str = ""):
    fig = go.Figure()

    # Fiyat çizgisi (primary y-axis)
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        name='Fiyat',
        yaxis='y1',
        line=dict(color='blue')
    ))

    # Portföy çizgisi (secondary y-axis)
    if 'Portfolio' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Portfolio'],
            name='Portföy',
            yaxis='y2',
            line=dict(color='orange', dash='dash')
        ))

    # Al/Sat sinyalleri
    if 'Signal' in df.columns:
        buy_signals = df[df['Signal'] == 'BUY']
        sell_signals = df[df['Signal'] == 'SELL']

        fig.add_trace(go.Scatter(
            x=buy_signals['Date'],
            y=buy_signals['Close'],
            mode='markers',
            name='AL',
            yaxis='y1',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))

        fig.add_trace(go.Scatter(
            x=sell_signals['Date'],
            y=sell_signals['Close'],
            mode='markers',
            name='SAT',
            yaxis='y1',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))

    # Layout ayarları
    fig.update_layout(
        title=title,
        xaxis=dict(title='Tarih'),
        yaxis=dict(title='Fiyat ($)', side='left'),
        yaxis2=dict(title='Portföy ($)', overlaying='y', side='right'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )

    return fig



# Mum (candlestick) grafiği (plotly)
def plot_candlestick(df: pd.DataFrame, title: str = "Hisse Fiyatı - Mum Grafiği"):
    fig = go.Figure(data=[
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Fiyat'
        )
    ])

    fig.update_layout(title=title, xaxis_title='Tarih', yaxis_title='Fiyat ($)',
                      xaxis_rangeslider_visible=False,
                      hovermode='x unified')

    return fig