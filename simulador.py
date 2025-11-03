
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Utilitários financeiros
# =========================
def future_value_schedule(P0, contrib_m, rate_am, months, fee_am=0.0, shock_vol_am=None, seed=42):
    rng = np.random.default_rng(seed) if shock_vol_am is not None else None
    saldo = P0
    rows = []
    for m in range(1, months+1):
        r = rate_am
        if shock_vol_am is not None:
            r = (1 + r) * np.exp(rng.normal(0.0, shock_vol_am)) - 1
        r_eff = r - fee_am
        saldo = saldo * (1 + r_eff) + contrib_m
        rows.append((m, saldo))
    return pd.DataFrame(rows, columns=["Mês", "Saldo"])

def gbm_paths(mu_am, cov_am, weights, S0=1.0, months=120, n_paths=2000, seed=123):
    rng = np.random.default_rng(seed)
    mu_p = np.dot(weights, mu_am)
    var_p = weights @ cov_am @ weights
    sigma_p = np.sqrt(var_p)
    S = np.zeros((months+1, n_paths))
    S[0, :] = S0
    for t in range(1, months+1):
        z = rng.standard_normal(n_paths)
        growth = np.exp((mu_p - 0.5*sigma_p**2) + sigma_p * z)
        S[t, :] = S[t-1, :] * growth
    return S

def portfolio_stats(paths, risk_free=0.06):
    rets = paths[1:, :] / paths[:-1, :] - 1.0
    mean_m = rets.mean()
    std_m  = rets.std()
    er_aa  = (1 + mean_m) ** 12 - 1
    vol_aa = std_m * np.sqrt(12)
    sharpe = (er_aa - risk_free) / (vol_aa + 1e-9)
    return {"retorno_esperado_aa": float(er_aa),
            "volatilidade_aa": float(vol_aa),
            "sharpe_aproximado": float(sharpe)}

def run_guided_vs_chaotic_scenarios(P0, aporte_m, anos, ret_aa, vol_aa, fee_aa, seed=777):
    months = int(anos * 12)
    rate_am = (1 + ret_aa) ** (1/12) - 1
    vol_am  = vol_aa / np.sqrt(12)
    fee_am_caos   = (1 + 0.02) ** (1/12) - 1
    fee_am_guiado = (1 + fee_aa) ** (1/12) - 1

    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, vol_am, size=months)
    market = np.exp((rate_am - 0.5*vol_am**2) + shocks) - 1

    # Caótico (aportes erráticos + saques)
    saldo_c, rows_c = P0, []
    for m in range(1, months+1):
        timing_bias = -0.5 * market[m-1]
        mult = np.clip(rng.uniform(0.0, 1.5) + timing_bias, 0.0, 1.5)
        aporte_var = aporte_m * mult
        saque = min(saldo_c * rng.uniform(0.01, 0.05), saldo_c) if rng.random() < 0.08 else 0.0
        r_eff = market[m-1] - fee_am_caos
        saldo_c = (saldo_c - saque) * (1 + r_eff) + aporte_var
        rows_c.append((m, saldo_c))
    df_caos = pd.DataFrame(rows_c, columns=["Mês", "Saldo"])

    # Guiado (DCA disciplinado + taxa menor)
    saldo_g, rows_g = P0, []
    for m in range(1, months+1):
        r_eff = market[m-1] - fee_am_guiado
        saldo_g = saldo_g * (1 + r_eff) + aporte_m
        rows_g.append((m, saldo_g))
    df_guiado = pd.DataFrame(rows_g, columns=["Mês", "Saldo"])
    return df_caos, df_guiado

# =========================
# Interface Streamlit
# =========================
st.set_page_config(page_title="Simulador de Investimentos — Ciclo 4", layout="wide")
st.title("📈 Simulador de Investimentos — Educação Financeira (Ciclo 4)")
st.markdown("""
Simulador educacional para demonstrar:
- **Juros compostos** (valor do dinheiro no tempo)
- **Carteira & risco** (volatilidade, correlação; simulação GBM)
- **Caótico vs. Guiado** (disciplina e impacto de taxas)
""")

st.sidebar.header("Configurações Gerais")
currency  = st.sidebar.selectbox("Moeda", ["R$", "US$"], index=0)
risk_free = st.sidebar.number_input("Taxa livre de risco (a.a., %)", value=6.0, step=0.1)

tab1, tab2, tab3 = st.tabs(["🔢 Juros Compostos", "📊 Carteira & Risco", "🧭 Guiado vs. Caótico"])

with tab1:
    st.subheader("🔢 Simulador de Juros Compostos")
    col_a, col_b = st.columns(2)
    with col_a:
        P0 = st.number_input("Capital inicial", min_value=0.0, value=1000.0, step=100.0)
        contrib_m = st.number_input("Aporte mensal", min_value=0.0, value=200.0, step=50.0)
        years = st.number_input("Tempo (anos)", min_value=0, value=10, step=1)
    with col_b:
        rate_aa = st.number_input("Taxa esperada (a.a., %)", value=10.0, step=0.5)
        vol_aa = st.number_input("Volatilidade anual (opcional, %)", value=0.0, step=0.5)
        fee_aa = st.number_input("Taxa de administração (a.a., %, opcional)", value=0.0, step=0.1)

    months  = int(years * 12)
    rate_am = (1 + rate_aa/100.0) ** (1/12) - 1
    fee_am  = (1 + fee_aa/100.0) ** (1/12) - 1
    vol_am  = (vol_aa/100.0) / np.sqrt(12) if vol_aa > 0 else None

    df = future_value_schedule(P0, contrib_m, rate_am, months, fee_am, vol_am)
    st.dataframe(df.head(12))

    fig1 = plt.figure()
    plt.plot(df["Mês"], df["Saldo"])
    plt.xlabel("Mês"); plt.ylabel(f"Saldo ({currency})"); plt.title("Evolução do capital (juros compostos)")
    st.pyplot(fig1)

    total_aportado = (contrib_m * months) + P0
    st.metric("Total aportado", f"{currency} {total_aportado:,.2f}")
    st.metric("Saldo final",     f"{currency} {df['Saldo'].iloc[-1]:,.2f}")

with tab2:
    st.subheader("📊 Carteira, Risco & CAPM (básico)")
    st.markdown("Defina até 3 ativos e simule a **carteira** via GBM.")
    n_assets = st.number_input("Número de ativos", min_value=1, max_value=3, value=3, step=1)
    names, exp_ret_aa, vol_aa_list, weights = [], [], [], []
    for i in range(int(n_assets)):
        st.markdown(f"**Ativo {i+1}**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: names.append(st.text_input(f"Nome {i+1}", value=f"Ativo {i+1}"))
        with c2: exp_ret_aa.append(st.number_input(f"Retorno esperado (a.a., %) {i+1}", value=10.0, step=0.5, key=f"ret{i}"))
        with c3: vol_aa_list.append(st.number_input(f"Volatilidade (a.a., %) {i+1}", value=20.0, step=0.5, key=f"vol{i}"))
        with c4: weights.append(st.slider(f"Peso {i+1}", 0.0, 1.0, 1.0/float(n_assets), 0.01, key=f"w{i}"))

    w = np.array(weights, dtype=float)
    w = w / (w.sum() if w.sum() != 0 else 1.0)

    corr    = st.slider("Correlação média entre os ativos", -0.9, 0.95, 0.2, 0.05)
    years_p = st.number_input("Horizonte (anos)", min_value=1, value=10, step=1)
    n_paths = st.number_input("Caminhos Monte Carlo", min_value=100, max_value=10000, value=2000, step=100)

    mu_aa      = np.array(exp_ret_aa)/100.0
    vol_aa_arr = np.array(vol_aa_list)/100.0
    mu_am      = (1 + mu_aa) ** (1/12) - 1
    vol_am_arr = vol_aa_arr / np.sqrt(12)

    # Matriz de covariância mensal a partir de uma correlação média
    if len(vol_am_arr) == 1:
        cov = np.array([[vol_am_arr[0]**2]])
    else:
        V = np.diag(vol_am_arr)
        R = np.ones((len(vol_am_arr), len(vol_am_arr))) * corr
        np.fill_diagonal(R, 1.0)
        cov = V @ R @ V

    horizon_m = int(years_p*12)
    paths = gbm_paths(mu_am, cov, w, S0=1.0, months=horizon_m, n_paths=int(n_paths))

    stats = portfolio_stats(paths, risk_free=risk_free/100.0)
    st.write("**Estatísticas da carteira (simulada):**")
    st.json(stats)

    fig2 = plt.figure()
    for j in range(min(50, paths.shape[1])):
        plt.plot(paths[:, j])
    plt.xlabel("Mês"); plt.ylabel("Valor da carteira (inicial=1.0)"); plt.title("Caminhos simulados (GBM)")
    st.pyplot(fig2)

    fig3 = plt.figure()
    plt.hist(paths[-1, :], bins=30)
    plt.xlabel("Valor final"); plt.ylabel("Frequência"); plt.title("Distribuição dos valores finais")
    st.pyplot(fig3)

with tab3:
    st.subheader("🧭 Cenário: Caótico (antes) vs. Guiado (depois)")
    c1, c2 = st.columns(2)
    with c1:
        P0_c    = st.number_input("Capital inicial (ambos)", min_value=0.0, value=2000.0, step=100.0)
        aporte  = st.number_input("Aporte mensal (guiado)",  min_value=0.0, value=300.0,  step=50.0)
        anos    = st.number_input("Tempo (anos, ambos)",     min_value=1,   value=8,     step=1)
    with c2:
        ret_aa  = st.number_input("Retorno base (a.a., %)",  value=9.0,  step=0.5)
        vol_aa  = st.number_input("Volatilidade (a.a., %)",  value=18.0, step=0.5)
        fee_aa  = st.number_input("Taxa de administração (a.a., %, guiado)", value=0.3, step=0.1)

    df_caos, df_guiado = run_guided_vs_chaotic_scenarios(
        P0=P0_c, aporte_m=aporte, anos=anos,
        ret_aa=ret_aa/100.0, vol_aa=vol_aa/100.0, fee_aa=fee_aa/100.0
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cenário Caótico (antes de aprender):**")
        fig4 = plt.figure(); plt.plot(df_caos["Mês"], df_caos["Saldo"]); plt.title("Evolução — Caótico"); st.pyplot(fig4)
        st.dataframe(df_caos.tail(5))
    with col2:
        st.markdown("**Cenário Guiado (depois de aprender):**")
        fig5 = plt.figure(); plt.plot(df_guiado["Mês"], df_guiado["Saldo"]); plt.title("Evolução — Guiado"); st.pyplot(fig5)
        st.dataframe(df_guiado.tail(5))

    delta = df_guiado["Saldo"].iloc[-1] - df_caos["Saldo"].iloc[-1]
    st.metric("Diferença (Guiado − Caótico)", f"{currency} {delta:,.2f}")

st.caption("Protótipo educacional — não é recomendação de investimento. Ciclo 4.")
