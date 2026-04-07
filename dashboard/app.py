"""
Dashboard Streamlit pour monitorer le modèle de scoring crédit en production.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# Configuration de la page
# ============================================================================

st.set_page_config(
    page_title="Credit Scoring - Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Monitoring du modèle de scoring crédit")
st.subheader("Dashboard de suivi en production")

# ============================================================================
# Chemins des fichiers
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_FILE = PROJECT_ROOT / "logs" / "api_logs.jsonl"
DRIFT_REPORT = PROJECT_ROOT / "reports" / "drift_report.csv"
QUALITY_REPORT = PROJECT_ROOT / "reports" / "quality_report.csv"

# ============================================================================
# Fonctions utilitaires
# ============================================================================

@st.cache_data
def load_logs():
    """Charge les logs de l'API depuis le fichier JSONL."""
    logs = []
    if LOGS_FILE.exists():
        try:
            with open(LOGS_FILE, 'r') as f:
                for line in f:
                    try:
                        log = json.loads(line.strip())
                        log['timestamp'] = pd.to_datetime(log['timestamp'])
                        logs.append(log)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            st.error(f"Erreur lors de la lecture des logs: {e}")
    return pd.DataFrame(logs)


@st.cache_data
def load_drift_report():
    """Charge le rapport de drift."""
    if DRIFT_REPORT.exists():
        return pd.read_csv(DRIFT_REPORT)
    return pd.DataFrame()


@st.cache_data
def load_quality_report():
    """Charge le rapport de qualité."""
    if QUALITY_REPORT.exists():
        return pd.read_csv(QUALITY_REPORT)
    return pd.DataFrame()


def filter_logs_by_date(df, min_date, max_date):
    """Filtre les logs par plage temporelle."""
    if df.empty or 'timestamp' not in df.columns:
        return df
    return df[(df['timestamp'].dt.date >= min_date) & (df['timestamp'].dt.date <= max_date)]


# ============================================================================
# Sidebar - Filtres et configuration
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Bouton de rafraîchissement
    if st.button("🔄 Rafraîchir les données", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    
    # Affichage du chemin du fichier
    st.subheader("📁 Chemins des fichiers")
    st.text_input(
        "Fichier de logs:",
        value=str(LOGS_FILE),
        disabled=True,
        key="logs_path"
    )
    
    # Charger les logs pour obtenir les dates
    df_logs = load_logs()
    
    if not df_logs.empty and 'timestamp' in df_logs.columns:
        min_available_date = df_logs['timestamp'].dt.date.min()
        max_available_date = df_logs['timestamp'].dt.date.max()
        
        # Sélecteur de plage temporelle
        st.subheader("📅 Plage temporelle")
        
        col1, col2 = st.columns(2)
        with col1:
            min_date = st.date_input(
                "Date de début:",
                value=min_available_date,
                min_value=min_available_date,
                max_value=max_available_date,
                key="min_date"
            )
        
        with col2:
            max_date = st.date_input(
                "Date de fin:",
                value=max_available_date,
                min_value=min_available_date,
                max_value=max_available_date,
                key="max_date"
            )
        
        # Filtrer les logs
        df_filtered = filter_logs_by_date(df_logs, min_date, max_date)
    else:
        st.warning("Aucune donnée de logs disponible")
        df_filtered = pd.DataFrame()
        min_date = max_date = None
    
    st.divider()
    
    # Statistiques générales en sidebar
    if not df_filtered.empty:
        st.subheader("📈 Statistiques rapides")
        st.metric("Total requêtes", len(df_filtered))
        
        if 'prediction' in df_filtered.columns:
            approval_rate = (df_filtered['prediction'].sum() / len(df_filtered)) * 100
            st.metric("Taux d'approbation", f"{approval_rate:.2f}%")
        
        if 'inference_time_ms' in df_filtered.columns:
            avg_inference = df_filtered['inference_time_ms'].mean()
            st.metric("Temps inf. moyen", f"{avg_inference:.2f} ms")

# ============================================================================
# Contenu principal - Onglets
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vue d'ensemble",
    "📉 Distributions des scores",
    "🔍 Détection de drift",
    "✅ Qualité",
    "⏱️ Performance"
])

# ============================================================================
# Onglet 1: Vue d'ensemble
# ============================================================================

with tab1:
    if not df_filtered.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📥 Requêtes traitées",
                len(df_filtered),
                delta=None
            )
        
        with col2:
            if 'prediction' in df_filtered.columns:
                approval_count = df_filtered['prediction'].sum()
                st.metric(
                    "✅ Approbations",
                    int(approval_count),
                    delta=f"{(approval_count/len(df_filtered)*100):.1f}%"
                )
        
        with col3:
            if 'prediction' in df_filtered.columns:
                rejection_count = len(df_filtered) - df_filtered['prediction'].sum()
                st.metric(
                    "❌ Refus",
                    int(rejection_count),
                    delta=f"{(rejection_count/len(df_filtered)*100):.1f}%"
                )
        
        with col4:
            if 'inference_time_ms' in df_filtered.columns:
                avg_time = df_filtered['inference_time_ms'].mean()
                st.metric(
                    "⏱️ Temps moyen",
                    f"{avg_time:.2f} ms"
                )
        
        st.divider()
        
        # Graphique: Prédictions dans le temps
        if 'timestamp' in df_filtered.columns and 'prediction' in df_filtered.columns:
            st.subheader("Évolution des prédictions dans le temps")
            
            # Agréger par heure
            df_time = df_filtered.set_index('timestamp').resample('1h').agg({
                'prediction': ['sum', 'count']
            }).reset_index()
            df_time.columns = ['timestamp', 'approvals', 'total']
            df_time['rejections'] = df_time['total'] - df_time['approvals']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_time['timestamp'],
                y=df_time['approvals'],
                name='Approbations',
                mode='lines+markers',
                line=dict(color='green'),
                fill='tozeroy'
            ))
            fig.add_trace(go.Scatter(
                x=df_time['timestamp'],
                y=df_time['rejections'],
                name='Refus',
                mode='lines+markers',
                line=dict(color='red'),
                fill='tozeroy'
            ))
            fig.update_layout(
                title="Prédictions par heure (Approbations vs Refus)",
                xaxis_title="Heure",
                yaxis_title="Nombre de requêtes",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau des dernières requêtes
        st.subheader("Dernières requêtes traitées")
        display_cols = ['timestamp', 'request_id', 'score', 'prediction', 'inference_time_ms']
        available_cols = [col for col in display_cols if col in df_filtered.columns]
        st.dataframe(
            df_filtered[available_cols].tail(10).sort_values('timestamp', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Aucune donnée disponible pour la période sélectionnée")

# ============================================================================
# Onglet 2: Distributions des scores
# ============================================================================

with tab2:
    if not df_filtered.empty and 'score' in df_filtered.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution des scores de probabilité")
            fig = px.histogram(
                df_filtered,
                x='score',
                nbins=30,
                labels={'score': 'Score', 'count': 'Nombre de requêtes'},
                color_discrete_sequence=['#636EFA']
            )
            fig.add_vline(
                x=0.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Seuil (0.5)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Statistiques des scores")
            st.write(f"**Minimum:** {df_filtered['score'].min():.4f}")
            st.write(f"**Maximum:** {df_filtered['score'].max():.4f}")
            st.write(f"**Moyenne:** {df_filtered['score'].mean():.4f}")
            st.write(f"**Médiane:** {df_filtered['score'].median():.4f}")
            st.write(f"**Écart-type:** {df_filtered['score'].std():.4f}")
            
            # Distribution par prédiction
            if 'prediction' in df_filtered.columns:
                st.subheader("Scores par prédiction")
                stats_by_pred = df_filtered.groupby('prediction')['score'].describe()
                st.dataframe(stats_by_pred, use_container_width=True)
        
        st.divider()
        
        # Box plot
        st.subheader("Distribution des scores (Box plot)")
        if 'prediction' in df_filtered.columns:
            fig = px.box(
                df_filtered,
                x='prediction',
                y='score',
                labels={'prediction': 'Prédiction', 'score': 'Score'},
                points='outliers',
                color_discrete_sequence=['#00CC96']
            )
            fig.update_xaxes(ticktext=['Refus', 'Approbation'], tickvals=[0, 1])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Données de scores non disponibles")

# ============================================================================
# Onglet 3: Détection de drift
# ============================================================================

with tab3:
    df_drift = load_drift_report()
    
    if not df_drift.empty:
        st.subheader("Rapport de drift (Détection de la dérive des données)")
        
        # Afficher les images si elles existent
        col1, col2 = st.columns(2)
        
        with col1:
            drift_img = PROJECT_ROOT / "reports" / "amt_credit_drift_verification.png"
            if drift_img.exists():
                st.image(str(drift_img), caption="Drift - AMT_CREDIT")
        
        with col2:
            dist_img = PROJECT_ROOT / "reports" / "feature_distributions.png"
            if dist_img.exists():
                st.image(str(dist_img), caption="Distributions des features")
        
        st.divider()
        
        # Tableau du rapport de drift
        st.subheader("Détails du rapport")
        st.dataframe(df_drift, use_container_width=True, hide_index=True)
        
        # Graphique: Features avec drift détecté
        if 'feature' in df_drift.columns and any(['drift' in col.lower() for col in df_drift.columns]):
            drift_col = [col for col in df_drift.columns if 'drift' in col.lower()][0]
            fig = px.bar(
                df_drift,
                x='feature',
                y=drift_col,
                labels={drift_col: 'Score de drift'},
                color_discrete_sequence=['#EF553B']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Rapport de drift non disponible")
    
    # Détection de drift à partir des logs
    if not df_filtered.empty and 'score' in df_filtered.columns:
        st.divider()
        st.subheader("Analyse du drift des scores en production")
        
        # Comparer score moyen au fil du temps
        if 'timestamp' in df_filtered.columns:
            df_daily = df_filtered.set_index('timestamp').resample('1d')['score'].agg(['mean', 'std', 'count']).reset_index()
            df_daily = df_daily[df_daily['count'] > 0]
            
            if len(df_daily) > 1:
                fig = px.line(
                    df_daily,
                    x='timestamp',
                    y='mean',
                    error_y='std',
                    labels={'timestamp': 'Date', 'mean': 'Score moyen'},
                    markers=True
                )
                fig.update_layout(height=400, title="Évolution du score moyen (avec écart-type)")
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# Onglet 4: Qualité
# ============================================================================

with tab4:
    df_quality = load_quality_report()
    
    if not df_quality.empty:
        st.subheader("Rapport de qualité du modèle")
        st.dataframe(df_quality, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Image du rapport opérationnel
        operational_img = PROJECT_ROOT / "reports" / "operational_metrics.png"
        if operational_img.exists():
            st.image(str(operational_img), caption="Métriques opérationnelles")
        
        flag_img = PROJECT_ROOT / "reports" / "flag_document_small_sample_analysis.png"
        if flag_img.exists():
            st.image(str(flag_img), caption="Analyse des petits échantillons")
    else:
        st.info("Rapport de qualité non disponible")

# ============================================================================
# Onglet 5: Performance
# ============================================================================

with tab5:
    if not df_filtered.empty and 'inference_time_ms' in df_filtered.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution des temps d'inférence")
            fig = px.histogram(
                df_filtered,
                x='inference_time_ms',
                nbins=30,
                labels={'inference_time_ms': 'Temps (ms)', 'count': 'Nombre'},
                color_discrete_sequence=['#AB63FA']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Statistiques de performance")
            st.write(f"**Min:** {df_filtered['inference_time_ms'].min():.4f} ms")
            st.write(f"**Max:** {df_filtered['inference_time_ms'].max():.4f} ms")
            st.write(f"**Moyenne:** {df_filtered['inference_time_ms'].mean():.4f} ms")
            st.write(f"**Médiane:** {df_filtered['inference_time_ms'].median():.4f} ms")
            st.write(f"**P95:** {df_filtered['inference_time_ms'].quantile(0.95):.4f} ms")
            st.write(f"**P99:** {df_filtered['inference_time_ms'].quantile(0.99):.4f} ms")
        
        st.divider()
        
        # Performance dans le temps
        if 'timestamp' in df_filtered.columns:
            st.subheader("Évolution du temps d'inférence (par heure)")
            df_perf = df_filtered.set_index('timestamp').resample('1h')['inference_time_ms'].agg(['mean', 'max', 'min', 'count']).reset_index()
            df_perf = df_perf[df_perf['count'] > 0]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_perf['timestamp'],
                y=df_perf['mean'],
                name='Moyenne',
                mode='lines+markers',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=df_perf['timestamp'],
                y=df_perf['max'],
                name='Maximum',
                mode='lines',
                line=dict(color='red', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=df_perf['timestamp'],
                y=df_perf['min'],
                name='Minimum',
                mode='lines',
                line=dict(color='green', dash='dash')
            ))
            fig.update_layout(
                title="Temps d'inférence par heure",
                xaxis_title="Heure",
                yaxis_title="Temps (ms)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Données de performance non disponibles")

# ============================================================================
# Footer
# ============================================================================

st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    if not df_filtered.empty and 'timestamp' in df_filtered.columns:
        earliest = df_filtered['timestamp'].min()
        st.caption(f"🕐 Première requête: {earliest.strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    if not df_filtered.empty and 'timestamp' in df_filtered.columns:
        latest = df_filtered['timestamp'].max()
        st.caption(f"🕐 Dernière requête: {latest.strftime('%Y-%m-%d %H:%M:%S')}")

with col3:
    st.caption("💡 Dashboard mis à jour automatiquement avec les caches Streamlit")
