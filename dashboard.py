# dashboard.py
# Day 6: Streamlit Dashboard for Store Owner
# Purpose: Visual interface to monitor detections, thefts, and manage products
# Impact: Store owner can review history, mark items as sold, see patterns

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from database import (
    init_db, get_recent_thefts, get_theft_summary,
    mark_as_sold, Session, DetectionEvent, TheftAlert, SessionLog
)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Vision Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e2e;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #ff4444;
    }
    .stolen-badge {
        background-color: #ff4444;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
    }
    .sold-badge {
        background-color: #44ff44;
        color: black;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
    }
    .available-badge {
        background-color: #4444ff;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Initialize Database ───────────────────────────────────────────────────────
init_db()

# ── Helper Functions ──────────────────────────────────────────────────────────
def get_all_detections():
    """Get all detection events from database"""
    session = Session()
    try:
        events = session.query(DetectionEvent)\
                       .order_by(DetectionEvent.timestamp.desc())\
                       .all()
        return [{
            'id': e.id,
            'timestamp': e.timestamp,
            'object_name': e.object_name,
            'tracker_id': e.tracker_id,
            'zone': e.zone,
            'confidence': e.confidence,
            'status': e.status
        } for e in events]
    except:
        return []
    finally:
        session.close()

def get_all_thefts():
    """Get all theft alerts from database"""
    session = Session()
    try:
        thefts = session.query(TheftAlert)\
                       .order_by(TheftAlert.timestamp.desc())\
                       .all()
        return [{
            'id': t.id,
            'timestamp': t.timestamp,
            'object_name': t.object_name,
            'tracker_id': t.tracker_id,
            'frame': t.frame
        } for t in thefts]
    except:
        return []
    finally:
        session.close()

def get_all_sessions():
    """Get all session logs"""
    session = Session()
    try:
        logs = session.query(SessionLog)\
                     .order_by(SessionLog.start_time.desc())\
                     .all()
        return [{
            'id': l.id,
            'start_time': l.start_time,
            'end_time': l.end_time,
            'total_frames': l.total_frames,
            'total_detections': l.total_detections,
            'total_thefts': l.total_thefts
        } for l in logs]
    except:
        return []
    finally:
        session.close()

def get_products_by_status(status):
    """Get products filtered by status"""
    session = Session()
    try:
        events = session.query(DetectionEvent)\
                       .filter_by(status=status)\
                       .order_by(DetectionEvent.timestamp.desc())\
                       .all()
        # Get unique tracker IDs only
        seen = set()
        unique = []
        for e in events:
            if e.tracker_id not in seen:
                seen.add(e.tracker_id)
                unique.append({
                    'tracker_id': e.tracker_id,
                    'object_name': e.object_name,
                    'status': e.status,
                    'timestamp': e.timestamp
                })
        return unique
    except:
        return []
    finally:
        session.close()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
    st.title("Retail Vision")
    st.markdown("---")
    
    # Auto refresh
    auto_refresh = st.checkbox("Auto Refresh (every 5s)", value=False)
    if auto_refresh:
        st.rerun()

    st.markdown("---")
    st.markdown("### Quick Stats")
    
    summary = get_theft_summary()
    st.metric("Total Thefts", summary.get('total_thefts', 0))
    st.metric("Total Sessions", summary.get('total_sessions', 0))
    
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Go to:", [
        "📊 Overview",
        "🚨 Theft Alerts",
        "📦 Product Status",
        "📈 Analytics",
        "📋 Session Logs"
    ])

# ── Main Content ──────────────────────────────────────────────────────────────
st.title("🏪 Retail Vision Intelligence Dashboard")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.header("📊 Overview")

    # ── Top metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    all_detections = get_all_detections()
    all_thefts     = get_all_thefts()
    all_sessions   = get_all_sessions()
    available      = get_products_by_status('available')
    stolen         = get_products_by_status('stolen')
    sold           = get_products_by_status('sold')

    with col1:
        st.metric("🚨 Total Thefts", len(all_thefts))
    with col2:
        st.metric("📦 Available Items", len(available))
    with col3:
        st.metric("✅ Items Sold", len(sold))
    with col4:
        st.metric("🔴 Items Stolen", len(stolen))

    st.markdown("---")

    # ── Recent thefts ─────────────────────────────────────────────────────
    st.subheader("🚨 Recent Theft Alerts")
    if all_thefts:
        df_thefts = pd.DataFrame(all_thefts[:10])
        df_thefts['timestamp'] = pd.to_datetime(df_thefts['timestamp'])
        df_thefts['timestamp'] = df_thefts['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_thefts.columns = ['ID', 'Time', 'Object', 'Tracker ID', 'Frame']
        st.dataframe(df_thefts, use_container_width=True)
    else:
        st.success("✅ No theft alerts recorded yet!")

    st.markdown("---")

    # ── Recent detections ─────────────────────────────────────────────────
    st.subheader("📹 Recent Detections")
    if all_detections:
        df_det = pd.DataFrame(all_detections[:20])
        df_det['timestamp'] = pd.to_datetime(df_det['timestamp'])
        df_det['timestamp'] = df_det['timestamp'].dt.strftime('%H:%M:%S')
        st.dataframe(df_det[['timestamp', 'object_name', 'tracker_id', 'zone', 'confidence', 'status']],
                    use_container_width=True)
    else:
        st.info("No detections recorded yet. Run fence.py to start detecting!")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — THEFT ALERTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Theft Alerts":
    st.header("🚨 Theft Alerts")

    all_thefts = get_all_thefts()

    if not all_thefts:
        st.success("✅ No theft alerts recorded!")
    else:
        st.warning(f"⚠️ {len(all_thefts)} theft alert(s) recorded!")

        for theft in all_thefts:
            with st.expander(f"🚨 {theft['object_name']} (ID#{theft['tracker_id']}) — {theft['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Object:** {theft['object_name']}")
                with col2:
                    st.write(f"**Tracker ID:** #{theft['tracker_id']}")
                with col3:
                    st.write(f"**Frame:** {theft['frame']}")
                st.write(f"**Time:** {theft['timestamp']}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRODUCT STATUS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Product Status":
    st.header("📦 Product Status")
    st.markdown("Mark items as sold so they don't trigger theft alerts at exit.")

    # ── Available products ────────────────────────────────────────────────
    st.subheader("🔵 Available Products")
    available = get_products_by_status('available')

    if not available:
        st.info("No available products tracked yet.")
    else:
        for product in available:
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1:
                st.write(f"**{product['object_name']}**")
            with col2:
                st.write(f"Tracker ID: #{product['tracker_id']}")
            with col3:
                st.write(f"Last seen: {product['timestamp'].strftime('%H:%M:%S')}")
            with col4:
                if st.button(f"✅ Sold", key=f"sold_{product['tracker_id']}"):
                    mark_as_sold(product['tracker_id'])
                    st.success(f"✅ {product['object_name']} ID#{product['tracker_id']} marked as sold!")
                    st.rerun()

    st.markdown("---")

    # ── Stolen products ───────────────────────────────────────────────────
    st.subheader("🔴 Stolen Products")
    stolen = get_products_by_status('stolen')

    if not stolen:
        st.success("✅ No stolen products!")
    else:
        for product in stolen:
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                st.error(f"**{product['object_name']}**")
            with col2:
                st.write(f"Tracker ID: #{product['tracker_id']}")
            with col3:
                st.write(f"Time: {product['timestamp'].strftime('%H:%M:%S')}")

    st.markdown("---")

    # ── Sold products ─────────────────────────────────────────────────────
    st.subheader("✅ Sold Products")
    sold = get_products_by_status('sold')

    if not sold:
        st.info("No sold products yet.")
    else:
        for product in sold:
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                st.success(f"**{product['object_name']}**")
            with col2:
                st.write(f"Tracker ID: #{product['tracker_id']}")
            with col3:
                st.write(f"Time: {product['timestamp'].strftime('%H:%M:%S')}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    st.header("📈 Analytics")

    all_detections = get_all_detections()
    all_thefts     = get_all_thefts()

    if not all_detections:
        st.info("No data yet. Run fence.py to start collecting data!")
    else:
        # ── Status breakdown pie chart ────────────────────────────────────
        st.subheader("Product Status Breakdown")
        df_det = pd.DataFrame(all_detections)
        status_counts = df_det['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']

        fig_pie = px.pie(
            status_counts,
            values='Count',
            names='Status',
            color='Status',
            color_discrete_map={
                'available': '#4444ff',
                'stolen':    '#ff4444',
                'sold':      '#44ff44'
            },
            title="Product Status Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # ── Most detected objects bar chart ──────────────────────────────
        st.subheader("Most Detected Objects")
        obj_counts = df_det['object_name'].value_counts().reset_index()
        obj_counts.columns = ['Object', 'Count']

        fig_bar = px.bar(
            obj_counts,
            x='Object',
            y='Count',
            color='Count',
            title="Detection Frequency by Object Type"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    if all_thefts:
        # ── Theft timeline ────────────────────────────────────────────────
        st.subheader("Theft Timeline")
        df_thefts = pd.DataFrame(all_thefts)
        df_thefts['timestamp'] = pd.to_datetime(df_thefts['timestamp'])
        df_thefts['hour'] = df_thefts['timestamp'].dt.hour

        hourly = df_thefts.groupby('hour').size().reset_index(name='thefts')

        fig_line = px.line(
            hourly,
            x='hour',
            y='thefts',
            title="Thefts by Hour of Day",
            markers=True
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # ── Most stolen items ─────────────────────────────────────────────
        st.subheader("Most Stolen Items")
        stolen_counts = df_thefts['object_name'].value_counts().reset_index()
        stolen_counts.columns = ['Object', 'Times Stolen']
        st.dataframe(stolen_counts, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SESSION LOGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Session Logs":
    st.header("📋 Session Logs")

    all_sessions = get_all_sessions()

    if not all_sessions:
        st.info("No sessions recorded yet.")
    else:
        df_sessions = pd.DataFrame(all_sessions)
        df_sessions['start_time'] = pd.to_datetime(df_sessions['start_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_sessions['end_time']   = pd.to_datetime(df_sessions['end_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_sessions.columns = ['ID', 'Start Time', 'End Time', 'Frames', 'Detections', 'Thefts']
        st.dataframe(df_sessions, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("*Retail Vision Intelligence System — Built with YOLOv8 + ByteTrack + PostgreSQL*")