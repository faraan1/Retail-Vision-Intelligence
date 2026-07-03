# api.py
# Day 10: REST API for Retail Vision System
# Purpose: Allow any device to access theft alerts and product data remotely
# Impact: Makes system accessible from mobile, web, or any external application

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import uvicorn
from database import (
    init_db, get_recent_thefts, get_theft_summary,
    mark_as_sold, check_product_status,
    Session, DetectionEvent, TheftAlert, SessionLog
)

# ── Initialize FastAPI ────────────────────────────────────────────────────────
app = FastAPI(
    title="Retail Vision Intelligence API",
    description="REST API for real-time retail theft detection system",
    version="1.0.0"
)

# ── CORS Middleware (allows any device to connect) ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize Database ───────────────────────────────────────────────────────
init_db()

# ── Request Models ────────────────────────────────────────────────────────────
class SellRequest(BaseModel):
    tracker_id: int

# ── Helper Functions ──────────────────────────────────────────────────────────
def get_products_by_status_api(status):
    session = Session()
    try:
        events = session.query(DetectionEvent)\
                       .filter_by(status=status)\
                       .order_by(DetectionEvent.timestamp.desc())\
                       .all()
        seen = set()
        unique = []
        for e in events:
            if e.tracker_id not in seen:
                seen.add(e.tracker_id)
                unique.append({
                    'id': e.id,
                    'tracker_id': e.tracker_id,
                    'object_name': e.object_name,
                    'status': e.status,
                    'zone': e.zone,
                    'timestamp': str(e.timestamp)
                })
        return unique
    except Exception as ex:
        return []
    finally:
        session.close()

# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "system": "Retail Vision Intelligence",
        "status": "online",
        "version": "1.0.0",
        "timestamp": str(datetime.now())
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": "connected",
        "timestamp": str(datetime.now())
    }

# ── Statistics ────────────────────────────────────────────────────────────────
@app.get("/stats")
def get_stats():
    summary = get_theft_summary()
    available = get_products_by_status_api('available')
    stolen    = get_products_by_status_api('stolen')
    sold      = get_products_by_status_api('sold')

    return {
        "total_thefts":     summary.get('total_thefts', 0),
        "total_sessions":   summary.get('total_sessions', 0),
        "available_items":  len(available),
        "stolen_items":     len(stolen),
        "sold_items":       len(sold),
        "timestamp":        str(datetime.now())
    }

# ── Theft Alerts ──────────────────────────────────────────────────────────────
@app.get("/thefts")
def get_all_thefts():
    session = Session()
    try:
        thefts = session.query(TheftAlert)\
                       .order_by(TheftAlert.timestamp.desc())\
                       .all()
        return {
            "count": len(thefts),
            "thefts": [{
                "id":          t.id,
                "object_name": t.object_name,
                "tracker_id":  t.tracker_id,
                "frame":       t.frame,
                "timestamp":   str(t.timestamp),
                "resolved":    t.resolved
            } for t in thefts]
        }
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
    finally:
        session.close()

@app.get("/thefts/recent")
def get_recent(limit: int = 10):
    thefts = get_recent_thefts(limit)
    return {
        "count": len(thefts),
        "thefts": [{
            "id":          t.id,
            "object_name": t.object_name,
            "tracker_id":  t.tracker_id,
            "timestamp":   str(t.timestamp)
        } for t in thefts]
    }

# ── Products ──────────────────────────────────────────────────────────────────
@app.get("/products")
def get_all_products():
    available = get_products_by_status_api('available')
    stolen    = get_products_by_status_api('stolen')
    sold      = get_products_by_status_api('sold')

    return {
        "total":     len(available) + len(stolen) + len(sold),
        "available": available,
        "stolen":    stolen,
        "sold":      sold
    }

@app.get("/products/available")
def get_available():
    products = get_products_by_status_api('available')
    return {"count": len(products), "products": products}

@app.get("/products/stolen")
def get_stolen():
    products = get_products_by_status_api('stolen')
    return {"count": len(products), "products": products}

@app.get("/products/sold")
def get_sold():
    products = get_products_by_status_api('sold')
    return {"count": len(products), "products": products}

@app.get("/products/{tracker_id}/status")
def get_product_status(tracker_id: int):
    status = check_product_status(tracker_id)
    return {
        "tracker_id": tracker_id,
        "status":     status,
        "timestamp":  str(datetime.now())
    }

# ── Mark as Sold ──────────────────────────────────────────────────────────────
@app.post("/products/sell")
def sell_product(request: SellRequest):
    success = mark_as_sold(request.tracker_id)
    if success:
        return {
            "success":    True,
            "message":    f"Product ID#{request.tracker_id} marked as sold",
            "tracker_id": request.tracker_id,
            "timestamp":  str(datetime.now())
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Product ID#{request.tracker_id} not found in database"
        )

# ── Sessions ──────────────────────────────────────────────────────────────────
@app.get("/sessions")
def get_sessions():
    session = Session()
    try:
        logs = session.query(SessionLog)\
                     .order_by(SessionLog.start_time.desc())\
                     .all()
        return {
            "count": len(logs),
            "sessions": [{
                "id":                l.id,
                "start_time":        str(l.start_time),
                "end_time":          str(l.end_time),
                "total_frames":      l.total_frames,
                "total_detections":  l.total_detections,
                "total_thefts":      l.total_thefts
            } for l in logs]
        }
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
    finally:
        session.close()

# ── Run API ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("RETAIL VISION - REST API")
    print("=" * 50)
    print("API running at: http://localhost:8000")
    print("Documentation:  http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)