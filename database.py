# database.py
# Day 5: PostgreSQL Event Logging
# Purpose: Store all detections and theft alerts permanently
# Impact: Store owner can review history, generate reports, track patterns

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# ── Database Connection ───────────────────────────────────────────────────────
DB_URL = "postgresql://postgres@localhost/retail_vision_db"
engine = create_engine(DB_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)

# ── Table 1: Detection Events ─────────────────────────────────────────────────
# Logs every single object detection
class DetectionEvent(Base):
    __tablename__ = 'detection_events'

    id          = Column(Integer, primary_key=True, autoincrement=True)
    timestamp   = Column(DateTime, default=datetime.now)
    object_name = Column(String(100))
    tracker_id  = Column(Integer)
    zone        = Column(String(50))
    confidence  = Column(Float)
    frame       = Column(Integer)
    status      = Column(String(20), default='available')  # available or sold

    def __repr__(self):
        return f"<Detection {self.object_name} ID#{self.tracker_id} in {self.zone}>"

# ── Table 2: Theft Alerts ─────────────────────────────────────────────────────
# Logs every theft alert triggered
class TheftAlert(Base):
    __tablename__ = 'theft_alerts'

    id          = Column(Integer, primary_key=True, autoincrement=True)
    timestamp   = Column(DateTime, default=datetime.now)
    object_name = Column(String(100))
    tracker_id  = Column(Integer)
    frame       = Column(Integer)
    resolved    = Column(Boolean, default=False)  # for future use

    def __repr__(self):
        return f"<TheftAlert {self.object_name} ID#{self.tracker_id}>"

# ── Table 3: Session Log ──────────────────────────────────────────────────────
# Logs each time the system runs
class SessionLog(Base):
    __tablename__ = 'session_logs'

    id              = Column(Integer, primary_key=True, autoincrement=True)
    start_time      = Column(DateTime, default=datetime.now)
    end_time        = Column(DateTime, nullable=True)
    total_frames    = Column(Integer, default=0)
    total_detections= Column(Integer, default=0)
    total_thefts    = Column(Integer, default=0)

    def __repr__(self):
        return f"<Session {self.start_time} → {self.total_thefts} thefts>"

# ── Database Functions ────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist"""
    Base.metadata.create_all(engine)
    print("✅ Database tables created successfully!")

def log_detection(object_name, tracker_id, zone, confidence, frame, status='available'):
    """Save a detection event to database"""
    session = Session()
    try:
        event = DetectionEvent(
            object_name=object_name,
            tracker_id=tracker_id,
            zone=zone,
            confidence=float(confidence),
            frame=frame,
            status=status
        )
        session.add(event)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"DB Error (detection): {e}")
    finally:
        session.close()

def log_theft(object_name, tracker_id, frame):
    """Save a theft alert to database"""
    session = Session()
    try:
        alert = TheftAlert(
            object_name=str(object_name),
            tracker_id=int(tracker_id),
            frame=int(frame)
        )
        session.add(alert)
        session.commit()
        print(f"💾 Theft saved to database: {object_name} ID#{tracker_id}")
    except Exception as e:
        session.rollback()
        print(f"DB Error (theft): {e}")
    finally:
        session.close()

def start_session():
    """Log the start of a new monitoring session"""
    session = Session()
    try:
        log = SessionLog()
        session.add(log)
        session.commit()
        session_id = log.id
        return session_id
    except Exception as e:
        session.rollback()
        print(f"DB Error (session start): {e}")
        return None
    finally:
        session.close()

def end_session(session_id, total_frames, total_detections, total_thefts):
    """Log the end of a monitoring session"""
    session = Session()
    try:
        log = session.query(SessionLog).filter_by(id=session_id).first()
        if log:
            log.end_time = datetime.now()
            log.total_frames = total_frames
            log.total_detections = total_detections
            log.total_thefts = total_thefts
            session.commit()
            print(f"💾 Session #{session_id} saved to database!")
    except Exception as e:
        session.rollback()
        print(f"DB Error (session end): {e}")
    finally:
        session.close()

def get_recent_thefts(limit=10):
    """Get most recent theft alerts"""
    session = Session()
    try:
        thefts = session.query(TheftAlert)\
                       .order_by(TheftAlert.timestamp.desc())\
                       .limit(limit).all()
        return thefts
    except Exception as e:
        print(f"DB Error (query): {e}")
        return []
    finally:
        session.close()

def get_theft_summary():
    """Get summary statistics"""
    session = Session()
    try:
        total_thefts = session.query(TheftAlert).count()
        total_sessions = session.query(SessionLog).count()
        return {
            'total_thefts': total_thefts,
            'total_sessions': total_sessions
        }
    except Exception as e:
        print(f"DB Error (summary): {e}")
        return {}
    finally:
        session.close()
def mark_as_sold(tracker_id):
    """Mark a product as sold — will not trigger theft alert in exit zone"""
    session = Session()
    try:
        # Find the most recent detection of this tracker_id
        event = session.query(DetectionEvent)\
                      .filter_by(tracker_id=tracker_id)\
                      .order_by(DetectionEvent.timestamp.desc())\
                      .first()
        if event:
            event.status = 'sold'
            session.commit()
            print(f"✅ {event.object_name} ID#{tracker_id} marked as SOLD")
            return True
        else:
            print(f"⚠️ No product found with ID#{tracker_id}")
            return False
    except Exception as e:
        session.rollback()
        print(f"DB Error (mark sold): {e}")
        return False
    finally:
        session.close()

def mark_as_stolen(tracker_id):
    """Mark a product as stolen in database"""
    session = Session()
    try:
        event = session.query(DetectionEvent)\
                      .filter_by(tracker_id=tracker_id)\
                      .order_by(DetectionEvent.timestamp.desc())\
                      .first()
        if event:
            event.status = 'stolen'
            session.commit()
            print(f"🚨 {event.object_name} ID#{tracker_id} marked as STOLEN in database")
            return True
        else:
            return False
    except Exception as e:
        session.rollback()
        print(f"DB Error (mark stolen): {e}")
        return False
    finally:
        session.close()

def check_product_status(tracker_id):
    """
    Check if a product is available or sold
    Returns: 'available', 'sold', or 'not_found'
    """
    session = Session()
    try:
        event = session.query(DetectionEvent)\
                      .filter_by(tracker_id=tracker_id)\
                      .order_by(DetectionEvent.timestamp.desc())\
                      .first()
        if event:
            return event.status
        else:
            return 'not_found'
    except Exception as e:
        print(f"DB Error (check status): {e}")
        return 'not_found'
    finally:
        session.close()

# ── Test the database ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("RETAIL VISION - DATABASE SETUP")
    print("=" * 50)
    init_db()
    print("Database ready — no test data inserted.")
    print("=" * 50)