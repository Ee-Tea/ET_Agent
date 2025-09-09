#!/usr/bin/env python3
"""
Database initialization script
Creates tables and runs migrations
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.db import engine
from common.models.base import Base
from common.models.user import User, OAuthAccount
from common.config import settings

def create_tables():
    """Create all database tables"""
    print("🔨 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")

def main():
    """Main initialization function"""
    print("🚀 Initializing database...")
    print(f"Database URL: {settings.DATABASE_URL}")
    
    try:
        create_tables()
        print("🎉 Database initialization completed successfully!")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

