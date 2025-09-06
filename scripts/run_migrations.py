#!/usr/bin/env python3
"""
Run database migrations using Alembic
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_migrations():
    """Run Alembic migrations"""
    print("🔄 Running database migrations...")
    
    try:
        # Change to project directory
        os.chdir(project_root)
        
        # Run alembic upgrade
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Migrations completed successfully!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Migration failed: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

def create_initial_migration():
    """Create initial migration"""
    print("📝 Creating initial migration...")
    
    try:
        # Change to project directory
        os.chdir(project_root)
        
        # Create migration
        result = subprocess.run(
            ["alembic", "revision", "--autogenerate", "-m", "Initial migration"],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Initial migration created successfully!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Migration creation failed: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        create_initial_migration()
    else:
        run_migrations()

if __name__ == "__main__":
    main()

