#!/usr/bin/env python3
"""
Garmin Connect Authentication Manager
This script verifies Garmin credentials and ensures authentication is set up.
The garminconnect library handles token storage automatically via garth.
"""
import os
import sys
from pathlib import Path

# Check Python version (requires 3.6+ for f-strings and garminconnect)
if sys.version_info < (3, 6):
    print("Error: Python 3.6 or higher is required. Current version: {}.{}".format(sys.version_info.major, sys.version_info.minor))
    sys.exit(1)

# Auto-detect and use venv Python if available
def ensure_venv():
    """Re-execute script with venv Python if not already using it."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return  # Already in a venv
    
    script_dir = Path(__file__).parent.absolute()
    venv_python = script_dir / "venv" / "bin" / "python3"
    
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

ensure_venv()

from dotenv import load_dotenv, set_key, find_dotenv
from garminconnect import Garmin

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv()

# Garmin credentials
GARMIN_EMAIL = os.getenv('GARMIN_EMAIL')
GARMIN_PASSWORD = os.getenv('GARMIN_PASSWORD')

def test_connection(email, password):
    """Test Garmin connection with provided credentials."""
    try:
        print("Testing Garmin connection...")
        client = Garmin(email, password)
        client.login()
        print("✓ Connection successful")
        
        # Get user profile to verify
        profile = client.get_user_profile()
        display_name = profile.get('displayName', 'User')
        print(f"✓ Authenticated as: {display_name}")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def ensure_valid_credentials():
    """Main function to ensure we have valid Garmin credentials."""
    print("Garmin Connect Authentication Manager")
    print("=" * 40)
    print()
    
    # Check if credentials exist
    if not GARMIN_EMAIL or not GARMIN_PASSWORD:
        print("⚠ Garmin credentials not found in .env file")
        print("Please set GARMIN_EMAIL and GARMIN_PASSWORD in your .env file")
        print()
        print("Would you like to enter them now? (y/n): ", end='')
        response = input().strip().lower()
        
        if response == 'y':
            email = input("Enter your Garmin email: ").strip()
            password = input("Enter your Garmin password: ").strip()
            
            if email and password:
                set_key(dotenv_path, "GARMIN_EMAIL", email)
                set_key(dotenv_path, "GARMIN_PASSWORD", password)
                print("✓ Credentials saved to .env file")
                
                # Test the connection
                if test_connection(email, password):
                    print("✓ Authentication setup complete")
                    print("Note: Tokens are stored automatically and last ~1 year")
                    return True
                else:
                    print("✗ Authentication test failed. Please check your credentials.")
                    return False
            else:
                print("✗ No credentials provided")
                return False
        else:
            print("✗ Credentials required. Exiting.")
            return False
    else:
        print(f"✓ Credentials found in .env file")
        print(f"Email: {GARMIN_EMAIL}")
        
        # Test the connection
        if test_connection(GARMIN_EMAIL, GARMIN_PASSWORD):
            print("✓ Authentication verified")
            print("Note: Tokens are stored automatically and last ~1 year")
            return True
        else:
            print("⚠ Connection test failed. Your credentials may need to be updated.")
            return False

if __name__ == "__main__":
    success = ensure_valid_credentials()
    if not success:
        exit(1)
