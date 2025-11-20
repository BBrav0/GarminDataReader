#!/usr/bin/env python3
"""
Fitbit OAuth2 Token Manager
This script handles automatic token refresh and only requires manual authorization when necessary.
"""
import os
import sys
from pathlib import Path

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

import webbrowser
import urllib.parse
import requests
import base64
import time
from dotenv import load_dotenv, set_key, find_dotenv
from datetime import datetime, timedelta

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv()

# Your Fitbit app credentials
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REDIRECT_URI = 'http://127.0.0.1:8080/'

def is_token_expired(access_token):
    """Check if the access token is expired by decoding JWT payload."""
    try:
        # JWT tokens have 3 parts separated by dots
        parts = access_token.split('.')
        if len(parts) != 3:
            return True  # Invalid token format
        
        # Decode the payload (middle part)
        payload = parts[1]
        # Add padding if needed
        payload += '=' * (-len(payload) % 4)
        decoded = base64.b64decode(payload)
        import json
        token_data = json.loads(decoded)
        
        # Check expiration time
        exp_time = token_data.get('exp')
        if exp_time:
            # Convert to datetime and compare with current time (with 5-minute buffer)
            exp_datetime = datetime.fromtimestamp(exp_time)
            return datetime.now() >= exp_datetime - timedelta(minutes=5)
        
        return True  # No expiration found, assume expired
    except Exception as e:
        print(f"Error checking token expiration: {e}")
        return True  # Assume expired if we can't parse

def refresh_access_token(refresh_token):
    """Use refresh token to get a new access token."""
    token_url = "https://api.fitbit.com/oauth2/token"
    headers = {
        'Authorization': f'Basic {base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    data = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'expires_in': 31536000  # 1 year
    }
    
    try:
        response = requests.post(token_url, headers=headers, data=data)
        response.raise_for_status()
        
        tokens = response.json()
        print("✓ Token refresh successful")
        
        # Update environment file with new tokens
        set_key(dotenv_path, "ACCESS_TOKEN", tokens['access_token'])
        if 'refresh_token' in tokens:
            set_key(dotenv_path, "REFRESH_TOKEN", tokens['refresh_token'])
        
        # Reload environment variables
        load_dotenv()
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Token refresh failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return False

def manual_authorization():
    """Perform manual OAuth2 authorization flow."""
    print("\n" + "="*40)
    print("MANUAL AUTHORIZATION REQUIRED")
    print("="*40)
    
    # Step 1: Generate authorization URL
    auth_url = f"https://www.fitbit.com/oauth2/authorize?response_type=code&client_id={CLIENT_ID}&redirect_uri={urllib.parse.quote(REDIRECT_URI)}&scope=activity%20heartrate%20location%20nutrition%20profile%20settings%20sleep%20social%20weight"

    print("Step 1: Authorization")
    print("Click the link below to authorize your app:")
    print(auth_url)
    print()

    # Open browser automatically
    print("Opening browser...")
    webbrowser.open(auth_url)

    print("After authorizing, you'll be redirected to a URL that looks like:")
    print(f"{REDIRECT_URI}?code=YOUR_AUTHORIZATION_CODE")
    print()
    print("Copy the authorization code from the URL and paste it below:")

    # Get authorization code from user
    auth_code = input("Enter the authorization code: ").strip()

    if not auth_code:
        print("No authorization code provided. Exiting.")
        return False

    print("\nStep 2: Getting tokens...")

    # Step 2: Exchange authorization code for tokens
    token_url = "https://api.fitbit.com/oauth2/token"
    headers = {
        'Authorization': f'Basic {base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {
        'client_id': CLIENT_ID,
        'grant_type': 'authorization_code',
        'redirect_uri': REDIRECT_URI,
        'code': auth_code,
        'expires_in': 31536000  # 1 year
    }

    try:
        response = requests.post(token_url, headers=headers, data=data)
        response.raise_for_status()
        
        tokens = response.json()
        
        print("✓ Manual authorization successful")

        set_key(dotenv_path, "ACCESS_TOKEN", tokens['access_token'])
        set_key(dotenv_path, "REFRESH_TOKEN", tokens['refresh_token'])
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error getting tokens: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return False

def ensure_valid_tokens():
    """Main function to ensure we have valid tokens, using refresh when possible."""
    print("Fitbit OAuth2 Token Manager")
    print("=" * 40)
    print(f"Client ID: {CLIENT_ID}")
    print(f"Redirect URI: {REDIRECT_URI}")
    print()

    if not CLIENT_ID or not CLIENT_SECRET:
        print("✗ Error: Please set CLIENT_ID and CLIENT_SECRET in your .env file")
        return False

    # Check existing tokens
    access_token = os.getenv('ACCESS_TOKEN')
    refresh_token = os.getenv('REFRESH_TOKEN')

    if not access_token or not refresh_token:
        print("⚠ No existing tokens found, initiating manual authorization...")
        return manual_authorization()

    # Check if access token is expired
    if is_token_expired(access_token):
        print("⚠ Access token expired, attempting refresh...")
        
        # Try to refresh the token
        if refresh_access_token(refresh_token):
            print("✓ Tokens refreshed successfully")
            return True
        else:
            print("⚠ Token refresh failed, initiating manual authorization...")
            return manual_authorization()
    else:
        print("✓ Existing tokens are valid")
        return True

if __name__ == "__main__":
    success = ensure_valid_tokens()
    if not success:
        exit(1)
