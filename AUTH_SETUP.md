# ET Agent Authentication Setup

This guide will help you set up PostgreSQL connection and Google OAuth authentication for the ET Agent system.

## Prerequisites

1. Docker and Docker Compose installed
2. Google Cloud Console project with OAuth 2.0 credentials
3. Python 3.12+ (for local development)

## Quick Start

### 1. Environment Configuration

Copy the example environment file and configure it:

```bash
cp env.example .env
```

Edit `.env` with your actual values:

```env
# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/postgres

***REMOVED*** Configuration
GOOGLE_CLIENT_ID=REDACTED
GOOGLE_CLIENT_SECRET=REDACTED
GOOGLE_REDIRECT_URI=http://localhost:8124/auth/google/callback

# JWT Configuration
SECRET_KEY=your-super-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS Configuration
ALLOWED_ORIGINS=["http://localhost:3000", "http://localhost:8123", "http://localhost:8124"]
```

### 2. Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Google+ API
4. Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client IDs"
5. Set application type to "Web application"
6. Add authorized redirect URIs:
   - `http://localhost:8124/auth/google/callback` (for development)
   - `https://yourdomain.com/auth/google/callback` (for production)
7. Copy the Client ID and Client Secret to your `.env` file

### 3. Start Services

Start all services including the new authentication API:

```bash
docker compose up -d
```

This will start:
- PostgreSQL database (port 5433)
- Redis (port 6380)
- LangGraph API (port 8123)
- Authentication API (port 8124)

### 4. Initialize Database

Run the database initialization script:

```bash
# Using Docker
docker compose exec auth-api python scripts/init_db.py

# Or locally (if you have Python installed)
python scripts/init_db.py
```

### 5. Test Authentication

Visit the authentication API documentation:
- Swagger UI: http://localhost:8124/docs
- ReDoc: http://localhost:8124/redoc

## API Endpoints

### Authentication Endpoints

- `GET /auth/google` - Initiate Google OAuth flow
- `GET /auth/google/callback` - Handle Google OAuth callback (web)
- `POST /auth/google/callback` - Handle Google OAuth callback (API)
- `GET /auth/me` - Get current user information
- `POST /auth/refresh` - Refresh access token
- `POST /auth/logout` - Logout user
- `GET /auth/health` - Health check

### Usage Examples

#### 1. Web Application Flow

```javascript
// Redirect user to Google OAuth
window.location.href = 'http://localhost:8124/auth/google';

// Handle callback (the API will redirect back with token)
// You can also use the callback endpoint to get JSON response
```

#### 2. API Integration

```python
import httpx

# Get auth URL
response = httpx.get("http://localhost:8124/auth/google")
auth_url = response.json()["auth_url"]

# After user authorizes, exchange code for token
auth_response = httpx.post(
    "http://localhost:8124/auth/google/callback",
    json={"code": "authorization_code_from_google"}
)
token_data = auth_response.json()

# Use token for authenticated requests
headers = {"Authorization": f"Bearer {token_data['token']['access_token']}"}
user_info = httpx.get("http://localhost:8124/auth/me", headers=headers)
```

#### 3. Frontend Integration (React/Vue/Angular)

```javascript
// Example with fetch
const authenticateWithGoogle = async () => {
  try {
    // Get auth URL
    const authResponse = await fetch('http://localhost:8124/auth/google');
    const { auth_url } = await authResponse.json();
    
    // Redirect to Google
    window.location.href = auth_url;
    
    // After callback, you'll have the token in the response
  } catch (error) {
    console.error('Authentication failed:', error);
  }
};

// Get current user
const getCurrentUser = async (token) => {
  const response = await fetch('http://localhost:8124/auth/me', {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  return response.json();
};
```

## Database Schema

### Users Table
- `id` (UUID, Primary Key)
- `email` (String, Unique)
- `name` (String)
- `is_active` (Boolean)
- `created_at` (DateTime)
- `updated_at` (DateTime)

### OAuth Accounts Table
- `id` (UUID, Primary Key)
- `user_id` (UUID, Foreign Key to users)
- `provider` (String) - 'google', 'github', etc.
- `provider_account_id` (String) - Google user ID
- `access_token` (Text, Optional)
- `refresh_token` (Text, Optional)
- `expires_at` (String, Optional)
- `scope` (Text, Optional)
- `created_at` (DateTime)
- `updated_at` (DateTime)

## Development

### Running Migrations

```bash
# Create new migration
python scripts/run_migrations.py create

# Run migrations
python scripts/run_migrations.py
```

### Local Development

```bash
# Install dependencies
pip install -e .

# Run authentication service
uvicorn common.main:app --host 0.0.0.0 --port 8000 --reload

# Run LangGraph API
uvicorn main:app --host 0.0.0.0 --port 8123 --reload
```

## Production Deployment

### Environment Variables

Make sure to set these environment variables in production:

```env
DATABASE_URL=postgresql://user:password@host:port/database
GOOGLE_CLIENT_ID=REDACTED
GOOGLE_CLIENT_SECRET=REDACTED
GOOGLE_REDIRECT_URI=https://yourdomain.com/auth/google/callback
SECRET_KEY=your-super-secure-secret-key
ALLOWED_ORIGINS=["https://yourdomain.com"]
```

### Security Considerations

1. **Secret Key**: Use a strong, random secret key for JWT signing
2. **HTTPS**: Always use HTTPS in production
3. **CORS**: Configure CORS origins properly
4. **Database**: Use connection pooling and proper credentials
5. **Tokens**: Consider implementing token refresh and expiration

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check if PostgreSQL is running: `docker compose ps`
   - Verify DATABASE_URL in .env file
   - Check database logs: `docker compose logs langgraph-postgres`

2. **Google OAuth Failed**
   - Verify GOOGLE_CLIENT_ID=REDACTED GOOGLE_CLIENT_SECRET=REDACTED Check redirect URI matches exactly
   - Ensure Google+ API is enabled

3. **CORS Issues**
   - Add your frontend domain to ALLOWED_ORIGINS
   - Check if credentials are being sent properly

4. **Token Issues**
   - Verify SECRET_KEY is set
   - Check token expiration
   - Ensure Authorization header format: `Bearer <token>`

### Logs

```bash
# View all logs
docker compose logs

# View specific service logs
docker compose logs auth-api
docker compose logs langgraph-api
docker compose logs langgraph-postgres
```

## Integration with LangGraph API

The authentication service is designed to work alongside your existing LangGraph API. You can:

1. **Protect LangGraph endpoints** by adding authentication middleware
2. **Pass user context** to your LangGraph workflows
3. **Store user-specific data** in the database
4. **Implement user sessions** and state management

Example integration:

```python
from fastapi import Depends, HTTPException, status
from common.auth_service import auth_service
from common.db import get_db

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    user = auth_service.get_current_user(db, token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

# Use in your LangGraph endpoints
@app.post("/graph/invoke")
async def invoke_graph(
    input_data: dict,
    current_user: User = Depends(get_current_user)
):
    # Add user context to your graph input
    input_data["user_id"] = current_user.id
    input_data["user_email"] = current_user.email
    
    # Invoke your graph
    result = your_graph.invoke(input_data)
    return result
```

This setup provides a complete authentication system that can be easily integrated with your existing ET Agent infrastructure.
