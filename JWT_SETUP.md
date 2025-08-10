# JWT Authentication Setup

This document explains how to set up JWT authentication for the Finance Agent API.

## Environment Variables

Add the following environment variables to your `.env` file:

```bash
# JWT Configuration
JWT_SECRET_KEY=your_jwt_secret_key_here_change_in_production
```


```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Dependencies

The following packages have been added to `requirements.txt`:

- `python-jose[cryptography]` - For JWT token creation and validation
- `python-multipart` - For handling form data in FastAPI

Install the new dependencies:

```bash
pip install -r requirements.txt
```

## API Changes

### Authentication Flow

1. **User Registration**: `POST /user/` (no authentication required)
2. **User Sign In**: `POST /user/signin` (no authentication required)
3. **Protected Endpoints**: All other endpoints require JWT authentication

### Sign In Response

The signin endpoint now returns:

```json
{
  "user": {
    "id": 123,
    "username": "john_doe",
    "email": "john@example.com",
    "created_at": "2024-01-01T00:00:00"
  },
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```
