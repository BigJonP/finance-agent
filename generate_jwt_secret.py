import secrets


def generate_jwt_secret():
    """Generate a secure JWT secret key"""
    secret = secrets.token_urlsafe(32)
    print("Generated JWT Secret Key:")
    print("=" * 50)
    print(secret)
    print("=" * 50)
    print("\nAdd this to your .env file:")
    print(f"JWT_SECRET_KEY={secret}")
    print("\n⚠️  Keep this secret secure and never commit it to version control!")


if __name__ == "__main__":
    generate_jwt_secret()
