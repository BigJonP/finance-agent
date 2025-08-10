from asyncio.log import logger
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from supabase import create_client, Client
from dotenv import load_dotenv
import bcrypt

load_dotenv()


class SupabaseManager:
    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv(
            "SUPABASE_ANON_KEY"
        )

        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables"
            )

        self.client: Client = create_client(supabase_url, supabase_key)

    def _hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def _verify_password(self, password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

    async def create_user(
        self, username: str, email: str, password: str
    ) -> Dict[str, Any]:
        try:
            hashed_password = self._hash_password(password)

            response = (
                self.client.table("user")
                .insert(
                    {
                        "username": username,
                        "email": email,
                        "password": hashed_password,
                        "created_at": datetime.now().isoformat(),
                    }
                )
                .execute()
            )

            if response.data:
                user = response.data[0]

                if "password" in user:
                    del user["password"]
                return user
            else:
                raise Exception("Failed to create user")

        except Exception as e:
            raise Exception(f"Error creating user: {str(e)}")

    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        try:
            response = self.client.table("user").select("*").eq("id", user_id).execute()

            if response.data:
                return response.data[0]
            return None

        except Exception as e:
            raise Exception(f"Error retrieving user: {str(e)}")

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        try:
            response = (
                self.client.table("user").select("*").eq("email", email).execute()
            )

            if response.data:
                return response.data[0]
            return None

        except Exception as e:
            raise Exception(f"Error retrieving user by email: {str(e)}")

    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        try:
            response = (
                self.client.table("user").select("*").eq("username", username).execute()
            )

            if response.data:
                return response.data[0]
            return None

        except Exception as e:
            raise Exception(f"Error retrieving user by username: {str(e)}")

    async def verify_user_credentials(
        self, username: str, password: str
    ) -> Optional[Dict[str, Any]]:
        try:
            user = await self.get_user_by_username(username)
            if user and self._verify_password(password, user["password"]):
                return user
            return None
        except Exception as e:
            logger.error(f"Error verifying user credentials: {str(e)}")
            raise Exception("Incorrect username or password")

    async def update_user(
        self,
        user_id: int,
        username: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            update_data = {}
            if username is not None:
                update_data["username"] = username
            if email is not None:
                update_data["email"] = email
            if password is not None:
                update_data["password"] = self._hash_password(password)

            if not update_data:
                raise ValueError("No fields to update")

            response = (
                self.client.table("user")
                .update(update_data)
                .eq("id", user_id)
                .execute()
            )

            if response.data:
                return response.data[0]
            else:
                raise Exception("Failed to update user")

        except Exception as e:
            raise Exception(f"Error updating user: {str(e)}")

    async def delete_user(self, user_id: int) -> bool:
        try:
            response = self.client.table("user").delete().eq("id", user_id).execute()
            return len(response.data) > 0

        except Exception as e:
            raise Exception(f"Error deleting user: {str(e)}")

    async def create_holding(self, user_id: int, stock: str) -> Dict[str, Any]:
        try:
            response = (
                self.client.table("holding")
                .insert({"user_id": user_id, "stock": stock})
                .execute()
            )

            if response.data:
                return response.data[0]
            else:
                raise Exception("Failed to create holding")

        except Exception as e:
            raise Exception(f"Error creating holding: {str(e)}")

    async def get_holdings_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        try:
            response = (
                self.client.table("holding")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )
            return response.data

        except Exception as e:
            raise Exception(f"Error retrieving user holdings: {str(e)}")

    async def delete_holding_by_user_and_stock(self, user_id: int, stock: str) -> bool:
        try:
            response = (
                self.client.table("holding")
                .delete()
                .eq("user_id", user_id)
                .eq("stock", stock)
                .execute()
            )
            return len(response.data) > 0

        except Exception as e:
            raise Exception(f"Error deleting holding by user and stock: {str(e)}")

    async def get_stock_metadata(self, stock: str) -> Optional[Dict[str, Any]]:
        try:
            response = (
                self.client.table("stock_metadata")
                .select("*")
                .eq("stock", stock)
                .execute()
            )

            if response.data:
                return response.data[0]
            return None

        except Exception as e:
            raise Exception(f"Error retrieving stock metadata: {str(e)}")


async def create_user(username: str, email: str, password: str) -> Dict[str, Any]:
    manager = SupabaseManager()
    return await manager.create_user(username, email, password)


async def get_user(user_id: int) -> Optional[Dict[str, Any]]:
    manager = SupabaseManager()
    return await manager.get_user_by_id(user_id)


async def create_holding(user_id: int, stock: str) -> Dict[str, Any]:
    manager = SupabaseManager()
    return await manager.create_holding(user_id, stock)


async def get_user_holdings(user_id: int) -> List[Dict[str, Any]]:
    manager = SupabaseManager()
    return await manager.get_holdings_by_user(user_id)


async def get_stock_metadata(stock: str) -> Optional[Dict[str, Any]]:
    manager = SupabaseManager()
    return await manager.get_stock_metadata(stock)
