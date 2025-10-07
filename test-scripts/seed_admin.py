import os
import sys
from typing import NoReturn

# Ensure repo root on sys.path so we can import config/app modules when run from test-scripts
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import asyncio
from getpass import getpass

from app.database.conn import mongo_client
from app.services.auth.auth import seed_admin
from app.models.admin.admin import Admin


async def main() -> NoReturn:
    email = os.getenv("SEED_ADMIN_EMAIL") or input("Admin email: ")
    password = os.getenv("SEED_ADMIN_PASSWORD") or getpass("Admin password: ")
    name = os.getenv("SEED_ADMIN_NAME") or input("Admin name (optional): ") or None

    await mongo_client.connect()
    try:
        admin = Admin(email=email, password=password, name=name)
        created = await seed_admin(admin)
        print({"_id": created.id, "email": created.email, "name": created.name})
    finally:
        if mongo_client.client:
            await mongo_client.close()


if __name__ == "__main__":
    asyncio.run(main())


