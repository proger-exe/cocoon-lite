import asyncio
import os
from getpass import getpass
from dotenv import load_dotenv
from pyrogram import Client

load_dotenv()

async def main():
    async with Client('login', api_id=int(os.environ['TG_API_ID']),
                      api_hash=os.environ['TG_API_HASH'], in_memory=True) as client:
        # Treat the result as an account credential; put it only in your local .env.
        print('\nTG_SESSION_STRING=' + await client.export_session_string())

if __name__ == '__main__':
    asyncio.run(main())
