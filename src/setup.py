import logging
import sys
import os

from db.db_util import DatabaseUtil


def setup_database(db_path: str = "finance_agent.db"):
    print(f"Setting up database at: {db_path}")
    print("=" * 50)

    db = DatabaseUtil(db_path)

    try:
        print("Clearing existing data...")
        db.clear_all_data()

        print("\nCreating sample users...")
        users_data = [
            ("alice_trader", "alice@tradingfirm.com"),
            ("bob_investor", "bob@investments.com"),
            ("charlie_analyst", "charlie@analysis.com"),
            ("diana_portfolio", "diana@portfolio.com"),
            ("eve_quant", "eve@quantfund.com"),
        ]

        user_ids = {}
        for username, email in users_data:
            user_id = db.create_user(username, email)
            user_ids[username] = user_id
            print(f"  Created user: {username} (ID: {user_id})")

        print("\nCreating sample stock mappings...")
        stock_mappings = [
            ("alice_trader", ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]),
            ("bob_investor", ["AAPL", "JNJ", "PG", "KO", "WMT", "JPM"]),
            ("charlie_analyst", ["JPM", "BAC", "GS", "MS", "WFC"]),
            ("diana_portfolio", ["TSLA", "NVDA", "AMZN", "NFLX", "META"]),
            ("eve_quant", ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "NVDA"]),
        ]

        total_mappings = 0
        for username, symbols in stock_mappings:
            user_id = user_ids[username]
            print(f"  Adding stocks for {username}:")
            for symbol in symbols:
                success = db.add_stock(user_id, symbol)
                if success:
                    print(f"    ✓ {symbol}")
                    total_mappings += 1
                else:
                    print(f"    ✗ {symbol} (already exists)")

        print(f"\nTotal stock mappings created: {total_mappings}")

        print("\n" + "=" * 50)
        print("DATABASE SETUP COMPLETE")
        print("=" * 50)

        print("\nCreated Users:")
        users = db.get_all_users()
        for user in users:
            stock_count = db.get_user_stock_count(user["id"])
            print(f"  {user['username']} ({user['email']}) - {stock_count} stocks")

        print("\nStock Distribution:")
        all_stocks = db.get_all_stocks()
        stock_counts = {}
        for stock in all_stocks:
            symbol = stock["symbol"]
            stock_counts[symbol] = stock_counts.get(symbol, 0) + 1

        print("\nWho owns AAPL:")
        aapl_owners = db.get_stocks_by_symbol("AAPL")
        owner_names = [owner["username"] for owner in aapl_owners]
        print(f"  {', '.join(owner_names)}")

        print(f"\nDatabase successfully created at: {db_path}")

    except Exception as e:
        print(f"Error setting up database: {e}")
        raise
    finally:
        db.close()


def clear_database(db_path: str = "finance_agent.db"):
    print(f"Clearing database at: {db_path}")

    db = DatabaseUtil(db_path)
    try:
        db.clear_all_data()
        print("Database cleared successfully!")
    except Exception as e:
        print(f"Error clearing database: {e}")
        raise
    finally:
        db.close()


def verify_database(db_path: str = "finance_agent.db"):
    print(f"Verifying database at: {db_path}")
    print("=" * 50)

    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist!")
        return

    db = DatabaseUtil(db_path)
    try:
        users = db.get_all_users()
        stocks = db.get_all_stocks()

        print(f"Users: {len(users)}")
        print(f"Stock mappings: {len(stocks)}")

        if users:
            print("\nUsers:")
            for user in users:
                stock_count = db.get_user_stock_count(user["id"])
                print(f"  {user['username']} - {stock_count} stocks")

        if stocks:
            print("\nStock mappings:")
            for stock in stocks:
                user = db.get_user(stock["user_id"])
                print(f"  {user['username']} -> {stock['symbol']}")

    except Exception as e:
        print(f"Error verifying database: {e}")
        raise
    finally:
        db.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Database setup utility for finance agent")
    parser.add_argument("action", choices=["setup", "clear", "verify"], help="Action to perform")
    parser.add_argument(
        "--db-path",
        default="finance_agent.db",
        help="Path to the database file (default: finance_agent.db)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        if args.action == "setup":
            setup_database(args.db_path)
        elif args.action == "clear":
            clear_database(args.db_path)
        elif args.action == "verify":
            verify_database(args.db_path)
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
