import sqlite3
import json

conn = sqlite3.connect("./data/pr_intelligence.db")
conn.row_factory = sqlite3.Row
print("Table list:")
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
for row in cursor.fetchall():
    print("- ", row["name"])

print("\nLast 5 mentions:")
try:
    cursor = conn.execute("SELECT id, title, sentiment, topics, web3_signals FROM mentions ORDER BY created_at DESC LIMIT 5")
    for row in cursor.fetchall():
        print(f"ID: {row['id']}")
        print(f"  Title: {row['title']}")
        print(f"  Sentiment: {row['sentiment']}")
        print(f"  Topics: {row['topics']}")
        print(f"  Web3: {row['web3_signals']}")
except Exception as e:
    print("Error reading mentions:", e)
conn.close()
