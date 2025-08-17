import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import praw
from time import sleep
from dotenv import load_dotenv
from langchain_core.documents import Document

from retriever.config import VECTOR_STORE_CONFIG
from retriever.vector_store import get_vector_store
from scripts.config import REDDIT_PULL_CONFIG

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RedditScraper:
    def __init__(
        self, client_id: str, client_secret: str, user_agent: str, subreddit_name: str
    ):
        self.reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret, user_agent=user_agent
        )

    def get_posts_from_last_hours(
        self, subreddit_name: str, hours: int
    ) -> List[Dict[str, Any]]:
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        cutoff_timestamp = cutoff_time.timestamp()

        posts = []
        logger.info(
            f"Fetching posts from r/{subreddit_name} from the last {hours} hours..."
        )

        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            for submission in subreddit.new(limit=None):
                if submission.created_utc >= cutoff_timestamp:
                    post_data = self._extract_post_data(submission)
                    posts.append(post_data)
                    logger.info(f"Found post: {post_data['title'][:50]}...")
                else:
                    logger.info(f"Reached posts older than {hours} hours, stopping...")
                    break

        except Exception as e:
            logger.error(f"Error fetching posts: {e}")
            raise

        logger.info(f"Total posts found: {len(posts)}")
        return posts

    def _extract_post_data(self, submission) -> Dict[str, Any]:
        return {
            "id": submission.id,
            "title": submission.title,
            "score": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "created_utc": submission.created_utc,
            "created_datetime": datetime.fromtimestamp(
                submission.created_utc
            ).isoformat(),
            "url": submission.url,
            "permalink": f"https://reddit.com{submission.permalink}",
            "selftext": submission.selftext,
            "is_self": submission.is_self,
        }

    def save_posts_to_json(
        self,
        posts: List[Dict[str, Any]],
        filename: Optional[str] = None,
        hours: int = 12,
        subreddit_name: str = "WallStreetBets",
    ) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{subreddit_name}_posts_{timestamp}.json"

        output_dir = Path("data/reddit")
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "subreddit": subreddit_name,
                        "scraped_at": datetime.now().isoformat(),
                        "total_posts": len(posts),
                        "time_window_hours": hours,
                    },
                    "posts": posts,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"Saved {len(posts)} posts to {filepath}")
        return str(filepath)


def load_reddit_credentials() -> Dict[str, str]:
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "Reddit_Scraper_Bot/1.0")

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "user_agent": user_agent,
    }


def main(subreddit_name: str, hours: int):
    logger.info(f"=== Starting to process r/{subreddit_name} ===")

    try:
        credentials = load_reddit_credentials()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        scraper = RedditScraper(
            client_id=credentials["client_id"],
            client_secret=credentials["client_secret"],
            user_agent=credentials["user_agent"],
            subreddit_name=subreddit_name,
        )

        posts = scraper.get_posts_from_last_hours(
            subreddit_name=subreddit_name, hours=hours
        )

        logger.info(f"Retrieved {len(posts)} posts from r/{subreddit_name}")

        logger.info("Initializing vector store...")
        vector_store = get_vector_store(VECTOR_STORE_CONFIG)
        logger.info("Vector store initialized successfully")

        initial_count = vector_store.get_document_count()
        logger.info(f"Initial document count in vector store: {initial_count}")

        documents = [
            Document(
                page_content=post["title"] + "\n" + (post["selftext"] or ""),
                metadata={"source": post["permalink"], "timestamp": timestamp},
            )
            for post in posts
            if post["title"] and (post["title"] + (post["selftext"] or "")).strip()
        ]

        logger.info(f"Created {len(documents)} valid documents from {len(posts)} posts")

        if not documents:
            logger.warning("No valid documents to add to vector store")
            return

        logger.info("Adding documents to vector store...")
        import asyncio

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(vector_store.add_documents(documents))
            logger.info("Documents added to vector store successfully")
        finally:
            loop.close()

        final_count = vector_store.get_document_count()
        logger.info(f"Final document count in vector store: {final_count}")
        logger.info(f"Documents added: {final_count - initial_count}")

        if posts:
            output_file = scraper.save_posts_to_json(
                posts,
                filename=f"{timestamp}.json",
                subreddit_name=subreddit_name,
                hours=hours,
            )
            logger.info(f"Posts saved to: {output_file}")
        else:
            logger.info(f"No posts found in the last {hours} hours.")

        logger.info(f"=== Successfully completed processing r/{subreddit_name} ===")

    except Exception as e:
        logger.error(f"=== Error processing r/{subreddit_name}: {e} ===")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    logger.info(f"Starting to process {len(REDDIT_PULL_CONFIG)} subreddits")

    for i, config in enumerate(REDDIT_PULL_CONFIG, 1):
        logger.info(
            f"=== Processing subreddit {i}/{len(REDDIT_PULL_CONFIG)}: {config['subreddit_name']} ==="
        )
        try:
            main(config["subreddit_name"], config["hours"])
            if i < len(REDDIT_PULL_CONFIG):
                logger.info("Waiting 3 seconds before next subreddit...")
                sleep(3)
        except Exception as e:
            logger.error(f"=== Failed to process r/{config['subreddit_name']}: {e} ===")
            logger.error("Continuing with next subreddit...")
            continue

    logger.info("=== Finished processing all subreddits ===")
