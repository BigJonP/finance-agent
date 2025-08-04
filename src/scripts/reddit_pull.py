import praw
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RedditScraper:

    def __init__(self, client_id: str, client_secret: str, user_agent: str, subreddit_name: str):
        self.reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret, user_agent=user_agent
        )
        self.subreddit_name = subreddit_name
        self.subreddit = self.reddit.subreddit(self.subreddit_name)

    def get_posts_from_last_hours(self, hours: int = 12) -> List[Dict[str, Any]]:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        cutoff_timestamp = cutoff_time.timestamp()

        posts = []
        logger.info(f"Fetching posts from r/{self.subreddit_name} from the last {hours} hours...")

        try:
            for submission in self.subreddit.new(limit=None):
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
            "created_datetime": datetime.fromtimestamp(submission.created_utc).isoformat(),
            "url": submission.url,
            "permalink": f"https://reddit.com{submission.permalink}",
            "selftext": submission.selftext,
            "is_self": submission.is_self,
        }

    def save_posts_to_json(
        self, posts: List[Dict[str, Any]], filename: Optional[str] = None
    ) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wsb_posts_{timestamp}.json"

        output_dir = Path("data/reddit")
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "subreddit": self.subreddit_name,
                        "scraped_at": datetime.now().isoformat(),
                        "total_posts": len(posts),
                        "time_window_hours": 12,
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

    return {"client_id": client_id, "client_secret": client_secret, "user_agent": user_agent}


def main():
    try:
        credentials = load_reddit_credentials()

        scraper = RedditScraper(
            client_id=credentials["client_id"],
            client_secret=credentials["client_secret"],
            user_agent=credentials["user_agent"],
            subreddit_name="WallStreetBets",
        )

        posts = scraper.get_posts_from_last_hours(hours=12)

        if posts:
            output_file = scraper.save_posts_to_json(posts)
            logger.info(f"\nPosts saved to: {output_file}")
        else:
            logger.info("No posts found in the last 12 hours.")

    except Exception as e:
        logger.error(f"Error running scraper: {e}")


if __name__ == "__main__":
    main()
