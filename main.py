import logging
from app.runner import run_scrappers
from app.services.process_youtube import process_youtube_transcripts
from app.services.process_anthropic_markdown import process_anthropic_markdown
from app.services.process_digest import process_digests
from app.services.process_email import send_digest_email

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting AI News Aggregator Pipeline...")
    
    # Step 1: Run scrappers to fetch new articles and videos
    logger.info("Step 1/5: Running scrappers...")
    try:
        results = run_scrappers(hours=150)
        logger.info(f"Scraped {len(results['youtube'])} YouTube videos, {len(results['openai'])} OpenAI articles, and {len(results['anthropic'])} Anthropic articles.")
    except Exception as e:
        logger.error(f"Error in running scrappers: {e}")
        
    # Step 2: Download YouTube transcripts
    logger.info("Step 2/5: Processing YouTube transcripts...")
    try:
        yt_results = process_youtube_transcripts()
        logger.info(f"Processed {yt_results['processed']} transcripts. Unavailable: {yt_results['unavailable']}. Failed: {yt_results['failed']}.")
    except Exception as e:
        logger.error(f"Error processing YouTube transcripts: {e}")

    # Step 3: Fetch Anthropic markdown
    logger.info("Step 3/5: Processing Anthropic markdown...")
    try:
        anthropic_results = process_anthropic_markdown()
        logger.info(f"Processed {anthropic_results['processed']} markdown articles. Failed: {anthropic_results['failed']}.")
    except Exception as e:
        logger.error(f"Error processing Anthropic markdown: {e}")
        
    # Step 4: Generate digests for new content
    logger.info("Step 4/5: Generating digests...")
    try:
        digest_results = process_digests()
        logger.info(f"Generated {digest_results['processed']} digests. Failed: {digest_results['failed']}.")
    except Exception as e:
        logger.error(f"Error generating digests: {e}")
        
    # Step 5: Send email digest
    logger.info("Step 5/5: Generating and sending email digest...")
    email_success = False
    try:
        email_results = send_digest_email(hours=150, top_n=10)
        if email_results.get("success"):
            logger.info(f"Email sent successfully! Subject: {email_results['subject']} | Articles: {email_results['articles_count']}")
            email_success = True
        else:
            logger.error(f"Failed to send email: {email_results.get('error')}")
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        
    logger.info("Pipeline finished!")
    if not email_success:
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
