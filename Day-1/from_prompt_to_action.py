import dotenv
import os
import logging
import asyncio 
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

# 2. Set the level of the logger to INFO
logging.basicConfig(
    level=logging.INFO, # Sets the minimum level to be INFO
    
)

# Set up logging
logger = logging.getLogger(__name__)

logger.info("Starting agent setup...")
# Load environment variables from .env file
logger.info("Loading environment variables from .env file...")
try:
    dotenv.load_dotenv()
    logger.info(".env file loaded successfully.")
except Exception as e:
    logger.error("Failed to load .env file.")
    print(f"Error loading .env file: {e}")

# API key to work with google API's.
# The reason to use an API key is to keep your chats 
# specific to you/your API. 
# Outside of this tutorial, do not share your API key with anyone.
logger.info("Retrieving GOOGLE_API_KEY from environment variables...")
GOOGLE_API_KEY = os.getenv("GOOGLE_API-KEY")



# Configure retry settings for Gemini model
# If you encounter rate limiting or transient errors, 
# these settings will help manage retries.
logger.info("Configuring retry settings for Gemini model...")
retry_settings = types.HttpRetryOptions(
    attempts=5, # Max retry attemps
    exp_base=2, # delay multiplier  
    initial_delay=1,
    http_status_codes=[492, 500, 503, 504],
)


logger.info("Initializing the agent with Google Search tool...")


# Initialize the agent with Google Search tool
root_agent = Agent(
    name="helpful_assistant",
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    description="An AI assistant that helps users find information using Google Search.",
    tools=[google_search],
)

logger.info("Agent initialized successfully.")

# Create an in-memory runner to execute the agent's tasks
runner = InMemoryRunner(agent=root_agent)

logger.info("Runner created.")

async def main():
    response = await runner.run_debug("Who won the FIFA World Cup in 2022?")

if __name__ == "__main__":
    asyncio.run(main())


