from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types
from google.adk.tools import FunctionTool, AgentTool, google_search


import dotenv
import os
import asyncio

import logging

# 2. Set the level of the logger to INFO
logging.basicConfig(
    level=logging.INFO, # Sets the minimum level to be INFO
    
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
logger.info("GOOGLE_API_KEY retrieved from environment variables.")

# Configure retry settings for Gemini model
# If you encounter rate limiting or transient errors,   
# these settings will help manage retries.
retry_settings = types.HttpRetryOptions(
    attempts=5, # Max retry attemps
    exp_base=2, # delay multiplier
    initial_delay=1,
    http_status_codes=[492, 500, 503, 504],
)

# research agent

research_agent = Agent(
    name="research_assistant",
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    description="An AI assistant that conducts research using Google Search.",
    tools=[google_search],
    output_key= "research_results"
)
logger.info("Research agent initialized successfully.")

# Summarizing agent
summarizing_agent = Agent(
    name="summarizing_assistant",
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    instruction="""Read the provided research results: {research_results} and Create a concise summary as a bulleted list with 3-5 key points.""",
    output_key="final_summary"
)
logger.info("Summarizing agent initialized successfully.")


# root agent that uses research and summarizing agents
root_agent = Agent(
    name="composite_assistant",
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    instruction="""You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.
1. First, you MUST call the `ResearchAgent` tool to find relevant information on the topic provided by the user.
2. Next, after receiving the research findings, you MUST call the `SummarizerAgent` tool to create a concise summary.
3. Finally, present the final summary clearly to the user as your response.""",
    tools=[AgentTool(research_agent), AgentTool(summarizing_agent)],
)
logger.info("Composite root agent initialized successfully.")

# create runner
runner = InMemoryRunner(agent=root_agent)
logger.info("Runner for composite agent created.")
async def main1():
    user_query = "What are the latest advancements in quantum computing and what do they mean for AI?"
    logger.info(f"Running agent for user query: {user_query}")
    response = await runner.run_debug(user_query)
    logger.info("Agent run completed.")
    

logger.info("Script execution completed.")


# Sequential workflows
outline_Agent = Agent(
    name="OutlineAgent",
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    instruction="""Create a blog outline for the given topic with:
    1. A catchy headline
    2. An introduction hook
    3. 3-5 main sections with 2-3 bullet points for each
    4. A concluding thought""",
    output_key="blog_outline",
)
logger.info("Outline agent initialized successfully.")

writer_Agent = Agent(
    name="WriterAgent", 
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    instruction="""Following this outline strictly: {blog_outline}
    Write a brief, 200 to 300-word blog post with an engaging and informative tone.""",
    output_key="blog_draft",
)
logger.info("Writer agent initialized successfully.")

editor_Agent = Agent(
    name="EditorAgent",
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    instruction="""Edit this draft: {blog_draft}
    Your task is to polish the text by fixing any grammatical errors, improving the flow and sentence structure, and enhancing overall clarity.""",
    output_key="final_blog_post",
)
logger.info("Editor agent initialized successfully.")

root_agent =   SequentialAgent(
    name="BlogPipeline",
    sub_agents=[outline_Agent, writer_Agent, editor_Agent],
)

logger.info("Sequential blog pipeline agent initialized successfully.")

runner = InMemoryRunner(agent=root_agent)
logger.info("Runner for blog pipeline agent created.")

async def main2():
    user_prompt = "Write a blog post about the benefits of multi-agent systems for software developers"
    logger.info(f"Running blog pipeline agent for user prompt: {user_prompt}")
    response = await runner.run_debug(user_prompt)
    logger.info("Blog pipeline agent run completed.")


# parallel workflows
tech_researcher = Agent(
    name="TechResearcher",
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    instruction="""Research the latest AI/ML trends. Include 3 key developments,
the main companies involved, and the potential impact. Keep the report very concise (100 words).""",
tools=[google_search],
    output_key="tech_summary",
)
logger.info("Tech researcher agent initialized successfully.")


health_researcher = Agent(
    name="HealthResearcher",
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    instruction="""Research recent medical breakthroughs. Include 3 significant advances,
their practical applications, and estimated timelines. Keep the report concise (100 words).""",
    tools=[google_search],
    output_key="health_summary",
)   
logger.info("Health researcher agent initialized successfully.")


finance_researcher = Agent(
    name="FinanceResearcher",
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    instruction="""Research current fintech trends. Include 3 key trends,
their market implications, and the future outlook. Keep the report concise (100 words).""",
    tools=[google_search],
    output_key="finance_summary",
)

logger.info("Finance researcher agent initialized successfully.")

aggregator_agent = Agent(
    name="AggregatorAgent",
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    instruction="""Combine these three research findings into a single executive summary:

    **Technology Trends:**
    {tech_summary}
    
    **Health Breakthroughs:**
    {health_summary}
    
    **Finance Innovations:**
    {finance_summary}
    
    Your summary should highlight common themes, surprising connections, and the most important key takeaways from all three reports. The final summary should be around 200 words.""",

    output_key="executive_summary",
)
logger.info("Aggregator agent initialized successfully.")

parallel_research_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[tech_researcher, health_researcher, finance_researcher],
   
)

root_agent = SequentialAgent(
    name="ResearchSystem",
    sub_agents=[parallel_research_team, aggregator_agent],
)
logger.info("Sequential research system agent initialized successfully.")

runner = InMemoryRunner(agent=root_agent)
logger.info("Runner for research system agent created.")

async def main3():
    user_prompt = "Run the daily executive briefing on Tech, Health, and Finance"
    logger.info(f"Running research system agent for user prompt: {user_prompt}")
    response = await runner.run_debug(user_prompt)
    logger.info("Research system agent run completed.")


#loopevent workflows

initial_writer = Agent(
    name="InitialWriter",
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    instruction="""Based on the user's prompt, write the first draft of a short story (around 100-150 words).
    Output only the story text, with no introduction or explanation.""",
    output_key="story_draft",
)
logger.info("Initial writer agent initialized successfully.")


critic_agent = Agent(
    name="CriticAgent",
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    instruction="""You are a constructive story critic. Review the story provided below.
    Story: {story_draft}
    
    Evaluate the story's plot, characters, and pacing.
    - If the story is well-written and complete, you MUST respond with the exact phrase: "APPROVED"
    - Otherwise, provide 2-3 specific, actionable suggestions for improvement.""",
    output_key="critique",
)
logger.info("Critic agent initialized successfully.")

def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED', indicating the story is finished and no more changes are needed."""
    return {'status': 'approved', 'message': 'Story approved, exiting loop.'}

refiner_agent = Agent(
    name="RefinerAgent",
    model=Gemini(
        model_name="gemini-2.5-flash-lite",
        api_key=GOOGLE_API_KEY,
        retry_options=retry_settings,
    ),
    instruction="""You are a story refiner. You have a story draft and critique.
    
    Story Draft: {story_draft}
    Critique: {critique}
    
    Your task is to analyze the critique.
    - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function and nothing else.
    - OTHERWISE, rewrite the story draft to fully incorporate the feedback from the critique.""",
    output_key="current_story",

    tools = [
        FunctionTool(exit_loop)
    ]

)
logger.info("Refiner agent initialized successfully.")


story_refinement_loop = LoopAgent(
    name="StoryRefinementLoop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=2,
)

root_agent = SequentialAgent(
    name="StoryWritingSystem",
    sub_agents=[initial_writer, story_refinement_loop],
)
logger.info("Sequential story writing system agent initialized successfully.")

runner = InMemoryRunner(agent=root_agent)
logger.info("Runner for story writing system agent created.")

async def main4():
    user_prompt =  "Write a short story about a lighthouse keeper who discovers a mysterious, glowing map"
    logger.info(f"Running story writing system agent for user prompt: {user_prompt}")
    response = await runner.run_debug(user_prompt)
    logger.info("Story writing system agent run completed.")

if __name__ == "__main__":
    asyncio.run(main4())
