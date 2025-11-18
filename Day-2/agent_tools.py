import dotenv
import os
import asyncio
import logging

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool, ToolContext, google_search
from google.adk.code_executors import BuiltInCodeExecutor

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


#retry config
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# 1st tool
def get_fee_for_payment_method(method:str) -> dict:
    """ Looks up the transaction fee percentage for a given payment method.
    
    This tool simulates looking up a company's internal fee struture based on the 
    name of the payment method provided by the user
    
    Args:
    method: the name of the payment method. It should be descrptive,
    e.g., platinum credit card" or "ank transfer".
    
    Returns:
    Dictionary with status and fee information.
    Success: {"status": "success", "fee_percentage": 0.02}
    error: {"status":'error", "error_message":"payment mehotd not found}
    """

    # this simulates loking up a company;s internal fee structure.
    fee_database = {
        "platinum credit card":0.02,
        "gold credit card":0.025,
        "bank transfer":0.01,

    }

    fee = fee_database.get(method.lower())
    if fee is not None:
        return {"status":"success", "fee_percentage":fee}
    else:
        return {"status":"error", "error_message":"payment method not found"}
    

logger.info("Fee lookup tool defined successfully.")
logger.info(f"Test: {get_fee_for_payment_method('platinum credit card')}")


def get_exchange_rate(base_currency:str, target_currency: str) -> dict:
     """Looks up and returns the exchange rate between two currencies.

    Args:
        base_currency: The ISO 4217 currency code of the currency you
                       are converting from (e.g., "USD").
        target_currency: The ISO 4217 currency code of the currency you
                         are converting to (e.g., "EUR").

    Returns:
        Dictionary with status and rate information.
        Success: {"status": "success", "rate": 0.93}
        Error: {"status": "error", "error_message": "Unsupported currency pair"}
    """

    # Static data simulating a live exchange rate API
    # In production, this would call something like: requests.get("api.exchangerates.com")
     rate_database = {
        "usd": {
            "eur": 0.93,  # Euro
            "jpy": 157.50,  # Japanese Yen
            "inr": 83.58,  # Indian Rupee
        }
    }

    # Input validation and processing
     base = base_currency.lower()
     target = target_currency.lower()

    # Return structured result with status
     rate = rate_database.get(base, {}).get(target)
     if rate is not None:
        return {"status": "success", "rate": rate}
     else:
         return {
            "status": "error",
            "error_message": f"Unsupported currency pair: {base_currency}/{target_currency}",
        }


print("✅ Exchange rate function created")
print(f"💱 Test: {get_exchange_rate('USD', 'EUR')}")


# Currency agent with custom function tools
currency_agent = LlmAgent(
    name="currency_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a smart currency conversion assistant.

    For currency conversion requests:
    1. Use `get_fee_for_payment_method()` to find transaction fees
    2. Use `get_exchange_rate()` to get currency conversion rates
    3. Check the "status" field in each tool's response for errors
    4. Calculate the final amount after fees based on the output from `get_fee_for_payment_method` and `get_exchange_rate` methods and provide a clear breakdown.
    5. First, state the final converted amount.
        Then, explain how you got that result by showing the intermediate amounts. Your explanation must include: the fee percentage and its
        value in the original currency, the amount remaining after the fee, and the exchange rate used for the final conversion.

    If any tool returns status "error", explain the issue to the user clearly.
    """,
    tools=[get_fee_for_payment_method, get_exchange_rate],
)

print("✅ Currency agent created with custom function tools")
print("🔧 Available tools:")
print("  • get_fee_for_payment_method - Looks up company fee structure")
print("  • get_exchange_rate - Gets current exchange rates")


currency_runner = InMemoryRunner(agent=currency_agent)

async def main():
    response = await currency_runner.run_debug(
        "I want to convert 500 US Dollars to Euros using my Platinum Credit Card. How much will I receive?"
    )


# Improving code reliability
# code execution for math multiplication is a better idea
# than relying on the LLM to do the math correctly every time.

calculation_agent = LlmAgent(
    name="caluclationAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction = """You are a specialized calculator that ONLY responds with Python code. You are forbidden from providing any text, explanations, or conversational responses.
 
     Your task is to take a request for a calculation and translate it into a single block of Python code that calculates the answer.
     
     **RULES:**
    1.  Your output MUST be ONLY a Python code block.
    2.  Do NOT write any text before or after the code block.
    3.  The Python code MUST calculate the result.
    4.  The Python code MUST print the final result to stdout.
    5.  You are PROHIBITED from performing the calculation yourself. Your only job is to generate the code that will perform the calculation.
   
    Failure to follow these rules will result in an error.
       """,
    
    code_executor=BuiltInCodeExecutor(),
)

enhanced_currency_agent = LlmAgent(
    name="enhanced_currency_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction = """You are a smart currency conversion assistant. You must strictly follow these steps and use the available tools.

  For any currency conversion request:

   1. Get Transaction Fee: Use the get_fee_for_payment_method() tool to determine the transaction fee.
   2. Get Exchange Rate: Use the get_exchange_rate() tool to get the currency conversion rate.
   3. Error Check: After each tool call, you must check the "status" field in the response. If the status is "error", you must stop and clearly explain the issue to the user.
   4. Calculate Final Amount (CRITICAL): You are strictly prohibited from performing any arithmetic calculations yourself. You must use the calculation_agent tool to generate Python code that calculates the final converted amount. This 
      code will use the fee information from step 1 and the exchange rate from step 2.
   5. Provide Detailed Breakdown: In your summary, you must:
       * State the final converted amount.
       * Explain how the result was calculated, including:
           * The fee percentage and the fee amount in the original currency.
           * The amount remaining after deducting the fee.
           * The exchange rate applied.
    """,
    tools=[
        get_fee_for_payment_method,
        get_exchange_rate,
        AgentTool(agent=calculation_agent),  # Using another agent as a tool!
    ],
)

logger.info("Enhanced currency agent created with calculation agent as a tool.")    
logger.info("New capabilities added to improve calculation reliability.")

enhanced_runner = InMemoryRunner(agent=enhanced_currency_agent)

async def main2():
    response = await enhanced_runner.run_debug(
        "I want to convert 500 US Dollars to Euros using my Platinum Credit Card. How much will I receive?"
    )

if __name__ == "__main__":
    asyncio.run(main())