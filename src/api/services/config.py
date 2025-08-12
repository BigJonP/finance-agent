SYSTEM_PROMPT = """You are a professional financial advisor. Analyze the user's current portfolio and provide thoughtful, personalized investment advice. 

Consider:
1. Portfolio diversification
2. Risk management
3. Market trends and analysis
4. Specific user needs and requests
5. Potential improvements to the portfolio

Provide actionable recommendations that are:
- Specific and practical
- Risk-aware
- Based on current market information
- Tailored to the user's existing holdings

Format your response with:
1. Portfolio Analysis
2. Key Recommendations
3. Risk Assessment
4. Next Steps"""

MODEL_NAME = "gpt-5-nano"

RETRIEVER_K = 2

RELEVANT_DOCUMENTS_TOP_K = 5

MAX_TOKENS = 1500

TEMPERATURE = 0.7


ERROR_ADVICE = """I apologize, but I encountered an error while analyzing your portfolio

Here are some general investment principles to consider:

1. **Diversification**: Spread your investments across different sectors and asset classes
2. **Risk Management**: Only invest what you can afford to lose
3. **Regular Review**: Periodically assess and rebalance your portfolio
4. **Long-term Perspective**: Focus on long-term growth rather than short-term fluctuations
5. **Research**: Always research investments thoroughly before making decisions

Please try again later, or consult with a qualified financial advisor for personalized guidance."""
