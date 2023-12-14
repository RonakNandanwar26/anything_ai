<h3> Possibel Error </h3>
 in case of very big book e.g more thans 200 pages, while doing embedding using OpenAI Embedding, It gives below error
openai.RateLimitError: Error code: 429 - {'error': {'message': 'Request too large for text-embedding-ada-002 in organization 
org-Cn3oodVtjqXfNE0HiKXiqIE4 on tokens per min (TPM): Limit 150000, Requested 159618. The input or output tokens must be reduced in 
order to run successfully. Visit https://platform.openai.com/account/rate-limits to learn more. You can increase your rate limit by 
adding a payment method to your account at https://platform.openai.com/account/billing.', 'type': 'tokens', 'param': None, 
'code': 'rate_limit_exceeded'}}
