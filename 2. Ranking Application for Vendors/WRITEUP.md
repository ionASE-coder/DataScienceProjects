This file documents and explains the solution.py

*AI in this document refers to Qwen2.5-7B-Instruct LLM, which was called through API

1. Approach

The entire solution is segmented in the following pipeline:

-reading and preprocessing the JSON dataset -> input_data_transform()

-inputting and analysing the query with the help of AI* and delivering
filters -> query_parser()

-if no filters are identified, adding more context to the query with AI -> query_rephraser()

-applying the identified filters e.g.: companies with more than 500 employees on the dataset -> apply_filters()

-calculating the embedding similarity between the query and selecting the top 5 companies according to this similarity -> embedder()

Example for the filters that are identified from the query and that are applied on the dataset:
Query: "Public companies in Romania founded after 2018 with less than 200 employees."

query_filters={'filter_founded': ['>', 2018],
'filter_revenue': None,
'filter_employees': ['<', 200],
'filter_countries': ['Romania'],
'filter_public': ['true']
}

Assumption:
A mechanism was implemented to always allow some options after filtering. If a filter from the proposed query_filters removes all available companies, it is ignored. The purpose of this approach is to ensure that the user always receives at least one result from the search.

Design:
This design was chosen to maximise the accuracy in detecting the best match, and it is a hybrid approach between LLMs and embedding similarity. This way, LLMs are not invoked for each company (avoiding huge computational stress and lack of determinism) and it avoids relying solely on embeddings (which are known for not being accurate when exact filters are needed, e.g., revenue is more than 50 million USD).

Innovation:
The innovation of this solution is when no filters are identified, e.g.:
query: "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand"
In thi case an LLM is involved to modify the query and bring more context. As a result, it matches more accurately on the dataset. This is done by an LLM, which adds 5 contextual words that summarise the query to broaden the meaning and increase the chances of a match without affecting accuracy. A special query function called query_rephraser() adds 5 more words using AI to the query so it can better match the companies in the list.

The classical approach without the proposed innovation doesn't bring the most relevant results. This is due to the additional reasoning that needs to be done on the initial query. Suggestions for the scalable script would be to convert the query to keywords that can facilitate better embedding similarity. Because this approach may be prone to hallucinations, extra care should be taken. That's why this risk is reserved only for queries that don't return any filters.

Example of the deployment of the innovation:

For the query: "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand", the query becomes:

"Companies that could supply packaging materials for a direct-to-consumer cosmetics brand: suppliers, manufacturers, vendors, producers, distributors"

As a result, when using the new query rephraser for the same query, the 5th company was swapped with a better company. This one has "distribution" as a keyword and increases the cosine similarity, creating chances for a better match overall (by bringing more context).

2. Tradeoffs

I prioritised multiple case handling and managing complex cases such as rephrasing the query. The tradeoffs include risking hallucinations but allowing more exact matching in a production setting.

This may add more costs—tokens for calling AI APIs—and reduces the simplicity of the code. Ultimately, speed is affected negatively by making two LLM API requests per query.

I identified 3 weak points in my proposed system:
-The AI prompts may overfit to queries provided in this example and have a problem understanding new queries.
-When no filters are identified by the query_parser(), the system relies solely on embeddings, which on large datasets may lead to latency.
-Queries which are vague are treated better by the innovation mentioned above of the query_rephraser(), but are still not error-proof.

3. Error Analysis

By running the proposed queries, the following errors were spotted:

Error 1:
Query: "Construction companies in the United States with revenue over $50 million"
This query has the problem that the AI doesn't pick up the revenue. This is because the Qwen agent cannot normalise the symbol ($) and cannot parse "million" to 10^6. To solve this, it is fed special instructions in the original prompt.
For the first 4 ranked companies, the results are very satisfactory, but the 5th result refers to a software company (Ripley Decision Advantage) that is specialised in ML and SaaS. It doesn't have anything to do with construction companies. This is due to triggering words that make the cosine similarity of embeddings go up artificially. This can be solved using a more detailed query, as described in the innovative part of this solution, namely the query_rephraser().

Error 2:
Query: "Renewable energy equipment manufacturers in Scandinavia"
For this prompt, the results are satisfactory, but one downside is that even though the correct filters for the countries Sweden, Norway, and Finland were identified, the first 5 results come from Sweden. This is due to the increased cosine similarity. That is a coincidence for Sweden, but may affect the end result.

Error 3:
Queries that have problems:
"Companies that could supply packaging materials for a direct-to-consumer cosmetics brand"
This query doesn't activate any filter in the AI model, so it relies solely on cosine similarity. This may stress the model on huge datasets because the architecture is designed to embed all rows that were filtered. In this case, because the filter feature won't work, it will embed all 100,000 rows. Apart from this disadvantage, the results are satisfactory.

4. Scaling

Given more time and resources, I would use a better agent than Qwen, opting for paid versions. This would refine the query_parser() and query_rephraser() functions and would allow for better results.

Secondly, I would create a mechanism to avoid embedding all the companies, as it would be very intensive on computational resources.

Thirdly, I would analyse the query with an LLM and alert the user if the query is too vague and would take more time to compute. This may encourage better process optimisation if that is a target of the user.

Lastly, I would normalise the dataset and descriptions of the firms.

5. Failure Modes

My approach has limits in parsing the query correctly. A well-known issue that I am aware of is the embedding similarity score. In production, I would monitor the optimal similarity threshold that should be set in order to deliver robust results. If that value is not met, then the system should not deliver any results.

Another issue in production would be filtering countries. For example, in the
query: "Fast-growing fintech companies competing with traditional banks in Europe.",
my solution would falsely assume that filter_countries should be set to European countries. Here, it would confidently show European fintech startups and leave Asian fintech startups unmatched, which would be an error. In this case, I would monitor the LLM parsing and design a fail-safe system to address such examples.