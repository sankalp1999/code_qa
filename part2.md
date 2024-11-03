# Learnings from codeQA - Part 2

## Quick recap

I recommend reading Part 1 before proceeding further; atleast refer the problem statement.

TODO: Add link

[**Github Link**](https://github.com/sankalp1999/code_qa)

[**Demo Link**](https://x.com/dejavucoder/status/1790712123292840326) # TODO: update video

In the previous post, I covered building a question-answering system for codebases, explaining why GPT-4 can't inherently answer code questions and the limitations of in-context learning. I explored semantic code search with embeddings and the importance of proper codebase chunking. I then detailed syntax-level chunking with abstract syntax trees (ASTs) using tree-sitter, demonstrating how to extract methods, classes, constructors, and cross-codebase references.

## What to expect in Part 2

Key topics covered in this part:

- Final step for codebase indexing - adding LLM based comments
- Considerations for choosing embeddings and vector databases
- Techniques to improve retrieval (HyDE, BM25, re-ranking)
- Choosing re-rankers; In-depth explanation of re-ranking (bi-encoding vs. cross-encoders)

 If you have ever worked with embeddings, you will know that embedding search is not as effective and you need to do lot of pre-processing, re-ranking and post-processing stuff to get good results. In next few sections, we discuss methods to improve our overall search and more implementation details and decisions.

## Adding LLM comments

![codebase indexing](Learnings%20from%20codeQA%20-%20Part%202%20ed70346f75364e0583a9173d7ea7dcf1/shapes_at_24-05-09_17.01.53.png)

codebase indexing

### Initial thoughts: framing codeQA as topK RAG on text

During the brainstorming phase, I thought of framing the codebase question answering problem as a topK RAG but on text instead of code. This would have involved generating LLM based comments for all the methods and classes and embedding the text; just store the code in meta-data. However, this method had drawbacks like it would be more expensive (embeddings are much cheaper than LLM tokens) and indexing would take more time so I didn’t go with it. 

### Implementing LLM comments for methods

Update: I decided to do away with LLM comments in CodeQA v2 because it slows down the indexing process. The relevant file
`llm_comments.py` is still there on repo.

Since our queries will be in natural language, I decided to integrate a natural language component by adding 2-3 line documentation for each method. This creates an annotated codebase, with each LLM-generated comment providing a concise overview. These annotations enhance keyword and semantic search, allowing for more efficient searches based on both ‘what the code does’ and ‘what the code is.’

I found a cool blogpost which validated my thoughts later on. I quote them below. Also  recommend reading the blog (after my post obviously), it’s a short read.

[Three LLM tricks that boosted embeddings search accuracy by 37% — Cosine](https://www.buildt.ai/blog/3llmtricks)

> Meta-characteristic search
> 
> 
> One of the core things we wanted to have is for people to be able to search for characteristics of their code which weren’t directly described in their code, for example a user might want to search for `all generic recursive functions` , which would be very difficult if not impossible to search for through conventional string matching/regex and likely wouldn’t perform at all well through a simple embedding implementation. This could also be applied to non-code spaces too; a user may want to ask a question of an embedded corpus of Charles Dickens asking `find all cases where Oliver Twist is self-reflective` which would not really be possible with a basic embedding implementation.
> 
> **Our solution with Buildt was, for each element/snippet we embed a textual description of the code to get all of the meta characteristics, and we get those characteristics from a fine-tuned LLM. By embedding the textual description of the code along side the code itself it allows you to search both against raw code as well as the characteristics of the code, which is why we say you can ‘search for what your code does, rather than what your code is’. This approach works extremely well and without it questions regarding functionality rarely return accurate results. This approach could easily be ported to prose or any other textual form which could be very exciting for large knowledge bases.**
> 
> There are pitfalls to this approach: it obviously causes a huge amount of extra cost relative to merely embedding the initial corpus, and increases latency when performing the embedding process - so it may not be useful in all cases, but for us it is a worthwhile trade-off as it produces a magical searching experience.
> 

![Untitled](Learnings%20from%20codeQA%20-%20Part%202%20ed70346f75364e0583a9173d7ea7dcf1/Untitled.png)

Phew. Finally done with the codebase indexing process. Now onto next steps - embedding.

---

## Embedding and vectorDB

I use OpenAI's text-embedding-3-large by default as they are very good plus everyone has OpenAI API keys so it's good for demoing. I have also added option for using jina-embeddings-v3 which are better than text-embedding-3-large in benchmarks.

Re: OpenAI embeddings - they are cheap plus have a sequence length of 8191 tokens. Rank 35 on MTEB, multi-lingual and easily accessible via API. Sequence length is important btw because it allows the embedding model to capture long length dependencies and more context.



## Things to consider for embeddings

### Benchmarks

You can checkout the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for embedding ranking with scores. Look at the scores, sequence length, the languages that are supported. Locally available models need your compute so look at size for open source models. 

### Latency

API-based embeddings can be slower than local ones simply because of network round-trips added latency. So if you want
speed and you have the compute, it may be better to use local embeddings.

### Cost

If you want free or just experimentation, go open source `sentence-transformers` . `bge-en-v1.5` or nomic-embed-v1 is all you need. Otherwise most closed source embeddings are cheap only.

### Use-case

They should serve for your use-case. Research accordingly on Hugging Face or website documentation or their research paper. If you are embedding code, the embeddings you use should have had code in pre-training data.

You can use **fine-tuned embeddings** to significantly improve your scores. LlamaIndex has a high-level API to fine-tune all Hugging Face embeddings. Jina recently announced a fine-tuning API too. **However, I was not able to figure out what specific methods people are using to fine-tune embeddings.**

[https://x.com/JinaAI_/status/1785337862755356685](https://x.com/JinaAI_/status/1785337862755356685)

### Privacy

For my project, I prioritized performance and chose OpenAI, not considering privacy concerns. Now I don’t know what happens with my code on their servers haha.

I quote a Sourcegraph blogpost here.

> While embeddings worked for retrieving context, they had some drawbacks for our purposes. Embeddings require all of your code to be represented in the vector space and to do this, we need to send source code to an OpenAI API for embedding. Then, those vectors need to be stored, maintained, and updated. This isn’t ideal for three reasons:
> 
> - Your code has to be sent to a 3rd party (OpenAI) for processing, and not everyone wants their code to be relayed in this way.
> - The process of creating embeddings and keeping them up-to-date introduces complexity that Sourcegraph admins have to manage.
> - As the size of a codebase increases, so does the respective vector database, and searching vector databases for codebases with >100,000 repositories is complex and resource-intensive. This complexity was limiting our ability to build our new multi-repository context feature.

As an aside, for a deeper understanding of codebase indexing methodologies, I recommend reviewing this blog post. Their approach parallels my implementation but operates at a significantly larger scale. They have developed an AI coding assistant named [Cody](https://github.com/sourcegraph/cody). It is worth noting that they have since moved away from using embeddings in their architecture.

[How Cody understands your codebase](https://sourcegraph.com/blog/how-cody-understands-your-codebase)

## VectorDB

I use [LanceDB](https://lancedb.com/) because it’s fast, easy to use - you can just pip install and import, no API key required. They support integration for almost all embeddings (available on Hugging Face) and most major players like OpenAI, Jina etc. There's easy support for integration for rerankers, algorithms, embeddings, third-party libraries for RAG etc.

### Things to consider for VDB

- Support for all the integrations you need - e.g LLMs, different companies,
- Recall and Latency
- Cost
- Familiarity/Ease of use
- Open source / closed source → there’s [FAISS](https://github.com/facebookresearch/faiss) by Meta that’s fast and opensource

### Implementation details

Code for embedding the codebase and making the tables can be found in `create_tables.py` . I maintain two separate tables - one for methods and other for classes and miscellaneous items like README files. Keeping things separate allows me to query separate metadatas plus separate vector searches; get the closest class, get the closest methods.

[https://github.com/sankalp1999/code_qa/blob/main/create_tables.py](https://github.com/sankalp1999/code_qa/blob/main/create_tables.py)

If you see the implementation, you will notice I don’t generate embeddings manually. That part is handled by LanceDB itself. I just add my chunks and they form batch and generate embeddings. Lancedb handles all the retry with backoff stuff.

## Retrieval

![shapes at 24-05-09 21.33.15.png](Learnings%20from%20codeQA%20-%20Part%202%20ed70346f75364e0583a9173d7ea7dcf1/shapes_at_24-05-09_21.33.15.png)

Once we have the tables ready, we can give queries to vectorDB and it will output the most similar documents using brute force search (cosine similarity/dot product). These results are going to be relevant but not as accurate as you think. Embedding search feels like magic until it does not. In my experience with some other projects, the results are relevant but often noisy plus the ordering can be wrong often. I attach a twitter thread below to show some shortcomings.

[https://x.com/eugeneyan/status/1767403777139917133](https://x.com/eugeneyan/status/1767403777139917133)

## Improving embedding search

### BM25

The first thing to do - is usually combine semantic search with a keyword based search like BM25. I used this in [semantweet search](https://github.com/sankalp1999/semantweet-search). These are supported out of box already by many vectorDBs. In semantweet search, I had used LanceDB which supported hybrid search (semantic search + bm25) out of box, I just had to write some additional code to create a full-text based index using tantivy. Some [benchmarks](https://lancedb.github.io/lancedb/hybrid_search/eval/) from lancedb site to demonstrate how different methods can impact results.

<aside>
💡 to measure search performance, recall is the metric.
recall → how many of the relevant documents are we retrieving / number of documents.

</aside>

## filtering using meta-data

Another easy way to improve semantic search → if it’s possible to filter data with stuff like date or certain keywords or metrics, then you could use sql to prefilter those results and then perform the semantic search. Note that embeddings themselves are independent points in latent space. 

## Reranking

Using re-rankers after the vector search is an easy way to improve results significantly. Let’s try to understand how they work on a high level. In CodeQA v2, I use `answerdotai/answerai-colbert-small-v1` as it is the best performing local re-ranker model based on [benchmarks](https://blog.lancedb.com/hybrid-search-and-reranking-report/) with performance close to Cohere Re-ranker 3 (which I used in CodeQA v1).



### Cross-encoding (re-ranking) vs bi-encoding (embeddings based search)

Embeddings are obtained from models like GPT-3 → decoder-only transformer 
or BERT (encoder-only transformer, BERT base is 12 encoders stacked together). 

![Untitled](Learnings%20from%20codeQA%20-%20Part%202%20ed70346f75364e0583a9173d7ea7dcf1/Untitled%201.png)

Both GPT-3 / BERT type of models can be made to operate in two styles → cross-encoder and bi-encoder. They have to be trained (or fine-tuned) for the same. I will not go into the training details here.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # biencoder

# Load the cross-encoder model
cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2") # cross-encoder
```

---

A cross encoder concats the query with each document and computes relevance scores. 

<query> <doc1>
<query><doc2>

Each pair of concatenated query+doc is passed through the pre-trained model (like BERT) going through several layers of attention and mlps. The self-attention mechanism helps to capture the interaction between the query and the doc (all tokens of query interact with document)

We get the output from a  hidden layer (usually the last one) to get contextualised embeddings. These embeddings are pooled to obtain a fixed size representation. These are then passed through a linear layer and then softmax/sigmoid to obtain logits → relevance scores → [0.5, 0.6, 0.8, 0.1, …]

Let’s say we have D documents and Q queries. To calculate relevance score for 1 query, we will have D (query+doc) passes. through the model. For Q queries, we will have **D * Q passes** since ****each concat of D and Q is unique. 

---

A bi-encoder approach (or the **embeddings search approach)** encoding documents and queries separately, and calculates the dot product. Let’s say you have D docs and Q queries. 

Precompute D embeddings. We can reuse the embedding instead of calculating again. Now for each query, compute dot product of D and Q. Dot product can be considered an O(1) operation. 

compute cost of encoding D docs → D
compute cost of encoding Q queries → Q 

compute cost then becomes **D + Q** 

Now this is much faster than cross-encoding approach.

---

**Since every token in query interacts with the documents) and assigns a relevance score of query vs. each document, the cross-encoder is more accurate than bi-encoders** but they are slow since individual processing of each pair is required (each Q+D combination is unique so cannot precompute embeddings)

Thus, we can stick to a bi-encoder approach (generating embeddings, encode query, encode docs and store in vector db, the calculate similarity) for fast retrieval. Then, we can use rerankers (cross-encoders) to get topK results for improving the results.

---

## Reranking demonstration

In the below example, note that `How many people live in New Delhi? No idea.` has the most lexical similarity so cos similarity / bi-encoder approach will say it’s most relevant.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# Load the bi-encoder model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the cross-encoder model
cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Function to calculate cosine similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

query = "How many people live in New Delhi?"
documents = [
    "New Delhi has a population of 33,807,000 registered inhabitants in an area of 42.7 square kilometers.",
    "In 2020, the population of India's capital city surpassed 33,807,000.",
    "How many people live in New Delhi? No idea.", 
    "I visited New Delhi last year; it seemed overcrowded. Lots of people.",
    "New Delhi, the capital of India, is known for its cultural landmarks."
]

# Encode the query and documents using bi-encoder
query_embedding = embedding_model.encode(query)
document_embeddings = embedding_model.encode(documents)

# Calculate cosine similarities
scores = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in document_embeddings]

# Print initial retrieval scores
print("Initial Retrieval Scores (Bi-Encoder):")
for i, score in enumerate(scores):
    print(f"Doc{i+1}: {score:.2f}")

# Combine the query with each document for the cross-encoder
cross_inputs = [[query, doc] for doc in documents]

# Get relevance scores for each document using cross-encoder
cross_scores = cross_model.predict(cross_inputs)

# Print reranked scores
print("\nReranked Scores (Cross-Encoder):")
for i, score in enumerate(cross_scores):
    print(f"Doc{i+1}: {score:.2f}")

Outputs:
❯ python demo.py
Initial Retrieval Scores (Bi-Encoder):
Doc1: 0.77
Doc2: 0.58
Doc3: 0.97
Doc4: 0.75
Doc5: 0.54

Reranked Scores (Cross-Encoder):
Doc1: 9.91
Doc2: 3.74
Doc3: 5.64
Doc4: 1.67
Doc5: -2.20

```

Outputs after better formatting to demonstrate the effectiveness of cross-encoder. 

### Initial Retrieval Scores (Bi-Encoder)

| Document | Sentence | Score |
| --- | --- | --- |
| Doc1 | New Delhi has a population of 33,807,000 registered inhabitants in an area of 42.7 square kilometers. | **0.77** |
| Doc2 | In 2020, the population of India's capital city surpassed 33,807,000. | 0.58 |
| Doc3 | How many people live in New Delhi? No idea. | 0.97 |
| Doc4 | I visited New Delhi last year; it seemed overcrowded. Lots of people. | 0.75 |
| Doc5 | New Delhi, the capital of India, is known for its cultural landmarks. | 0.54 |

The answer to our question was Doc1 but due to lexical similarity, doc3 is has highest score. Now cosine similarity is kinda dumb so we can’t help it.

### Reranked Scores (Cross-Encoder)

| Document | Sentence | Score |
| --- | --- | --- |
| Doc1 | New Delhi has a population of 33,807,000 registered inhabitants in an area of 42.7 square kilometers. | **9.91** |
| Doc2 | In 2020, the population of India's capital city surpassed 33,807,000. | 3.74 |
| Doc3 | How many people live in New Delhi? No idea. | 5.64 |
| Doc4 | I visited New Delhi last year; it seemed overcrowded. Lots of people. | 1.67 |
| Doc5 | New Delhi, the capital of India, is known for its cultural landmarks. | -2.20 |

 

![Screenshot 2024-05-13 at 12.47.20 PM.png](Learnings%20from%20codeQA%20-%20Part%202%20ed70346f75364e0583a9173d7ea7dcf1/Screenshot_2024-05-13_at_12.47.20_PM.png)

References in this section: 

Image used from [Jina AI’s blog on ColBERT](https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/). In, ColBERT they do the “crossing” between the query and document but it is in the later stages of model (to improve latency)

[https://github.com/openai/openai-cookbook/blob/main/examples/Search_reranking_with_cross-encoders.ipynb](https://github.com/openai/openai-cookbook/blob/main/examples/Search_reranking_with_cross-encoders.ipynb)

## HyDE (hypothetical document embeddings)

The user’s query is most likely going to be in English and less of code. But our embeddings are mostly made up of code. Now if you think about the latent space, code would be nearer to code than english (natural language) being nearer to code. This is the idea of [HyDE paper](https://arxiv.org/abs/2212.10496).

![Untitled](Learnings%20from%20codeQA%20-%20Part%202%20ed70346f75364e0583a9173d7ea7dcf1/Untitled%202.png)

You ask an LLM to generate a hypothetical answer to your query and then you use this (hallucinated) query for embedding search. The intuition is the that embedding of hypothetical query is going to be closer in latent/embedding space than your actual natural language query. It's funny how you are  using a hallucinated answer to get better results.

---

### Implementation details

Code used in this section is mainly from `app.py` and `prompts.py`

Using `gpt-4o-mini` for both HyDE queries as it's cheap, fast and decent with code-understanding.


Here's how HyDE query looks like:
```python
# app.py

def openai_hyde(query):
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": HYDE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Help predict the answer to the query: {query}",
            }
        ]
    )
    return chat_completion.choices[0].message.content
```

```python
# prompts.py
HYDE_SYSTEM_PROMPT = '''You are an expert software engineer. Your task is to predict code that answers the given query.

Instructions:
1. Analyze the query carefully.
2. Think through the solution step-by-step.
3. Generate concise, idiomatic code that addresses the query.
4. Include specific method names, class names, and key concepts in your response.
5. If applicable, suggest modern libraries or best practices for the given task.
6. You may guess the language based on the context provided.

Output format: 
- Provide only the improved query or predicted code snippet.
- Do not include any explanatory text outside the code.
- Ensure the response is directly usable for further processing or execution.'''
```

```python
# app.py
def openai_hyde_v2(query, temp_context, hyde_query):
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": HYDE_V2_SYSTEM_PROMPT.format(query=query, temp_context=temp_context)
            },
            {
                "role": "user",
                "content": f"Predict the answer to the query: {hyde_query}",
            }
        ]
    )
    return chat_completion.choices[0].message.content
``` 

The hallucinated query is used to perform an initial embedding search, retrieving the top 5 results from our tables. These results serve as context for a second HyDE query. In the first query, the programming language was not known but with the help of fetched context, the language is most likely known now.

Second HyDE query is more context aware and expands the query with relevant code-specific details.

```python
# prompts.py
HYDE_V2_SYSTEM_PROMPT = '''You are an expert software engineer. Your task is to enhance the original query: {query} using the provided context: {temp_context}.

Instructions:
1. Analyze the query and context thoroughly.
2. Expand the query with relevant code-specific details:
   - For code-related queries: Include precise method names, class names, and key concepts.
   - For general queries: Reference important files like README.md or configuration files.
   - For method-specific queries: Predict potential implementation details and suggest modern, relevant libraries.
3. Incorporate keywords from the context that are most pertinent to answering the query.
4. Add any crucial terminology or best practices that might be relevant.
5. Ensure the enhanced query remains focused and concise while being more descriptive and targeted.
6. You may guess the language based on the context provided.

Output format: Provide only the enhanced query. Do not include any explanatory text or additional commentary.'''
```



By leveraging the LLM's understanding of both code and natural language, it generates an expanded, more contextually-aware query that incorporates relevant code terminology and natural language descriptions. This two-step process helps bridge the semantic gap between the user's natural language query and the codebase's technical content.


```python
def openai_query_for_references(query, context):
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": REFERENCES_SYSTEM_PROMPT.format(query=query, context=context)
            },
            {
                "role": "user",
                "content": f"<query>{query}</query>",
            }
        ]
    )
    return chat_completion.choices[0].message.content
```

```python
# prompts.py
REFERENCES_SYSTEM_PROMPT = '''You are an expert software engineer. Given the <query>{query}</query> and <context>{context}</context>, your task is to enhance the query:

1. Analyze the query and context thoroughly.
2. Frame a concise, improved query using keywords from the context that are most relevant to answering the original query.
3. Include specific code-related details such as method names, class names, and key programming concepts.
4. If applicable, reference important files like README.md or configuration files.
5. Add any crucial programming terminology or best practices that might be relevant.
6. Ensure the enhanced query remains focused while being more descriptive and targeted.

Output format:
<query>Enhanced query here</query>

Provide only the enhanced query within the tags. Do not include any explanatory text or additional commentary.'''
```

A vector search is performed using the second query and topK results are re-ranked using cohere reranker v3 and then relevant meta-data is fetched.

```python
...
    hyde_query = openai_hyde(query)

    method_docs = method_table.search(hyde_query).limit(5).to_pandas()
    class_docs = class_table.search(hyde_query).limit(5).to_pandas()

    temp_context = '\n'.join(method_docs['code'] + '\n'.join(class_docs['source_code']) )

    hyde_query_v2 = openai_query_for_references(query, temp_context)

    logging.info("-query_v2-")
    logging.info(hyde_query_v2)

    method_search = method_table.search(hyde_query_v2)
    class_search = class_table.search(hyde_query_v2)

    if rerank: # if reranking is selected by user from the UI
        method_search = method_search.rerank(reranker)
        class_search = class_search.rerank(reranker)

    method_docs = method_search.limit(5).to_list()
    class_docs = class_search.limit(5).to_list()

    top_3_methods = method_docs[:3]
    methods_combined = "\n\n".join(f"File: {doc['file_path']}\nCode:\n{doc['code']}" for doc in top_3_methods)

    top_3_classes = class_docs[:3]
    classes_combined = "\n\n".join(f"File: {doc['file_path']}\nClass Info:\n{doc['source_code']} References: \n{doc['references']}  \n END OF ROW {i}" for i, doc in enumerate(top_3_classes))

```

## Possible expansions

- One can use the codebase wide references to make a call graph and use it as a repository map. Pass this as context and it may help the LLM to understand the flow of the repository
- Use evaluations (this is something I need to learn, maybe [Ragas](https://docs.ragas.io/en/stable/))

## Conclusion

This concludes this 2 part series of posts. Thanks for reading.


## References

Links as they appear in the post

1. [Learnings from codeQA - Part 1](https://www.notion.so/Learnings-from-codeQA-Part-1-5eb12ceb948040789d0a0aca1ac23329?pvs=21)
2. [Three LLM tricks that boosted embeddings search accuracy by 37% — Cosine](https://www.buildt.ai/blog/3llmtricks)
3. [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
4. [JinaAI's fine-tuning API announcement](https://x.com/JinaAI_/status/1785337862755356685)
5. [How Cody understands your codebase](https://sourcegraph.com/blog/how-cody-understands-your-codebase)
6. [Cody on GitHub](https://github.com/sourcegraph/cody)
7. [LanceDB](https://lancedb.com/)
8. [FAISS by Meta](https://github.com/facebookresearch/faiss)
9. [create_tables.py on GitHub](https://github.com/sankalp1999/code_qa/blob/main/create_tables.py)
10. [Twitter thread on embedding search shortcomings](https://x.com/eugeneyan/status/1767403777139917133)
11. [Semantweet search on GitHub](https://github.com/sankalp1999/semantweet-search)
12. [BM25 benchmarks on LanceDB site](https://lancedb.github.io/lancedb/hybrid_search/eval/)
13. [Cohere reranker v3](https://cohere.com/blog/rerank-3)
14. [ColBERT and late interaction](https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/)
15. [HyDE paper](https://arxiv.org/abs/2212.10496)
16. [Ragas documentation](https://docs.ragas.io/en/stable/)