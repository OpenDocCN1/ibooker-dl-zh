![](assets/cover.png)

![](assets/cover.png)

# Fine-Tuning AI

by Laurence Moroney

Copyright © 2027 Laurence Moroney. All rights reserved.

Published by O’Reilly Media, Inc., 141 Stony Circle, Suite 195, Santa Rosa, CA 95401.

O’Reilly books may be purchased for educational, business, or sales promotional use. Online editions are also available for most titles ([https://oreilly.com](https://oreilly.com)). For more information, contact our corporate/institutional sales department: 800-998-9938 or *corporate@oreilly.com*.

*   Acquisitions Editor: Nicole Butterfield
*   Development Editor: Michele Cronin
*   Production Editor: Katherine Tozer
*   Interior Designer: David Futato
*   Interior Illustrator: Kate Dullea

*   July 2027: First Edition

# Revision History for the Early Release

*   2026-03-06: First Release

See [https://oreilly.com/catalog/errata.csp?isbn=9798341673373](https://oreilly.com/catalog/errata.csp?isbn=9798341673373) for release details.

The O’Reilly logo is a registered trademark of O’Reilly Media, Inc. *Fine-Tuning AI*, the cover image, and related trade dress are trademarks of O’Reilly Media, Inc.

The views expressed in this work are those of the author and do not represent the publisher’s views. While the publisher and the author have used good faith efforts to ensure that the information and instructions contained in this work are accurate, the publisher and the author disclaim all responsibility for errors or omissions, including without limitation responsibility for damages resulting from the use of or reliance on this work. Use of the information and instructions contained in this work is at your own risk. If any code samples or other technology this work contains or describes is subject to open source licenses or the intellectual property rights of others, it is your responsibility to ensure that your use thereof complies with such licenses and/or rights.

PROD: Uncomment the following sentence if appropriate and add it to the above para: This book is not intended as \[legal/medical/financial; use the appropriate reference\] advice. Please consult a qualified professional if you require \[legal/medical/financial\] advice. 

979-8-341-67333-5

\[LSI\]

# Brief Table of Contents (*Not Yet Final*)

Chapter 1: From Generalist to Specialist (available)

Chapter 2: Model Architectures: The DNA of AI (available)

Chapter 3: The Fine-Tuner’s Workshop (available)

*Chapter 4: Excavating Data* (unavailable)

*Chapter 5: Synthetic Data* (unavailable)

*Chapter 6: Formatting for Thought* (unavailable)

*Chapter 7: The Mechanics of Adaptation* (unavailable)

*Chapter 8: The Training Loop* (unavailable)

*Chapter 9: Alignment* (unavailable)

*Chapter 10: Case Study* (unavailable)

*Chapter 11: Vision Language Models* (unavailable)

*Chapter 12: Controlling the Canvas* (unavailable)

*Chapter 13: Style Injection* (unavailable)

*Chapter 14: Evaluation* (unavailable)

*Chapter 15: Optimization and Quantization* (unavailable)

*Chapter 16: Serving the Expert* (unavailable)

 italicize any chapter titles that are not yet available in ER  

# Chapter 1. From Generalist to Specialist

The honeymoon is over.

LIke me, you’ve probably already built prototypes using Large Language Models (LLMs) as the backend. You’ve felt that initial rush when your complex query, expressed as a prompt, returns a perfect, coherent response. Design that ‘surprises and delights’ is a desirable goal, and the response from the LLM attains that. For a moment, you feel that the future of software is infinite, and that prompting is the new programming language. You just need clever ones.

Since the emergence of LLMs, popularized by ChatGPT, we’ve been in the era of the *generalist.* We’ve been awed by large language models that have been trained on the entire Internet and beyond. They can do everything from writing Haiku poetry to optimizing RUST functions. They are technological marvels.

But what if you want to ship a *real* product? When using these models, you’ll often find that, for various reasons, despite the initial euphoria, they could be the wrong tool for the job. While its ability to rhyme is impressive, you really want it not to hallucinate non-existent tables while querying your legacy SQL database!

And this is what’s emerging as the “API Wall”. It’s that invisible barrier that separates a cool hackathon demo from a profitable business, and that separates successful startups from buzzworthy ones. There are hard realities in overcoming that wall: the potentially crushing cost of scale, unpredictable latency, and, of course, the terrified look on your CISO’s face when you mention that you’ll be sharing customer data with a third-party.

This book is ultimately about breaking through that wall. It’s about moving from *renting* intelligence to *owning* it. It will have you replace the massive generalists with agile, hyper-competent specialists: Fine-Tuned AI that knows your business better than any trillion-parameter giant ever could.

You’re going to dismantle the ‘prompt engineering’ paradigm. In this chapter, we’ll look at the brutal ‘Unit Economics’ that has killed many AI startups. And we’ll see why a small, open-source model running on a single GPU can—and should—beat the world’s smartest AI in *your* specific domain.

# The Cost of Renting Intelligence

Every project starts innocently. You get an API key and start coding. You craft a system prompt honed from lots of experimentation. It works. You deploy to beta.

And then you look at the costs. Your eyes go wide. You think about what it would look like. They go oven wider!

## The Token Tax

In the prototyping phase, inference costs feel negligible. A fraction of a cent per request is basically free. But when you move from “chatting” to automation, the math turns against you.

Most providers charge by the token (roughly 0.75 words). For a high-end reasoning model, you might pay $0.01 to $0.03 per 1,000 input tokens.

Let’s play that out with a scenario. Consider a ‘Legal Contract Analyzer’ app. Your users upload documents, and your app will extract summaries of potential issues or risks in the contract.

The architecture might look like [Figure 1-1](#ch01.html_ch01_figure_1_1772730610555180).

![A blue rectangle with a blue gear and white text AI-generated content may be incorrect.](assets/ch01_figure_1_1772730610555180.png)

###### Figure 1-1. A Simple LLM-based architecture for a Legal Contract Analyzer

Your input is your prompt plus the contract. A typical contract—with lots of legalese—can be about 15,000 tokens. Your system prompt is something you’ve carefully crafted with lots of instructions to avoid hallucinations, and to structure your output exactly the way the rest of the system needs it—maybe as JSON or other structured files. This takes tokens. And of course your output is yet more tokens.

So 17,000 tokens in, and 1,000 tokens out. Given the costs outlined above, that could be 20 cents per document. If you have 1,000 users each analysing 5 documents per day, that’s $1000 per day, or $365,000 per year.

That’s the money you have to spend just to operate this system. And you own *nothing* at the end of it. You’re *renting* intelligence. And if you charge your users a monthly fee—say $20 per month, they become a net loss very quickly. You have very risky margins.

## The Agentic Multiplier

Now consider the impact of Agents. They’ve been the darling of the tech industry since about mid 2024, and for good reason—they’re an excellent design pattern to get the most out of LLMs by moving beyond just getting an answer for a question. I usually like to think of agents as following the pattern shown in [Figure 1-2](#ch01.html_ch01_figure_2_1772730610555218).

![](assets/ch01_figure_2_1772730610555218.png)

###### Figure 1-2. An Agentic Workflow

Ultimately, the goal here is to follow these steps:

1.  *Understand the user’s intent*. For example, if the agent is designed to help with restaurant selection, and the user says they would like to eat *okonomiyaki*, the agent will be smart enough to understand that this is a branch of Japanese cuisine. The user doesn’t need to specify this.
    
2.  *Make a plan* *using available tools.* If the agent knows that its tools are Google Search, Map Search, and Browse websites, it can make a plan to search for Japanese restaurants near the current location. For each of these, it can then browse their web sites to find their menus. It can then search the menus for *okonomiyaki*.
    
3.  *Execute the plan using the tools.* It then follows the plan, using LLMs as appropriate—for example, generating code to call the Map Search APIs, or parsing downloaded HTML to find the desired dish
    
4.  *Reflect on the results.* After it’s done all this, it should then match its results with the user’s requirements. Did it find restaurants serving this dish? Or just generic Japanese ones. Stuff like that. If it did, great, it can move onto step 5—the solution. Otherwise, it’ll have to go back to steps 1 and 2.
    
5.  *The Solution*. Present the solution to the user, and possibly flow to other agents, such as one that can handle making a reservation at the desired restaurant!
    

This workflow, with a dedicated chain of thought, is the simplest DNA of an agent. But when it comes to token usage, you can see that it can use significantly more tokens than a simple prompt->result like you saw in the earlier section. That $1000 a day that we speculated about earlier will more likely be close to $5000 a day when using an agentic workflow like this.

No wonder companies are investing billions of dollars in data centers!

### Another hidden cost: The instruction tax

There’s also a more subtle inefficiency here. The Generalist model doesn’t understand anything about your specific task until you tell it. To make it useful, you have to paste your rules, formatting guides, and how to handle edge cases into the System Prompt on *every single request*!

“You are an expert at understanding a variety of dishes and where to find them. Understand the user’s request, and map it to one or more popular cuisines. If there isn’t a direct match, find other restaurant types that might serve this dish. Output your results as JSON structured in this way {...}”

You’re paying to send this instruction manual, and all of its tokens, to the server over and over again. It’s extremely wasteful.

And, Context Window Pollution is something to consider. Research has shown that as you stuff the context window with instructions, the model’s performance will degrade (the ‘Lost in the Middle’ phenomenon). I’ve seen so many examples where strict instructions (usually in ALL CAPS for emphasis) have to be entered to get the model to pay attention to a particular part of the prompt. And this, of course, consumes more tokens.

Fine-tuning moves these instructions from the Context Window (which you pay for) into the Model Weights (which are permanent). The model doesn’t need to be told how to behave, it’s already baked in.

# The Latency Killer

Costs are a business problem. Latency is a user experience problem. And if your users have a poor experience, they won’t pay the costs!

Consider a massive Foundation Model with hundreds of billions of parameters. The physics of inference on this model is inescapable. Even with state of the art GPUs, moving terabytes of weights through memory takes time and power. An important metric: Time to First Token (TTFT) can sit anywhere from 500ms to several seconds. The next metric, Tokens Per Second (TPS), then measures how frequently tokens appear after the first one.

Think of this as you build a chatbot. If it takes 500ms to produce the first token (TTFT) and then 50 tokens per second (TPS) your user experience will be fine. Your tolerance to higher TTFT and less TPS is probably okay—because you are used to people thinking before they respond.

But what about other scenarios? For example coding with intelligent autocomplete. Developers typically need suggestions *quickly*. If the TTFT is very high, they can probably just type it quicker themselves. Or with *voice* applications? If there’s a multi-second lag, then conversations might feel broken. As such, depending on your application type, the quality of response is important.

And moving tokens through a behemoth is slower due to the size of the model. Faster, better, machines, with dedicated hardware are at best a patch for this problem.

And, of course, if that machine is shared with millions of other customers, you face the possibility of different TTFT and TPS, because your query just has to get in line alongside others.

It’s not a good look if your Enterprise customer’s dashboard is slow, because your system is competing with high schoolers using ChatGPT to write their essays!

Thus, if you consider an on-premise model that’s fine tuned to the specific need, you’ll see that it’s generally smaller—so it doesn’t need the dedicated high cost hardware to manage synching the model across many machines. There’s less latency—no round tripping to the cloud or getting in line behind other models. You can scale your hardware investments to your needs. And, of course, you can tune the model yourself to optimize it for your particular workload.

I’ve seen applications that were sluggish and brittle on a hosted model, but they become instant and snappy when moved to a local model. The problem then shifts from mitigating failures due to others hosting the model, towards adding business value by fine tuning the local one. Ultimately this adds to the value of your business, instead of flowing operating expenses to a third party.

# The Data Firewall

If cost and latency weren’t enough to make you rethink, the Enterprise sales cycle, and necessary compliance when working with them should!

True story: I once sat in on a potential investment on a very hot startup that was pitching AI to solve a common problem in hospitals. It used computer vision with vision language models. I can’t go into the specifics, of course, but ultimately it would save hospitals a lot of time and money, as well as helping them improve quality of care. But, being vision-based, it involved putting cameras in common places, including the hospital ward.

Given the requirements for *flexibility*, where one day they might want to be able to spot a particular issue, but another day they might want to see something different, the idea of vision language models (VLM) was appealing. One can train a non-transformer based model for computer vision, but then it becomes static. It’s good at spotting X or Y, but if you then want to see Z, you need a whole new model. With a VLM, it’s just a prompt. “What’s in this picture?” or “Tell me if you see violation (X) in this image” etc.

The startup demonstrated how they would do this. The demo was amazing. It did everything that the customers thought hospitals would need. There was a lot of excitement. Then, someone asked about the architecture. The VLM was a hosted giant model.

“Wait,” the CISO asked. “You’re sending images to <Company Name>?”

The CEO of the startup nodded slowly. “Yes,” she replied. She then held up documents showing how <Company Name> manages privacy, guarantees anonymity, and everything else to ensure that their system had integrity.

But that wasn’t enough. They would have to go to the hospital with a ‘trust me bro’ attitude around a very large third party, who themselves weren’t at the table.

Naturally, despite the optimism, the deal died at that table. Hospitals aren’t nearly as secure as they might be as a result.

And that’s a good thing.

At least until now. Now there’s an alternative path forward. I hope that the startup has moved into the space of local VLMs so they can get to the best of both worlds. If you’re them, and you’re reading this, please reach out and let me know!

## The Metadata Leak

The above story is an example of where the *data* that belongs to the customer was a showstopper when designing a product that shares it with a third party.

But it’s not just the data or the prompt content that can be a risk. It’s also the metadata. Do note that an API provider, even if they don’t look at or leak your data, still knows who you are, when you send requests, how big the requests are, and what you might be working on.

In high-stakes businesses like financial investment, a leak about the things you are researching can tip the hands of your competition. For example, in businesses that are considering mergers and acquisition (M & A), if it leaked that Company A is spending a lot of time looking at the fundamentals of Company B, it could be inferred that they’re considering acquiring them. Knowledge of this could lead third parties to purchase shares in B and drive up the eventual acquisition price. This can easily become a new form of ‘insider trading.

## Compliance Issues

Frameworks like GDPR can also make this much stickier. If a user demands their data to be deleted, you can remove it from your infrastructure—but what if you used it to prompt a third party model? How can you guarantee that it’s deleted from their servers?

When you own the model, you have full provenance. You can show exactly what data you used to fine tune. You can keep logs of all data that’s passed into your model for inference—and delete as necessary. You can retrain from a clean dataset if you need to.

Ultimately—you have control.

## Model Drift

I’ve personally been very surprised by how many times I’ve spoken with companies that had a ‘It worked yesterday, it fails today’ problem. They didn’t understand why. And the explanation was simple: API providers update their models constantly. So, all the effort that goes into prompt engineering—in particular injecting commands into the context window—can evaporate overnight. The only solution—to re-engineer the prompts and hope the vendor doesn’t introduce breaking changes to *your* prompt-- isn’t exactly the most inspiring!

But, of course, owning your own fine-tuned model creates that static artifact that you control. It won’t change unless you want it to. And if you do change it, you still hold the older versions, so you can introduce regression and gradually update your apps that use it!

# The Specialist Revolution

For a long time Scaling Laws were the vectors that pointed to a better future. Bigger—more data, more layers, more transformers, more parameters was always better. The direction of innovation followed this.

Once models reached a certain size, it became infeasible to host them yourself, and you had to depend on the giants with the billion dollar data centers to host them for you, and you pay by the token.

But that’s no longer strictly true. We are well into the transition towards a bifurcation with the era of LLMs being small enough to self-host, licenses that allow for this, and open pre-trained models being freely shared. With the ability to further fine tune those models on your downstream tasks, and your particular data, you now have a viable alternative.

In early 2023, the release of LLaMA started this ball rolling. And, it proved that well-engineered models, despite their much smaller size, could punch way above their weight. It also started massive innovation in research where new optimizations drove better quality of model, instead of just throwing scale at the problem.

And that’s the spirit of this book. The Generalist is still alive and well and able to do amazing work with its vast knowledge.

But maybe it’s time for more specialists in our lives. And that’s what I think is a very bright future for developers like you: building specialists.

# Hardware is no longer the blocker

I once spoke at a conference in Wales. The speaker before me was a medical researcher using AI to detect brain disease. Over a beer later, I asked him what his biggest bottleneck was. I expected him to say data.

“Compute,” he said. “We have ten researchers and only enough budget for one GPU. My slot is Tuesday afternoon. The rest of the week, I just have to theorize.”

That was only a few years ago. Today, the laptop I’m writing this on—a standard MacBook Pro—can fine-tune a model that is orders of magnitude more powerful than what he was using.

The shift to *Unified Memory* (in Apple Silicon and newer chips) means we don’t need massive industrial server farms to build AI. We can load 70B parameter models into system RAM. We can build and test locally. The AI R&D lab now fits in a backpack.

# The Spectrum of Customization

There are many ways that you can customize your model, which fall into three broad areas:

Using prompts
: 

Within the prompt you can ‘teach’ the model how to do things by sharing examples. This can include more advanced stuff such as soft-prompting

Using RAG
: 

Retrieval Augmented Generation is a way of embedding ‘new’ knowledge by having an automated system add related content to your prompt to give your model extra context

Fine Tuning
: 

Here’s where you will burn new patterns into the model’s brain so it does them without extra prompting. It’s the hardest to do, but the most valuable in the long run.

I’m not going to cover the first two in this book, as they’ve extensively been covered, and ultimately don’t change the model—they are just smarter ways of prompting.

Instead, this book is for the hard part—the fine tuning of models—we won’t just talk theory. We’re going to calculate VRAM requirements, scrape and clean messy data, and debug training loops that refuse to converge.

The API wall is very high, but now, more than ever there’s an alternative. And I do believe the high-value skill of tomorrow lies with the developers that can overcome that, and build an AI that you own completely.

In [Chapter 2](#ch02.html_ch02_model_architectures_the_dna_of_ai_1772730611468542), you’ll begin that journey by looking at a few of the more popular models and how they’re built. That will hopefully help you choose the best one for your task!

# Chapter 2. Model Architectures: The DNA of AI

In [Chapter 1](#ch01.html_ch01_from_generalist_to_specialist_1772730610561549), we explored that the era of ‘one giant model fits all’ is evolving, and why fine tuning smaller, more specialised models represents a strong future of practical AI deployment. The opportunity is in front of you to expand your skills to be a model customizer. But before you can do this, you need to understand *what* you are customizing. To use an analogy—a mechanic who doesn’t understand how an engine works is able to change the oil, but they cannot rebuild the transmission. Similarly, if you’re a fine tuning practitioner—you need to grasp the fundamentals of how transformer models work. That way you won’t be limited to surface-level adjustments, and you’ll be able to make the deep, strategic decisions that separate a mediocre fine-tuned model from an exceptional one.

So, in this chapter, we’re going to do an anatomy lesson. The goal is to dissect the architecture that powers every major language model today. You’ll examine the key innovations that differentiate the leading open source model families, and understand how they convert raw text into the mathematical representations that make ‘understanding’ possible. It will be a very high-level look, as one could write several whole books on how transformers work under-the-hood, so we’ll just be hitting the highlights.

By the end, you’ll have the foundation you need to make informed decisions about which base model to choose for your specific fine-tuning project.

# The Flow of Information: How transformers work

Before going into the details, it’s good to get a high-level mental model for how a transformer-based language model deals with text. It would take several books to go into this in detail, so I’m just going to look into the highlights.

You can see this in Figure 2-1.

![A diagram of a block diagram AI-generated content may be incorrect.](assets/ch02_figure_1_1772730611456644.png)

###### Figure 2-1. High Level Transformers Architecture

The key insight is that text enters the model as a sequence of discrete symbols. Usually these are character encodings like ASCII or Unicode for ‘T’, ‘h’, and ‘e’. Those character encodings are grouped into words by separating based on punctuation. The words are then converted into tokens using a tokenizer. These tokens are mere numerical representations of words or subwords.

There’s no inherent ‘meaning’ or ‘positional’ ideas encoded within them. That’s where the second box—the embeddings—comes in. An embedding is a vector that represents the meaning of the word. Think of an embedding as an arrow in a higher-dimensional space. The direction it points in represents the meaning. Words that are similar—like ‘King’ or ‘Prince’ would point in roughly the same direction.

In a transformer, positional encoding is also added to the vector. This positional encoding represents the location of the word in the expression, on the theory that word location can impact its semantics. A particular example of this is the use of adjectives. For example, ‘the big dog ate meat,’—we can see that the adjective ‘big’ directly applies to the dog, but it can also directly apply to the ‘meat’ indirectly—as it’s more likely that a big dog will eat a lot of meat!

All this happens *before* the tokens flow into the transformer blocks, and here is where the innovation that led to the current explosion in AI technology lies.

## Attention Mechanisms

This is the heart of the transformer. I like to think of them using this analogy: imagine you are reading a document with a highlighter in your hand. As you read each paragraph, you might highlight certain words or phrases because they’ll help you understand the current sentence. Maybe it’s a piece of fiction, and the character is eating a certain food that you’ve never heard of. You look back and you see that the events take place in a particular location. This can help you understand, or at least make an educated guess, about the piece of food.

The attention mechanism artificially does the same type of process. For each position in the sequence, it will calculate how much ‘attention’ should be paid to every other position.

Let’s consider this sentence: “The cat sat on the mat because it was tired.”

When processing the word ‘it,’ you would need to refer to what ‘it’ refers to. As a human, reading that passage, you know it refers to the cat. But when you think about it a moment, it could also refer to the mat. And with strict grammatical rules, and not semantics, one might argue that it does!

The attention mechanism calculates attention scores for the word ‘it’, to see what it refers to. Ultimately, based on lots of training data, it likely makes a stronger connection between the word ‘it’ and ‘cat’, instead of ‘it’ and ‘mat’!

This works through an elegant mathematical operation. It does this through calculation of three vectors:

*   The Query (Q): “What am I looking for?”
    
*   The Key (K): “What do I contain?”
    
*   The Value (V): “What information do I provide?”
    

They are calculated by multiplying the embedding vector for the word (as discussed earlier in this chapter) with weight matrices that will be learned over time.

You can see this in [Figure 2-2](#ch02.html_ch02_figure_2_1772730611456682).

![A white paper with black lines AI-generated content may be incorrect.](assets/ch02_figure_2_1772730611456682.png)

###### Figure 2-2. Understanding the attention mechanism

In this case, we explore the input tokens, and focus on the token for ‘it’, which is at position 8.

Then, we look at the Keys for each of the words, and examine their values. They have a score, and we multiply the Q for ‘it’ by the K for each of these words. We’ll take the results and softmax them to get probabilities to have a resulting set of attention weights.

These weights are the probabilities of attention that should be applied to the word. So, for the word ‘it’, the highest probability word that is closest associated with it is ‘cat’. To calculate the output, and the final attention, we then multiply out the weights by the values.

The beauty of this mechanism is that it is entirely *learned*. You might wonder where those values in the diagram come from, and without a good set of K values to begin with, all this will be nonsense. Realize the vast amounts of text that models are trained on—and from these, over time, with trial and error, and the classic ‘Machine Learning’ paradigm, the useful patterns get learned.

These ‘chunks’ of Q, K, V to learn attention on text embedding vectors are often referred to ‘heads’, and when multiple heads are used, we’ve learned that some will end up tracking semantic relationships, some grammatical, and some will capture relationship patterns for words that we don’t even have names for.

Typically, when there are multiple layers of heads, then the earlier heads in the sequence will learn the more obvious connections like syntax or grammar, and the later heads will capture the more complex, semantic relationships.

## Multi Head Attention

A single attention operation captures one type of relationship type between the tokens. Language is rich with multiple, simultaneous relationship types. And different languages can have different relation types that don’t exist in English. To capture this richness, the idea of Mullti-Head attention, which runs multiple attention operations in parallel, each with their own learned Q, K , V matrices is useful.

You can see this in [Figure 2-3](#ch02.html_ch02_figure_3_1772730611456705).

![A diagram of a multi-head attention AI-generated content may be incorrect.](assets/ch02_figure_3_1772730611456705.png)

###### Figure 2-3. Multi Head Attention

Models like Llama 3 8B, for example, will use 32 attention heads per layer, across 32 layers.

Each layer’s heads will operate at different levels of abstraction—early layers tend to learn syntactic patterns, while later ones get into semantics.

### Reducing dimensions in MHA

These layers will reduce dimensions per head in order to save computation costs.

The *hidden* dimension is 4,096—the size of the vector representing each token as it flows through the network. Rather than running the attention operation on all 4,096 dimensions, the model will split this into the 32 parallel heads, where each will work on 128 dimensions (4096 / 32 = 128).

Thus, each head will compute its own attention scores and produce output in 128 dimensions. These 32 outputs will then be concatenated back together and passed through one more learned weight matrix—often called the ‘output projection’. This serves an important purpose: it allows the model to mix information across heads. Without it, the heads would be completely independent (i.e. head 1 has the first 128 dimensions, head 2 has the next 128 and so on). With this output projection, the model learns how to combine the different perspectives from different heads into a unified representation for the next layer.

This keeps the total computation roughly similar to a single large attention operation, but it gives the model much more flexibility to learn diverse patterns.

###### Note

It’s tempting to imagine that different heads neatly divide labor, and visualizations like [Figure 2-3](#ch02.html_ch02_figure_3_1772730611456705) tend to represent this because it’s easier for us to grasp things like ‘Head 1 does grammar, Head 2 does semantics etc’, but reality is much messier. Research into transformer interpretability has shown that specialization tends to happen more *across* layers than *within* them. Earlier layers—closer to the input—tend to capture local syntactic patterns: which words are adjacent, basic grammatical relationships, positional information etc. Deeper layers progressively build more abstract, semantic relationships: what the sentence means, how concepts relate, what an appropriate response to a question might be, and stuff like that!

Within any single layer, multiple heads do learn different patterns, but in reality they do not map to human-interpretable categories. The power of multi head attention comes from this redundancy and diversity—the model figures out what’s useful for its corpus while training, and not through an engineered direction by a human.

### Self-Attention vs Cross-Attention

There are two main types of attention in transformer architectures you should consider.

What we’ve been discussing so far is *self-attention*, where a sequence attends to itself. The queries, keys, values, all come from the same sentence. In our hypothetical scenario of reading a fiction book, and we have a paragraph with a location mentioned and a food type mentioned, our attention is *only* on that paragraph.

*Cross-attention* extends on this, and considers where one sequence can attend to a different one. In the hypothetical example, we might have learned in an earlier chapter that the character is vegetarian, which would give us a clue about what the unknown food might be. In ML terms—ultimately what’s happening here is that the Q values could come from one sequence, while the K and V come from another.

For *most* attention models, including all the ones you’re likely to fine-tune, self-attention is the dominant mechanism. Cross attention appears in encoder-decoder type architectures, and in multi-modal models, where the model might need to attend to image features as well as text.

## The Feed Forward Network

Once the attention mechanism has done its thing, and allowed tokens to gather context from other ones, each position will pass through a feed forward network (FFN). This is a simple, but powerful two-layer neural network that is applied independently to each position.

You can see it in [Figure 2-4](#ch02.html_ch02_figure_4_1772730611456723).

![A diagram of a algorithm AI-generated content may be incorrect.](assets/ch02_figure_4_1772730611456723.png)

###### Figure 2-4. The Feed Forward Network

Its purpose is ultimately to store the ‘knowledge’ of the model. Think of the attention mechanism as gathering *context,* where after attention is applied, each token’s representation is enriched with information for other tokens that it attended to. The FFN will then process this contextualized representation, extract useful features, and transform it for the next layer.

This expand-then-contract structure is ultimately the key. The first layer will expand the dimensionality of the embedding from 4,096 to 14,336 dimensions. This larger space allows the network to represent more complex patterns. It does this with that number of neurons, and each could be thought of as a ‘pattern detector’. When the input matches a pattern that the neuron has learned to recognize, it will activate strongly.

Once you’ve heightened the dimension then the *non linear* activation function, such as SiLU or SwiGLU, is applied. I’m emphasising non linear, because if you think about a linear transformation—it’s just a matrix multiplication that scales or rotates vectors, but doesn’t handle curves or thresholds. SO, if you stack two linear layers—you’re effectively multiplying by Matrix A and then again by B, but that’s mathematically equivalent to multiplying by a single matrix BA. No matter how many you stack, they still effectively collapse into one—you get nothing from the additional depth.

But if you use a non linear activation function, the neurons get more selective. Imagine one of them, where its weights in the first layer compute a high value when the input looks like ‘question about cuisine,’ and a low or negative one otherwise. The activation function, if non linear, will be curved—thus strengthening the activation for cuisine questions, and staying near zero for everything else. Without the non-linearity, the neuron would respond proportionally for *everything*, and wouldn’t learn to trigger on specific relevant patterns, and ignore the others.

The second linear layer then projects this back down to 4,096 dimensions. Each neuron in layer 1 now has a corresponding one in layer 2. Often this is considered to be the ‘knowledge’ associated with the input pattern, and what the model should output when that pattern is detected. This is why researchers often describe FFN layers as functioning like key-value *memory*. The first layer’s weights are the keys that match input values, and the second layer’s are values that get retrieved when a key matches.

For example, a neuron on the first layer might activate when the input represents ‘Food from Nagasaki’, and the corresponding output one might push the representation towards the tokens for ‘Okonomiyaki’. This is an oversimplified abstraction—the real ones don’t map nearly as closely to human-understandable concepts as this, but it does capture how the basic mechanism of how factual knowledge from the text corpus gets encoded.

For fine tuning this really matters—the FFN layers capture a significant portion of the model’s deep knowledge. In techniques like LoRA (which we’ll explore in Chapter 7), we can inject knowledge to this layer. Ultimately, if you can build the mental model of the attention layers focusing on ‘what to look at’, while the FFN handles ‘what to do with it’, you’ll have a huge headstart in understanding how and where to apply custom adaptation.

## Layer Normalization and Residual Connections

Two additional components that are crucial but often overlooked, but you may see in any discussion of the transformer are *Layer Normalization* and *Residual Connections.* These enable stable training of deep networks:

*Residual connections* (also called skip connections) add the input of each sub-layer to its output. This allows gradients to flow more easily during training and lets the model learn to make incremental refinements at each layer rather than complete transformations. They also help prevent gradients exploding (growing too large as to be meaningless) or vanishing (approaching zero, so they lose information)

*Layer normalization* normalizes the activations across the feature dimension, stabilizing the distribution of values as they pass through the network. Noting that the ultimate value of a vector is relatively meaningless, and that its distributions for probabilities can be more important, the idea of normalization means that values are generally distributed between 0 and 1, avoiding clustering of values and loss of information. Modern models like Llama use “RMSNorm” (Root Mean Square Normalization), which is computationally simpler.

# Decoder-Only Dominance

The original transformer paper described an *encoder-decoder* architecture designed for machine translation. The encoder processes the input sequence (source language), and the decoder generates the output sequence (target language), attending to the encoder’s representations via cross-attention.

![A diagram of a process AI-generated content may be incorrect.](assets/ch02_figure_5_1772730611456741.png)

###### Figure 2-5. Encoder-Decoder Transformer

For *language translation* this was essential, because when translating one language to another, one needs to look ahead and behind in a sentence to be able to translate it effectively.

But for *generation* purposes, one cannot pay attention to words that are ‘ahead’, because they don’t exist yet. If you have a sequence of tokens like ‘The cat sat’, then you can only look back for attention. This process is called ‘causal’ attention, where each position will only attend to previous positions in the sequence. This is achieved using a mask on the tokens to prevent lookahead, as shown in [Figure 2-6](#ch02.html_ch02_figure_6_1772730611456757).

![A sheet of paper with text AI-generated content may be incorrect.](assets/ch02_figure_6_1772730611456757.png)

###### Figure 2-6. Decoder and Causal Attention

As such, an encoder isn’t needed with its bidirectional self attention, and only the decoder is necessary.

The decoder-only architecture became very popular with open source models like Llama or Mistral.

It’s relatively simple! It has one architecture, and one training objective (next token prediction), which makes it easier to scale. It also has unified input and output. Ultimately, the same architecture handles both “understanding” the prompt and “generating” the response. There’s no artificial boundary between input and output.

Researchers discovered that decoder-only models, when scaled sufficiently, develop remarkable capabilities, including in-context learning, chain-of-thought reasoning, and instruction following, without being explicitly trained for these tasks.

For training and fine-tuning, every position in the sequence provides a training signal (predicting the next token), making training data highly efficient.

When you fine-tune these models, you will most likely work with this decoder-only architecture. The training process adjusts the model’s weights so that given your specific prompts and contexts, it can predict the next token in a way that matches your desired outputs.

# Grouped Query Attention (GQA)

To understand Grouped Query Attention (GQA), it’s good to look at the early history of Llama.

Open, tunable, models are pretty much synonymous with this model. While the term ‘open source’ is typically used to describe them, I generally resist this because we don’t have the source, or the data. What we do have is the pre-trained weights of the model under various open source licenses so we can modify it, and papers describing the architecture.

The first version of this was released in February 2023 with a relatively standard transformers implementation. Its overall significance was to demonstrate that smaller, well-trained models were in the game–at least when comparing with larger models using standard benchmarks. It should be noted that many benchmarks were invented with larger models in mind, so the idea of benchmarks will need to evolve as more smaller models come online. Similarly, while the performance against benchmarks might have been equivalent, the context window size was the biggest notable difference between smaller and larger models. Llama had a context window of only 2,048 tokens.

A few months later, version 2 was introduced that doubled the context window size (to 4,096 tokens), as well as a new innovation called ‘Grouped Query Attention (GQA), which was a crucial optimization for performance. You’ll also see that it’s the constraints of going smaller that has driven innovation, and GQA is an excellent example of that.

A little over a year after the first release, LLama 3 doubled the context window size again—to 8,192 tokens in the small model, as well as a family of larger ones with up to 128k tokens. It also improved the vocabulary, with 128k tokens, a 4x improvement, as well as updates to their implementation of GQA.

So let’s explore what GQA is, and why it’s so important.

Grouped Query Attention is one of the most important optimizations in modern LLMs, and understanding it will help you make better decisions about which model to choose in solving your problem.

In standard multi-head attention, as you saw earlier in this chapter, each head has its own Query, Key, and Value projections. During inference, the Key and Value tensors for all previous tokens must be cached (this is the “KV cache”) to avoid recomputing them at each generation step.

For long sequences, this cache becomes a significant memory bottleneck.

You can see this in [Figure 2-7](#ch02.html_ch02_figure_7_1772730611456773).

![A diagram of a diagram AI-generated content may be incorrect.](assets/ch02_figure_7_1772730611456773.png)

###### Figure 2-7. Multi-Head Attention and KV Cache Size

The idea of GQA is to strike a balance here by grouping Query heads, with each group sharing K and V projections. You can see this in [Figure 2-8](#ch02.html_ch02_figure_8_1772730611456788).

![A diagram of a number of questions AI-generated content may be incorrect.](assets/ch02_figure_8_1772730611456788.png)

###### Figure 2-8. Grouped Query Attention

In this simple example, where there are 6 heads, instead of each Q having its own dedicated KV, we have a situation where Q1 and Q2 share a KV, Q3 and Q4 share one etc. This gives us a 50% reduction in KV Cache size.

In LLama 3 8B, there are 32 query heads grouped into 8 groups of 4, and each shares 8 key-value heads.

This reduces the KV cache by 75% compared to standard multi-head attention, dramatically improving inference speed and reducing memory requirements—especially for long contexts.

So, when it comes to fine tuning, it’s good to consider models that use GQA because they can give you better inference performance without sacrificing much quality.

# Continued Innovation: Mistral and Mixtral

Continuing the story of innovation being driven by smaller models, and the needs for efficiency that can be generally forgotten in larger models, let’s now explore Mistral and Mixtral.

Mistral 7B, introduced in September 2023 demonstrated that a relatively small model—only 7B—could compete with much larger models on benchmarks. It did with several key innovations.

## Sliding Window Attention

As we pay attention to previous tokens, the longer the sequence of tokens, the more the scale of being able to manage it explodes in size. So, if every token has to look at every other token, including itself, we end up with a quadratic number of lookups. For 2 tokens, there would be 2 comparisons of 2 tokens each, so 4 lookups. With 10 tokens, this becomes 100. With 1,000 tokens we now have 1 Million lookups! How large would this get for a context window of 128k tokens?

Mistral introduced a sliding window, where each token only directly attends to the previous 4,096 tokens. Your initial thoughts on this might be that information outside the context window might be lost, but think about it a little.

If the context is 128k tokens, and you are using a sliding window of only 4k, it may feel like the majority of tokens are ignored.

But in a multi-layer model, this isn’t necessarily true. Let’s explore this.

![A diagram of a diagram AI-generated content may be incorrect.](assets/ch02_figure_9_1772730611456804.png)

###### Figure 2-9. Sliding Window Context movement.

Imagine 10 tokens, with a 4 token context window. So, for example for tokens 7 through 10, the output of our sliding window will be the ‘rightmost’ token in Layer 2. Similarly, the window of tokens 4 through 7 will have an output at an equivalent position for token 7.

You can see that 5 through 8 will output at position 8, and 6 through 9 will output at position 9, though these aren’t drawn in [Figure 2-9](#ch02.html_ch02_figure_9_1772730611456804).

Then, when you get to layer 3, the token at position 10 is actually receiving context from all the way at position 4 because of the ‘pyramid’ effect with the context being fed through the layers.

The receptive field grows linearly with depth, and the result is that the amount of processing required for tokens loses the quadratic explosion of processing. Amazingly enough, the model performed equivalently to many larger models that didn’t innovate on managing the processing of the attention! They simply brute forced it!

## Sparse Mixture of Experts

While Llama 4 may have been the famous model to broadly use Mixture of Experts, Mixtral introduced a transformative version of this—a Sparse Mixture of Experts. It’s a fundamentally different approach to scaling.

In a standard transformer, every token passes through every parameter of the model as shown in [Figure 2-10](#ch02.html_ch02_figure_10_1772730611456819).

![A diagram of a single person AI-generated content may be incorrect.](assets/ch02_figure_10_1772730611456819.png)

###### Figure 2-10. Feed Forward in a Standard transformer

In a Mixture of Experts model, the feed-forward network (FFN) layer is replaced by multiple “expert” FFNs, and a routing/gating network decides which experts to use for each token. You can see this in [Figure 2-11](#ch02.html_ch02_figure_11_1772730611456834).

![A diagram of a algorithm AI-generated content may be incorrect.](assets/ch02_figure_11_1772730611456834.png)

###### Figure 2-11. Sparse Mixture of Experts

In this case, the input is passed through the attention layers, and the router/gate chooses which feed forward network to use. So, for example if the token is ‘code’, the top 2 experts that are appropriate may be one that’s dedicated to coding, and another that’s dedicated to math. The data is routed to these FFNs, and a weighted combination forms the output.

This is what makes the Mixtral model a ‘Sparse’ Mixture of experts. Instead of passing the tokens through *every* expert, and then weighting the output, it simply selects 2 and then weights the output.

With a model like Mixtral 8x7B, the name may suggest it’s a 7B model, but it isn’t—the 8x7 refers to 8 experts, each of which are close to 7B parameters!

For each token, the router will select the top 2 experts as shown.

So, for this model, the total parameters are 46.7B—used to store all experts. But for any given inference, only 2 experts are used, so 12.9B parameters are used. So, you get the capacity of a much larger model, while paying the computational cost of a smaller one at inference time.

When it comes to fine-tuning, MoE models are tricker to deal with. You might want to only fine tune selected experts, or perhaps the routing model. The training dynamics are far more complex, and you need much more VRAM to hold all the parameters even if the inference is efficient. For many fine tuning projects, a non MoE may be an easier approach!

# Gemma: Google’s Open Approach

Google entered the open-weights space with a variety of models over the years, but the most recent, at time of writing, is called *Gemma*. Based on the same research that powers Google giant hosted Gemini models, Gemma comes in a variety of sizes. The most recent version at time of writing, Gemma 3, comes in variations from 270M parameters up to 27B.

One key characteristic of Gemma is that many of the variations of Gemma are *instruction tuned* from the start. This means that they are already aligned to follow instructions, making them ready for many tasks out of the box. When you inspect the list of Gemma models on a hub like huggingface ([Gemma 3 release](https://huggingface.co/collections/google/gemma-3-release)) you’ll see all of them listed, and for Gemma 3 it looks like [Figure 2-12](#ch02.html_ch02_figure_12_1772730611456848).

![A screenshot of a computer AI-generated content may be incorrect.](assets/ch02_figure_12_1772730611456848.png)

###### Figure 2-12. Gemma on HuggingFace

You’ll see some models with the prefix ‘-it’ for ‘Instruction Tuned’, and ‘-pt’ for ‘Pre Trained’. The latter hasn’t had additional alignment to make it useful for chat scenarios, and instead is just raw tokens in and tokens out. This is pretty powerful for you to be able to create your own instruction tuning, without needing to work around Google’s work here.

Given the variety of sizes, there are slightly different architectures with Gemma 3 when it comes to managing the KV cache.

The *Standard* size lineup (4B, 12B, and 27B sizes) all use Grouped Query Attention (GQA) as outlined above. This is the secret to how small models like these can support a *128K Context Window* without the memory requirements for the KV cache exploding.

The *Smaller* size lineup (1B and 270M) are optimized for on-device speed. They use a new attention type called *Multi Query Attention* (MQA) which is a very aggressive GQA where *all* attention heads share a single key/value head. This makes it very fast, but it loses nuance. It does keep the model small enough to perform on basic hardware.

###### Note

Gemma 3 has a unique twist on GQA with something called an *attention ratio*. The goal here is to mix the use of the sliding window. 5 out of every 6 layers use a sliding window of 1024 tokens as we discussed earlier, but 1 out of every 6 will look at the full 128k context!

# Tokenizers and Embeddings

In the discussion on architectures, it’s also important to think about different methods for tokenization (conversion of the human-language words into numbers that represent them,) and embeddings (giving a high dimensional vector representation for a token to try to capture context), and methodologies to use these efficiently.

## Tokenization

When we look in detail at tokenization, and define it as the process of breaking text into discrete units that the model can process, we can see that it will be more complex that simply splitting on words.

Some reasons for this may be situations like:

*   Rare words would require enormous vocabularies. Is it worth having a token dedicated to the word ‘antidisestablishmentarianism’?
    
*   Non-English Languages may handle spacing between words differently
    
*   Typos, new words, and code would be very difficult to handle
    
*   Variants of a word (‘run’,’ running’, ‘runs’) would be completely separate from each other, potentially losing their semantic similarity.
    

To overcome this, most modern LLMs use *subword* tokenization, where the vocabulary consists of common words, or pieces of them. It gives a balance—common words like ‘the’ will be single tokens, which is efficient, while more rare words will be broken down into multiple tokens like ‘anti’, ‘dis’, ‘esta’, ‘blish’ etc.

The tokenizer choice matters because different models use different tokenizers, trained on different data! The tokenizer will have a vocabulary of tokens that it works with, and this affects efficiency. Models like Llama 2 have 32,000 tokens, and LLama 3 has over 128,000. A larger vocab means common words and phrases are more likely to be single tokens—which will reduce sequence length, giving you more information within the context window. But it also means bigger embedding matrices, giving you a tradeoff.

### Understanding Language and Tokenization

If you are working with models for code or special characters, the tokenizers will vary widely. Many are trained primarily on *English* text, and won’t be optimized for code. Others will have programming languages in their training data—and most notably, the tokenizers used will be ‘code friendly’ in their vocab.

Additionally, when it comes to multilingual coverage, a tokenizer that is trained on English will likely fragment non-English text into many smaller tokens.

Consider the phrase ‘你好世界’ which is ‘Hello World’ in Chinese. If this uses a tokenizer trained on English text, it will encode this into 4 separate tokens, representing the Unicode characters for the Chinese text \[你\]\[好\]\[世\]\[界\]

A multi-lingual tokenizer that understands Chinese characters, and how they are grouped into words would be more efficient, and paid it appropriately into the ‘words’ for ‘Hello’ and ‘World’ \[你好\]\[世界\]

This, as you can see, will greatly affect both efficiency and quality. More tokens means longer sequences that are slower and carry less context, and the performance can be negatively impacted.

###### Warning

Ultimately you need to consider the tokenizer when fine tuning. You cannot change a model’s tokenizer without retraining from scratch. When selecting a base model, check that its tokenizer handles your domain’s text efficiently. If you are fine tuning for a specific programming language or a non-English language, you have to verify that the tokenizer doesn’t produce excessively long sequences.

## Byte Pair Encoding (BPE)

The most common algorithm to perform tokenization is byte pair encoding, a methodology that was originally developed for data compression!

It works like this:

*   Start with a vocabulary of individual characters, each represented by a byte
    
*   Count all adjacent pairs in the training corpus
    
*   Merge the most frequent pair into a new token
    
*   Repeat the process until the vocab reaches the desired size
    

For example, imagine a corpus of only 3 words ‘low’, ‘lower’, and ‘lowest’.

We would start with the full vocab of individual characters `[l]`, `[o]`, `[w]`, `[e]`, `[r]`, `[s]`, `[t]`, `[]` where I am representing a space with the `[*]`.

Then, we would look at the tokenization of each word:

```
[l], [o], [w], [*], [l], [o], [w], [e], [r], [*], [l], [o], [w], [e], [s], [t]
```

From this we can see that `[l]` followed by `[o]` is the most frequent pairing, so we re-encode to have an `[lo]` character instead of `[l]` and `[o]`.

```
[lo], [w], [*], [lo], [w], [e], [r], [*], [lo], [w], [e], [s], [t]
```

Now we can see that \[lo\] followed by \[w\] is the most common, so we re-encode to:

```
[low], [*], [low], [e], [r], [*], [low], [e], [s], [t]
```

…and we would continue until we have the desired number of tokens.

## Embeddings

Once text is tokenized, each token ID is mapped to an embedding vector. This is a representation of the token in a higher dimensional space. Let’s explore what that means. Let’s use Llama 3 8B, where the embedding dimension is 4,096. In other words—the vector representing the semantics for a token is a direction in a space of 4,096 dimensions.

While this is hard to visualize, consider that the goal here is to encode semantic relationships. Tokens with similar meanings tend to have similar embeddings after training. Similar embeddings are close together in vector space, where their vectors are ‘pointing’ in similar directions.

A good way to understand this is to consider it in only 2 dimensions. For example, suppose we were to look at characters from the Jane Austen novel *Pride and Prejudice*, considering the dimensions of gender and nobility. We could plot the former on the x-axis and the latter on the y-axis, with the length of the vector denoting each character’s wealth (Figure 2-13).

 
| ![Characters in Pride and Prejudice as vectors](assets/ch02_figure_13_1772730611456863.png)

###### Figure 2-13. Characters in Pride and Prejudice as vectors

 |
| --- |

From an inspection of the graph, you can derive a fair amount of information about each character. Three of them are male. Mr. Darcy is extremely wealthy, but his nobility isn’t clear (he’s called “Mister,” unlike the less wealthy but apparently more noble Sir William Lucas). The other “Mister,” Mr. Bennet, is clearly not nobility and is struggling financially. Elizabeth Bennet, his daughter, is similar to him, but female. Lady Catherine, the other female character in our example, is nobility and incredibly wealthy. The romance between Mr. Darcy and Elizabeth causes tension—*prejudice* coming from the noble side of the vectors toward the less-noble.

As this example shows, by considering multiple dimensions we can begin to see real meaning in the words (here, character names). Again, we’re not talking about concrete definitions, but more a *relative* meaning based on the axes and the relation between the vector for one word and the other vectors.

### Understanding Positional Embeddings with RoPE

Transformers process all tokens in parallel, so they have no inherent sense of order. To inject positional information, we add positional embeddings to the token embeddings.

Modern models like Llama use Rotary Position Embeddings (RoPE), which encode position through an innovative technique involving rotation in the embedding space. The key advantage of RoPE is that it allows the model to generalize to longer sequences than it saw during training while minimizing quality degradation.

Consider the sentence “The cat ate the mouse”. Without positional information, the model would find the sentences “The cat ate the mouse” and “The mouse ate the cat” to be *mathematically* identical.

Rotary Positional Embedding (RoPE) solves this by treating word embeddings not just as static points in space, but as vectors that can be *rotated*.

To encode position, instead of adding a fixed value to the embedding representing the position, RoPE does it by rotating the embedding vector. So, for each step forward in a sequence, the idea is to rotate the vector for that word by a specific angle. If we call that angle ‘theta’ θ, then the first embedding is rotated by θ, the second by 2θ, the third by 3θ and so on. As we move further down the sequence, the embedding vector ‘spins’ further away from the origin in a predictable, calculable way.

The idea now is that as well as the semantics of the words being encoded in the embedding, as we saw in the previous section, the *position* of the word can also be encoded, so when the model calculates the attention scores it can keep both in mind. So, recalling that the embeddings are vectors, and the relative *meanings* between words (like we saw with characters in Pride and Prejudice) can be encoded in the relationships between those vectors, we can evaluate this in mathematics using a dot product. When words are semantically similar, their dot product is high because their vectors are similar. Otherwise the dot product is low. Models use this to pay attention based on relevance.

But without positional encoding, the model may think that the word “cat” at the beginning of a book is the exact same ‘cat’ that shows up at the end of the book. It can see the meaning of a cat, but is blind to the order in which the cat appears.

Using RoPE, you don’t destroy this semantic: you just twist it by rotating it through an angle. Now, words that are closer together in a sentence have similar rotations, and as such a similarity of *position* is encoded into the vector also. It’s complex math that gives a simple solution.

# Other Considerations

Beyond the architecture, there are several other model capabilities you should keep in mind. We’ll discuss them in this section.

## Instruction Following

Some models are better at following complex, multi-step instructions. This emerges from instruction-tuning during pretraining. This means that the training data for the model was explicitly done as question-answer pairs to look more like real world usage. This makes the model better at generalizing for conversations where the user typically asks a question, and the model answers. Models labeled “Instruct” or “Chat” have been specifically trained for this. We saw this when looking at Gemma, when the models are helpfully named with an '-it’ at the end when instruction-tuned. Do note that if your task requires precise instruction following, starting from an instruction-tuned base is usually better than starting from a base model.

## Reasoning Depth

Often reasoning is just the model ‘talking to itself’. It looks at the user’s query, then looks at its answer, and then uses that to try to drive a better answer. Larger models generally show better multi-step reasoning, but architectural choices, particularly in smaller models also matter greatly. MoE models like Mixtral can route different reasoning steps through different experts, potentially improving complex reasoning, and giving you a generalized overview of the solution. While this is usually a ‘plus', it’s not always necessarily the case. For example, a single step through the code expert for a code solution, may work better than multiple steps through other experts! For tasks requiring deep reasoning (mathematical proofs, complex analysis), you may need a larger model or accept multiple inference calls with chain-of-thought prompting where, instead of relying on the model to internally reason, you can just prompt it again (and again, and again) to force it to reason. This can be very difficult to fine tune, so manage your expectations accordingly!

## Context Windows and the ‘Lost in the Middle’ problem

Just because a model has a large context window doesn’t mean it uses long contexts effectively. “Lost in the middle” is a known phenomenon where models attend strongly to the beginning and end of the context but miss information in the middle. This can be particularly dangerous in the case of a sliding window to drive longer contexts. When fine tuning in particular, you should test the base model’s ability to recall information from various positions in the document to see if it has this problem, before committing to the significant investment of fine tuning it!

# Summary

As you can see, deciding which model to use relies on a variety of factors. Personally, I like to use the following questions, in no particular order of priority.

1.  What tokenizer does the model use? Will it encode *my* data accurately and meaningfully?
    
2.  How large is the context window? How large is it in reality, and how large is it because of neat tricks in the KV cache? Does the context window meet the expected demands of my users?
    
3.  How big is the model? Will it fit into the available memory I have in order to host it where I want? Can it be quantized? What will the impact of quantizing be?
    
4.  If I am retraining, what are the memory constraints of my training system? Will it work with the number of parameters, the context window size, and the memory available to me?
    

Also note the following findings:

1.  When it comes to Attention Type, models that use GQA generally offer high inference efficiency, but MQA offers much better efficiency as a cost of quality
    
2.  Dense models like Llama and Mistral are usually easier to fine-tune. MoE models offer more capacity but are more difficult to change.
    
3.  Tokenizer efficiency can have a huge impact on the time it takes to retrain or fine tune a model
    
4.  The Conext window size should accommodate your longest training examples and expected inference inputs.
    
5.  Typically Base Models are better to give you the control you want over the final behavior, but if your scenario is an instruction-based one (like chat,) you might be better off using an instruction tuned variant of the model.
    
6.  Hardware. Remember that *training* takes significantly more resources than inference. Make sure that your model isn’t just suited to your end users inference needs, but also that you can think about the training needs and costs, which can be significant!
    

The number of models available to you is growing all the time. But I think these questions give you a great starting point!

Understanding these architectural foundations will serve you throughout your fine-tuning journey. In the next chapter, we’ll translate this theoretical knowledge into practical hardware and software setup—the fine-tuner’s workshop where you’ll start building your expert models.

# Chapter 3. The Fine-Tuner’s Workshop

Have you ever tried to build furniture with the wrong tools? Thankfully the Ikea’s of the world include the necessary allen keys and screwdrivers. But before they started providing those, you usually, at best, ended up with a wobbly desk. And that’s *after* fighting with your equipment before you actually built anything.

Fine tuning a model is no different. The right hardware and software setup is everything, and the perfect investment to prevent frustrating long nights full of cryptic error messages.

In this chapter, we’ll talk about that workbench. The search space for the correct hardware is near infinite, so I’ll be opinionated about what I think is a good set to begin with! You don’t have to go with my suggestions, of course, but I *will* be using them through all the examples in this book.

# Why Setup Matters

When you run inference on a model, feed a prompt and get a response, you’re doing relatively lightweight work. The model’s weights sit in memory, data flows through the network and causes a bunch of calculations. The tensors get collapsed into output tokens that get decoded into words. A decent laptop can handle inference on a 7B sized model.

But fine tuning is fundamentally different. You’re not just running data through the model—you’re updating the parameters within the model itself. So, every training step requires you not just to store the weights and biases, but also the gradients—the direction each weight should move to get your desired goals. Additionally the optimizer keeps states, which is momentum and variance history that is tracked for every parameter. Some optimizers like Adam may have double the storage than the weights and biases.

A rough calculation is that a 7B parameter with 16-bit precision takes about 14Gb of memory. But there’s much more to it than just the parameters—Add the gradients, optimizer states, activations, etc and you’re pushing 70-80Gb, which is way beyond consumer GPUs at reasonable price points, likely setting you back five figures for a workstation. You’ll see that calculation in more detail later in this chapter.

If you’ve tried doing anything with fine tuning models you’ve probably seen a ‘Cuda Out of Memory’ message. It worked yesterday when you did inference, but now you’re asking a lot more of that memory!

So, if you get nothing else from this chapter, please realize that measuring the size of the model is just the first step in understanding your needs for tuning that model.

After memory, the next important thing to consider is your software environment. For fine tuning, the ecosystem is constantly evolving, but consider it to sit at the intersection of PyTorch, CUDA, Hugging Face, and quantization tools. Each has their own version requirements and interdependencies. For example, version mismatching between CUDA and PyTorch will often give you a silent failure—your code works, but it doesn’t take advantage of the GPU exploding the time it takes to do anything! A mismatch between bitsandbytes and your CUDA version can give you a load failure or a compilation error that sends you down a Stack Overflow rabbit hole or a very long conversation with your LLM of choice.

###### Note

*Vibe coding* is awesome, but it is prone to issues like this. Training sets used to train models to understand and generate code don’t have errors like this—they’re based on reference implementations that just work. So the code the LLM spits out might be factually correct, but they still have problems because your environment isn’t correctly set up for it. And when you ask the LLM to fix the problem, it will likely try to fix the code, but the code wasn’t broken to begin with! The bottom line: You have to understand your environment, and you have to ensure that it works!

The good news: These problems aren’t insurmountable. When you know what to look for, you’ll know how to solve them. This chapter will give you the map to that hidden treasure.

One thing to keep in mind is that there are many options for the environment, not least the GPU—AMD and ROCm, or Apple and Metal are valid competitors to NVidia and CUDA. For the most part, in this book, I’ll follow the most common path (NVidia and CUDA).

# The GPU: Your First Decision

Modern language models are typically built on transformer, or transformer-derivative architecture– these function primarily using matrix multiplications. A single forward pass through a model will involve many thousands of large matrix operations. We saw in [Chapter 2](#ch02.html_ch02_model_architectures_the_dna_of_ai_1772730611468542) how we multiply weight matrices against input vectors, computing attention scores across sequences of tokens, and projecting values through feed forward layers.

A CPU is designed for sequential logic–branching decisions, routing control flow, and threading output. Generally it has a handful of very powerful cores, each of which can do one thing very very fast.

A GPU, on the other hand, can have many thousands of smaller cores. For example, an NVidia A100 has almost 7,000 of them! And they are designed to do the same operation on many pieces of data simultaneously. Despite being designed for graphics (that’s the ‘G’ in GPU!), this also perfectly suits AI’s hunger for matrix multiplication. The architecture of the GPU allows you to do thousands of them *in parallel*.

The result is that AI workloads can be orders of magnitude faster on a GPU than on a CPU. So they’re not optional. Don’t worry if you don’t own them—many cloud platforms offer access to GPUs by-the-hour, or tools like Google’s Colab will give you GPU access for free. We’ll explore those later.

## VRAM: One Number to Rule Them All

Given that GPUs aren’t optional, we now have to explore the memory that’s available to the GPU, and that’s VRAM, or Video RAM. Why Video? Well—it’s another holdover from video games and GPUs. In our case it has nothing to do with video or games, but the important point is that it’s the memory that is *directly* accessible to the GPU. Many computers have a motherboard and a CPU with RAM, and then a card with the GPU on it as a separate plug-in device. As the GPU needs to work with numbers, those numbers have to be in memory—and transferring to and from the system RAM through the card slot slows things down. The obvious solution—give the GPU Card its own dedicated memory. And that’s VRAM.

It is a hard constraint that determines what you can and cannot fine-tune effectively. If the combined footprint of your model, gradients, optimizer states, activations and other overhead exceeds the available VRAM, training will at best be very slow (because of all the memory swapping across the card interface and the motherboard), or at worst crash completely. My experience has been that if it doesn’t fit in VRAM, don’t bother—you’ll be setting yourself up for a world of pain.

A drawback: VRAM is very expensive. It’s rare to find GPUs with more than 32Gb of VRAM without breaking several banks. The solution is *distributed* training, where multiple GPUs work in tandem *sharing* their VRAM. 2 GPUs with 32Gb each are a lot cheaper than 1 GPU with 64Gb.

Let’s now get a little deeper into VRAM requirements and take a look at how it can and will impact your decisions.

### Calculating VRAM requirements for a 7B model

To fine tune a 7B model, we need to calculate how much VRAM we will use during training, and this will start with the *model weights*. As there are 7 billion of them, they’ll have an immediate impact. The first thing we need to figure out is how much they store based on their precision. You’ll see several methods of precision in technical documentation, but the most common four are:

*   FP32: (32-bit Floating point) - 4 bytes per parameter, so 28B Bytes
    
*   FP16: (16-bit Floating point) - 2 bytes per parameter, so 14B Bytes
    
*   INT8: (8-bit quantized) - 1 byte per parameter, so 7B Bytes
    
*   INT4: (4-bit quantized) - 0.5 bytes per parameter, so 3.5B Bytes
    

There are other areas in research looking at different levels of precisions—FP8 is becoming popular, for example, as are much smaller ones like 2-bit encoding. As you can probably guess, the lower the memory, the lower the precision, but often we have discovered for tasks like language models, you really don’t need 32-bit precision. Think about it, when predicting the next token, you don’t gain anything from being 99.0000234023523% accurate as opposed to 99.00002% accurate!

So ultimately there’s a tradeoff, but fine tuners will typically begin with FP16, costing 14B bytes off the bat. You can experiment with INT8 and INT4, which I would definitely recommend at least to begin with, if you’re hitting memory constraints.

Next up is the *gradients*. While this book won’t go into depth on the Machine Learning process, an important part of it is called *backpropagation*, where, in order to adjust parameters, and by definition ‘learn’, the algorithm has a gradient that it calculates for the *direction* in which it should adjust a parameter. In simplistic terms, that means ‘up’ or ‘down’ and by ‘how much’. Each gradient is the *same size* as its associated parameter. The calculations above hold true, effectively doubling the amount of memory. So at 16 bit precision, you need 14B Bytes for the parameters, and another 14B Bytes for the gradients.

###### Note

A fine-tuning technique called *LoRA* which we will explore in Chapter X doesn’t use *all* the parameters, just a very small subset of them. So for example, if it only uses 1% of the parameters, that’s 0.14B for the parameters and another 0.14B for the gradients. That’s very little, and possible to do on a GPU with only 8Gb of memory! As a result, you can see why it’s very popular!

A necessary part of Machine Learning and fine-tuning is to use an *Optimizer*. A very common one in fine-tuning is called Adam. The goal of the optimizer is to compute the gradients we mentioned above—determining direction and quantity that parameters need to move in in order to become more effective at their job (aka ‘learning’). The Adam optimizer needs *two* additional values per parameter—a running mean, and a running variance. These need to be *highly* accurate, so they are typically stored in FP32 regardless of how the weights are encoded. As a result, they take up 8 bytes per trainable parameter. So, for a full fine tune of a 7B model, that’s 56Gb extra just for the optimizer! Again, if using LoRA, this might be 1% of that, or 0.56GB, much more palatable!

One part of Machine Learning that is often forgotten about storage-wise is *Activations*,** where the output of each layer is stored, so it can be used in the backward pass to compute the gradients. The memory here depends on the model architecture for things like batch size, sequence length etc. For a 7B model with a batch size of 1, and a sequence of 2048 tokens, the activations might consume 2-4Gb. Again it depends very much on the model architecture, but it’s important to note that it will scale linearly with batch size, so that will double for a batch size of 2 and so on. So when it comes to fine tuning, the memory requirements might constrain your batch sizes. Less memory, and you can only do the smallest batch (1), more memory and you can do more batches. Another reason why VRAM is important—because the more data you can move in a single pass by batching it, the faster your training or retraining will move!

So let’s add these numbers up.

For full fine tuning at FP16 (Table 3-1):

| Component || Memory (approximate) |
| --- | --- |
| Model Weights (FP16) || 14Gb |
| --- | --- |
| Gradients (FP16) || 14Gb |
| Optimizer States (FP32) || 56Gb |
| Activations || 2Gb |
| Total || 84Gb |

That’s more than any single consumer GPU can hold. Even the superpowered 80GB A100, the superstar workhorse for machine learning would struggle without aggressive optimization. At time of writing, these GPUs cost ~ $14,000 each

If, instead of full fine-tuning, you were to do a LoRA at FP16, it would look like Table 3-2:

| Component || Memory (approximate) |
| --- | --- |
| Model Weights (FP16) || 14Gb |
| --- | --- |
| LoRA Adapter Weights || 0.14Gb |
| Gradients (FP16) || 0.14Gb |
| Optimizer States (FP32) || 0.56Gb |
| Activations || 2Gb |
| Total || 17Gb |

Now we’re in the territory of a consumer grade GPU, like an RTX 4090 with 24Gb of VRAM, which costs about 1/10th of the A100.

Finally, if we look at QLoRA (we’ll explore it in detail in Chapter X), we can see that it gets even smaller (Table 3-3):

| Component || Memory (approximate) |
| --- | --- |
| Model Weights (4-Bit Quantized) || 3.5Gb |
| --- | --- |
| LoRA Adapter Weights || 0.14Gb |
| Gradients (FP16) || 0.14Gb |
| Optimizer States (FP32) || 0.56Gb |
| Activations || 2Gb |
| Total || 7Gb |

This will fit in only 8Gb of VRAM. GPUs with 8Gb of will typically cost 1/10th of a 24GB Variant like the RTX 4090 mentioned above. I’ve seen them in stores for $100-$200. Much more manageable, and also available on the free tier of Google Colab—with the T4 GPU that has 16Gb VRAM.

This was just for a 7B model. Use Table 3-4 to explore what you would need for other common sizes:

| Size || Full FT (FP16) || LoRA (FP16) || QLoRA(4-bit) |
| --- | --- | --- | --- |
| 3B || ~36Gb (A100 40Gb) || ~8 Gb (RTX 3060) || ~4Gb (RTX 3060) |
| --- | --- | --- | --- |
| 7B || ~86Gb (A100 80Gb) || ~17Gb (RTX 4090) || ~7Gb (T4/ RTX 4070) |
| 13B || ~160Gb (multi GPU) || ~ 32Gb (A100 40Gb) || ~12Gb (RTX 4080) |
| 70B || ~800Gb (multi node) || ~160Gb (multi GPU) || ~40Gb (A100 40Gb) |

Here’s a comparison of each of these (Table 3-5):

| Component || FP16 Full Tune || LoRA (1% Rank) || QLoRA (4-bit) |
| --- | --- | --- | --- |
| Model weights || 14 GB || 14 GB || 3.5 GB |
| --- | --- | --- | --- |
| Gradients || 14 GB || 0.14 GB || 0.14 GB |
| Optimizer states || 56 GB || 0.56 GB || 0.56 GB |
| Total (approx) || 84+ GB || ~17 GB || ~7 GB |

The takeaway: to get started on fine tuning without taking on excessive GPU expense, work with LoRA or QLora. It’s a sweet spot, it’s highly effective, and it’s accessible on consumer hardware!

## Understanding GPUs

In the previous section, I noted a lot about GPUs with names like RTX, A100, and so on. If you aren’t familiar with them, don’t worry! But let’s take a quick look at the variety of GPUs that are out there, and which ones to consider.

### Consumer GPUs

I’ll define consumer grade GPUs as ones that are readily available to purchase in retail, and which use standard card interfaces in PCs to use. The flagship of these at time of writing is the NVIDIA RTX 4090 with 24Gb of VRAM. Their MSRP is about $1400, but they can often be found on secondary markets selling for much more because of rarity! With one of these LoRA or QLoRA on 7B models is a snap. And while it’s a lot of money, consider that it’s more powerful than the data center dedicated GPUs of just a few years ago. Failing that, the previous generation RTX 3090, with 24Gb is also perfectly serviceable, and usually can be found at a significant discount. Ultimately, the amount of VRAM is the better indicator for you of performance. While 40x0 class machines have better performance metrics, if they don’t have 24Gb, you’ll be limited.

AMD also provide excellent devices that are unlocked by the ROCm platform. The Radeon RX 7900 XTX has 24Gb VRam and competitive specs to the RTX 4090 generally available at a steep discount from its NVidia counterpart. However, the fine tuning ecosystem for AMD-based chips isn’t as mature as for NVidia. Technologies like Flash Attention, bitsandbytes, and parts of the Hugging Face stack were built to be NVidia first. This puts you at a significant disadvantage.

If you’re on a Mac with *Apple Silicon* (M-series chips), you have an interesting option that’s worth understanding. Apple’s unified memory architecture means the GPU and CPU share the same RAM, so the need for dedicated VRAM on other systems goes away. This is ultimately because the CPU and the GPU are the same chip—as opposed to the GPU being on a card that’s plugged into the motherboard.

When a framework reports available “VRAM,” it’s reporting your system RAM—and that can be substantial. An M4 Max with 128 GB of unified memory technically has more “VRAM” than an A100.

The catch is throughput. Apple’s GPU cores are very capable but not as numerous or as fast for matrix operations as NVIDIA’s data center GPUs. A training run that takes 1 hour on an A100 might take 4-6 hours on an M4 Max. The memory is there, but the compute takes longer to chew through it. And, similar to working with any other non-Nvidia chip, the ecosystem that has grown around NVidia is a significant moat, so you may have compatibility issues.

### Data center GPUs

The NVIDIA A100 is the industry standard for many AI workloads, including fine-tuning.

It’s not often clear when using it in hosted environments like Google Colab, but it comes in 40 GB and 80 GB variants, with the 80 GB version being the more common offering on cloud platforms, but you should check before use if you need more than 40GB before you pay for it!

An A100 can handle LoRA on 70B models and full fine-tuning on models up to about 7B. Its memory bandwidth (2 TB/s on the 80 GB version) means it doesn’t just have more VRAM—it moves data through the GPU faster, which translates to much shorter training times.

The NVIDIA H100 is the current flagship. I don’t know why they choose these names, with the ‘A’ looking very similar to the ‘H’ in many fonts. So be sure when you’re selecting which one you’re getting! It’s faster than the A100 in every dimension—more memory bandwidth, better tensor core performance, support for FP8 precision—and cloud providers charge a premium for it. For most fine-tuning workloads covered in this book, an A100 is more than sufficient, and the cost savings are meaningful.

Newer options like the NVIDIA L4 (24 GB) and L40S (48 GB) are showing up on cloud platforms as cost-effective mid-tier options. The L4 in particular is positioned as a fine-tuning and inference GPU, and its pricing on platforms like Google Cloud makes it attractive for sustained workloads.

Indeed, when you look at Google Colab, you can see the GPUs available for you to use. Please note that this list can change over time—and the ones available at time of writing are shown in [Figure 3-1](#ch03.html_ch03_figure_1_1772730612412203).

![A screenshot of a computer program AI-generated content may be incorrect.](assets/ch03_figure_1_1772730612412203.png)

###### Figure 3-1. Data Center GPUs in Google Colab

As you can see, the H100 and the A100 are available, as are L4 and T4. There’s also a switch for ‘High-RAM,’ where for machines like the A100 that have different VRAM configurations you can switch between the two.

Once you start an instance, there’s a small ‘RAM/DISK’ button at the top right, and selecting that will give you your current machine specs ([Figure 3-2](#ch03.html_ch03_figure_2_1772730612412233)).

![A screenshot of a computer AI-generated content may be incorrect.](assets/ch03_figure_2_1772730612412233.png)

###### Figure 3-2. System Specs for a T4 High Ram

In this case I selected a T4 with High Ram, and you can see that the GPU VRam available is 15.0Gb. Good for LoRA or QLora on smaller models, but not much else. Still—it’s cheap, at 1.41 units per hour, where 100 units cost me $10, so about 14 cents per hour to use. This is a great alternative to paying hundreds, or thousands for your own GPU that would take many hours before it pays for itself!

### Distributed training with multiple GPUs

There will come a point—maybe not in this book, but hopefully sometime in your fine-tuning career—when a single GPU isn’t enough.

Maybe you need to train a 70B model that doesn’t fit on one card even with QLoRA. Maybe you have a production deadline and need to train faster. Or maybe new model innovations have grown so far that they’ve outpaced the readily available VRAM sizes. So you can combine CPUs and pool their VRAM, but that comes with overhead.

There are two fundamental strategies to work with multiple GPUs to solve this problem:

*Data parallelism* copies the *entire* model onto *each* GPU, then splits each training batch across the GPUs. Each GPU processes its slice of the batch independently, computes gradients, and then the gradients are averaged across GPUs before updating the model. The *model* must fit on each individual GPU, but training is faster because you’re effectively multiplying your batch throughput by the number of GPUs. So, when it comes to VRAM size, you have to take into account how much the model takes up, and that will be replicated across each GPU, eating multiples of the available VRAM. The rest of the available VRAM headspace will be used for everything else, and that can typically be shared.

*Model parallelism* (sometimes also called tensor parallelism or pipeline parallelism, depending on the approach) splits the *model* itself across multiple GPUs. Different layers—or even different parts of the same layer—live on different cards. This lets you train models that are too large to fit on any single GPU. It’s more complex to set up and has significant communication overhead, but it’s the only way to handle truly large models on limited hardware.

Frankly, you’ll typically have to use model parallelism, because while data parallelism is better for speeding up training, the replication of the model across different GPUs means that things like the optimizer weights become very difficult to manage without eating excess VRAM.

Fortunately the HuggingFace ‘accelerate’ library makes handling this very approachable.

You can start it by calling the following code:

```
accelerate config
```

This will walk you through an interactive setup where you can specify your GPU system—how many you have, what type of parallelism you want, and how to use precision. Once you’ve got that then you can start your training with code like this:

```
accelerate launch train.py
```

This will use the configuration and span your training load across available GPUs as specified.

That said, for the models and techniques covered in this book, I will focus on using a single low VRAM GPU with LoRA and QLoRA .

Multi-GPU is a “nice to have” for speed, or if you are working with larger models, but it is not a prerequisite for capability. If you have access to multiple GPUs (say, through a cloud instance with 4× A100s), \`accelerate\` makes it easy to take advantage of them. If you don’t, the good news is that you won’t be left behind when skilling up!

# The Software Stack

Now that you understand the hardware requirements, choosing your software stack is probably a bit easier. In this section, we’ll talk through the stack, starting with the operating system.

## Choosing Your Operating System

Ultimately, you want an OS that supports your hardware! Or, if you are choosing cloud-based hardware, you at least need to understand the software stack that the cloud service provider gives you.

### Using Linux

The default choice is generally some form of Linux. If you’re creating your own rig, Ubuntu is the best flavor of this because of its support. For example NVidia drivers and CUDA have first-class Ubuntu support, and virtually every tutorial, Docker image, or Cloud template you’ll see likely assumes you are using it.

If you aren’t familiar with Linux, I’d encourage you to set up a Linux machine using the LTS (Long Term Support) version of Ubuntu. At time of writing, this is Ubuntu 22.04 LTS.

### Using Windows

Windows is an interesting operating system, to be sure! Once the stalwart of the Web and Server business, it kinda lost its way to Linux-based systems thanks to the LAMP (Linux, Apache, Php, MySql) stack. As the cloud evolved, this only accelerated. On the Desktop, Windows is still quite popular, but that popularity has massively waned with the popularity of Apple desktop and laptop computers running MacOS.

To that end, Microsoft introduced WSL2 (Windows Subsystem for Linux 2) which is surprisingly good. It runs a real Linux kernel, and has native CUDA passthrough where your Windows NVidia drivers are shared directly with the WSL2 environment, meaning you don’t need separate installations of them inside of WSL. Given that Windows is still the primary desktop operating system used for *gaming*, NVidia will continue to support Windows as a first class system.

It’s not a *bad* option, and it’s worth considering if you are already using Windows. But directly using Linux instead of a ‘subsystem’ is likely to introduce less complexity.

### Using macOS

For macOS users on Apple Silicon, the situation is continually evolving. PyTorch’s MPS backend works for basic training, and MLX ([*https://opensource.apple.com/projects/mlx/*](https://opensource.apple.com/projects/mlx/)) provides a more native experience. Apple has clearly invested heavily in Machine Learning, and are building the software stack to match.

The primary limitation is ecosystem breadth: libraries like bitsandbytes have limited macOS support, and most troubleshooting resources assume NVIDIA hardware.

My primary system is a Mac Mini, and I use it for many ML tasks, but for fine tuning anything but the smallest models, it can be a bit of a stretch. That being said, it’s nice to be able to use your primary system, and not have to build a dedicated one. So, if you, like me, are a Mac person—don’t rush out to buy a Linux box with an Nvidia GPU *just* yet. You may be able to do enough with your Mac!

And when it doesn’t have the power, using your Mac as a thin client to a cloud GPU is a valid way forward—SSH into a cloud instance and run your training there. Or, of course, using something like Colab within your browser.

## Your Programming Environment

Let’s start with your programming language. While there are many options with various levels of maturity, there’s really only one you should start with—and that’s Python. It has the best support in terms of libraries for all ML and FT tasks, as well as an enormous support ecosystem. It’s hard to think about being effective in any way with ML and FT without first using Python. Later, you might start thinking about low-level efficiency and running code closer to the metal than you would with a high level interpreted language like Python, but don’t get distracted by those! Solidify your base with Python first.

That comes with a drawback. With such a variety of ecosystems, there’s massive drift in version support, not just of Python, but also adjacent technologies like CUDA. To get around this, understanding Virtual environments is a massive prerequisite.

There are two main approaches to this: *Conda* or *Venv*.

*Conda* is the more comprehensive choice. Conda manages not just Python packages but also system-level dependencies like CUDA toolkit versions. This means you can have one environment with CUDA 11.8 and another with CUDA 12.1, switching between them seamlessly. For beginners or anyone who wants their CUDA management handled automatically, I would encourage time and effort investing in Conda!

*Venv (with pip)* is a much lighter weight option. It works really well if CUDA is already installed system-wide, is faster to set up, and pretty easy to manage. I personally like the control I get with multiple Venv setups, and in particular working with a Mac where I don’t have to worry about CUDA, it makes my ML life a lot easier!

Whichever you choose—you will of course need to run Python within it, and with that, you’ll also need PyTorch. The next step would be to install PyTorch with GPU support, so if you are using an NVidia chip and need CUDA, you have to *really* pay attention. I have found that this is the step where most environmental problems begin—so keep a close eye on the details.

The PyTorch foundation has done a great job in simplifying this if you start at: [*https://pytorch.org/get-started/locally/*](https://pytorch.org/get-started/locally/), but it’s still easy to make a mistake.

You can see this in [Figure 3-3](#ch03.html_ch03_figure_3_1772730612412252).

![A screen shot of a computer AI-generated content may be incorrect.](assets/ch03_figure_3_1772730612412252.png)

###### Figure 3-3. Starting with PyTorch

Note that, by default, when you click ‘Linux’ as your machine type, PyTorch will have the CPU type loaded. That’s all very well, but if you’re using a GPU, then PyTorch will effectively ignore it! So, you have to look at the Compute Platform column, and *before* you start with PyTorch make sure you are using the *exact* version that is listed. Intra-version compatibility often leads to problems, and I have found that CUDA releases in particular may be later than supported ones. The old adage of ‘use the latest version,’ which is typically good advice may not apply here. For example, at time of writing, the Compute Platform supports CUDA 12.6, 12.8 and 13.0, but the *latest* CUDA is 13.1. While it’s great to have the latest drivers for the best support of your machine should you be using it also for gaming or other activities, I strongly recommend you *strictly* follow the platform that PyTorch supports. In this case, you would use 13.0.

Once you’ve installed it, you’re still not oout of the woods! I would encourage double checking that it’s working as recommended with code like this:

```
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

If the CUDA available line prints ‘false’, something went wrong. You might have CUDA installed, and there’s a mismatch with the PyTorch version, or you accidentally installed the CPU version of PyTorch. I hold my hand up as guilty there—did it too many times!

Another thing you can investigate is the version that your machine or environment is using by issuing the nvidia-smi command. Make sure it matches the one for which you selected a PyTorch environment. If it doesn’t get new drivers that match one of the options, clean your system, and start again!

It’s a lot of hassle up front, but it can save you a world of hurt down the line.

## The Core Library Stack

Now that we have PyTorch verified, and it’s working with the GPU, you’ll next install the libraries you’ll use throughout this book. Each serves a specific role, and understanding what each one does will help you understand and debug issues later.

*PyTorch*, of course, is the deep learning framework everything else is built on. It provides the tensor operations, automatic differentiation (autograd), and GPU acceleration that power model training. You installed this in the previous step. Check out my O’Reilly book “AI and Machine Learning for Coders in PyTorch” if you want to learn more.

*Hugging Face Transformers* is the library that provides access to thousands of pre-trained models through a consistent API. When you write code like \`AutoModelForCausalLM.from\_pretrained(“meta-llama/Llama-3.2-3B”)\`, where models are specified with a namespace, Transformers downloads the model architecture and weights from the Hugging Face Hub and turns them into PyTorch model that’s ready to use. It handles the differences between architectures—Llama, Mistral, Gemma, Phi, Qwen—behind a unified interface, so your training code doesn’t need to change when you switch models.

*Hugging Face PEFT* (Parameter-Efficient Fine-Tuning) implements the adapter methods that make fine-tuning practical on consumer hardware. LoRA, QLoRA, and other techniques are all provided through PEFT. When you apply a LoRA adapter to a model, PEFT injects small trainable matrices into the model’s layers while freezing the original weights. We’ll explore PEFT in depth in Chapter X.

*BitsAndBytes* is the quantization library that enables QLoRA specifically. As we saw earlier in this Chapter, in order to work on smaller grade GPUs, quantization is necessary, and bitsandbytes is your friend for this. It provides 4-bit and 8-bit quantization for model weights, allowing you to load a 14 GB model into 3.5 GB of VRAM. It’s developed by Tim Dettmers and is tightly integrated with Hugging Face Transformers—you can load a model in 4-bit precision with a single argument: \`load\_in\_4bit=True\`. This library, at time of writing, is *highly* optimized for NVidia. Support for other environments (Mac and AMD) is being added all the time, but you may hit walls here if you aren’t using an NVidia/CUDA stack.

*TRL* (Transformer Reinforcement Learning) provides high-level training APIs. Its \`SFTTrainer\` (Supervised Fine-Tuning Trainer) wraps PyTorch’s training loop with sensible defaults for fine-tuning language models. It handles data collation, gradient accumulation, logging, checkpointing, and evaluation. TRL also provides \`DPOTrainer\` for direct preference optimization, which we’ll cover in Chapter X.

*Hugging Face Accelerate* manages the details of running training across different hardware configurations. It handles mixed-precision training (using FP16 or BF16 to speed up computation), device placement (making sure tensors end up on the right GPU), and distributed training (splitting work across multiple GPUs). In most cases, you won’t interact with Accelerate directly—TRL uses it behind the scenes—but it’s there if you need to customize your training setup.

*Hugging Face Datasets* provides efficient data loading. It uses memory-mapped files and a framework called Apache Arrow under the hood, giving you the means to work with datasets much larger than your RAM. It integrates with the Hugging Face Hub, so loading a public dataset is a one-liner: \`load\_dataset(“tatsu-lab/alpaca”)\`.

*Weights & Biases (wandb)* is an experiment tracking platform. Every training run generates metrics—loss curves, learning rates, gradient norms—and you need somewhere to store and visualize them. Wandb integrates with TRL’s \`SFTTrainer\` with a single configuration parameter, and its free tier is generous enough for individual practitioners.

All of these can be installed with a single pip command:

```
pip install transformers peft bitsandbytes trl accelerate datasets 
            wandb sentencepiece protobuf
```

The eagle-eyed among you might have spotted `sentencepiece` and `protobuf` at the end. These are tokenizer dependencies that some models require. Including them pre-emptively so you don’t hit a confusing error when you use those models, and wonder if there was something missing in your stack!

You can test this with the code I’ve provided in the GitHub as a notebook called Chapter3.ipynb.

When run on colab with an A100 NVidia GPU, you should see output like this. Note that inference worked, as did a very simple fine-tune using LoRA

```
==================================================
STEP 1: GPU Check
==================================================
GPU: NVIDIA A100-SXM4-40GB
VRAM: 42.4 GB
 
==================================================
STEP 2: Loading model in 4-bit
==================================================
...
Model loaded: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Model memory footprint: 0.75 GB
 
==================================================
STEP 3: Running inference
==================================================
Prompt: The capital of France is
Response: The capital of France is Paris.
What is the capital of France? Paris.
 
==================================================
STEP 4: Applying LoRA adapter
==================================================
trainable params: 1,126,400 || all params: 1,101,174,784 || trainable%: 0.1023
 
==================================================
STEP 5: Single training step
==================================================
Loss: 3.4863
Gradient computed successfully!
 
==================================================
ALL CHECKS PASSED—Your environment is ready!
==================================================
```

If you don’t get these results, these are some common errors:

*   “CUDA is Not available” - You have a CPU-only PyTorch. That’s not the case in Colab, unless, of course, you select a CPU backend!
    
*   “RuntimeError: CUDA out of memory” - The provided code downloads a tiny model of less than 1Gb. If you’re getting this error you either have a tiny GPU, or, more likely another model loaded and ate most of the memory and wasn’t cleared before this run!
    
*   The ‘bitsandbytes’ library is compiled against your CUDA installation. I’ve seen lots of compilation errors because of mismatches between it and the CUDA toolkit. Check your versions carefully!
    
*   Model Download failures. The HuggingFace APIs provide a nice visualization of the fragmented download. Keep an eye on these to check for any failure. Models get downloaded to a hidden cache directory (~/.cache/huggingface/)—and if there isn’t sufficient space here, you can have problems.
    

## Experiment Tracking

Beyond directory organization, you need a way to track what happened during each training run. What was the loss at step 1,000? How did the learning rate decay? At which step did the model start overfitting? These questions are nearly impossible to answer from log files alone, but trivial with a proper experiment tracker.

### Weights and Biases (wandb)

We previously saw Weights & Biases (wandb), and this is the standard tool for experiment tracking for most of the ML community. It’s free for individual use, integrates easily with your code, and provides a web dashboard where you can visualize loss curves, compare runs side by side, and annotate your results.

Setting it up is straightforward:

```
# Create a free account at wandb.ai, then
wandb login
# Paste your API key when prompted
```

Once authenticated, the transformers library gives you an API to automatically log into wandb. You can also configure it explicitly in your training arguments:

```
from transformers import TrainingArguments
 
training_args = TrainingArguments(
    output_dir="./outputs/checkpoints",
    report_to="wandb",
    run_name="...",
    # ... other arguments
)
```

Every training run becomes a dashboard showing loss curves, learning rate schedules, gradient norms, and GPU memory usage. You can track them in real time as training progresses, which is useful for a manual early stop if things aren’t going well!

### TensorBoard

Part of the TensorFlow ecosystem, is an open source alternative that runs entirely locally, and can easily be used in environments like Colab.

```
pip install tensorboard
 
# In your TrainingArguments:
# report_to="tensorboard"
 
# After training, launch the viewer:
tensorboard --logdir ./logs
```

Even if you don’t use these tools, I strongly recommend that you keep a simple experiment log—even a spreadsheet or notepad—that records the date, model name, dataset version, key hyperparameters (LoRA rank, learning rate, epochs), final training loss, and evaluation score for each run.

When you’ve run thirty experiments and need to find the best one, you’ll be grateful for this discipline. You’ll also find that it gets you into a routine of naming your checkpoints according to entries in this sheet or notepad, making them easier to find later, too!

# Cloud and Pay-as-You-Go

In the previous section, I used Colab as the environment for the model with an A100 backend. It’s a really good option if you don’t want to manage ownership of a GPU *or* build your own environment, the host does that for you. It’s a great example of Pay-as-you-Go making sense.

But Colab isn’t the only option, of course. I’d like to introduce some others so you can see what’s available.

*Kaggle Notebooks* offer a similar experience to Colab. They typically have free GPU access (I have recently seen T4 and P100 available) that have a 30-hour-per-week quota. That’s really generous. The interface is similar to Colab, albeit a little more vanilla notebook. One added bonus is for datasets—many are hosted on Kaggle already, so the integration is a little easier.

*Lambda Labs* takes it in a different direction. Instead of a notebook, you get a full Linux machine via SSH.

They offer A100 and H100 instances and a carte blanche to install your own software, manage your own files, and have complete control over the environment.

It’s really useful if you want to poke around in the underlying software stack and not have it fully managed for you, and feels more like your own machine in the cloud.

*RunPod* is an interesting alternative, operating as a GPU marketplace with both “Community Cloud” (cheaper, less reliable) and “Secure Cloud” (more expensive, more reliable) options. They offer spot/interruptible instances at lower prices—but the tradeoff is that your instance might be pre-empted by others—which is not ideal for long training runs. One nice thing is that ML software stacks are usually preinstalled which helps your setup time.

Another interesting option is [Vast.ai](http://vast.ai), which is kind of like the Uber of GPU rental. It’s where individual GPU owners can rent out their hardware! Of course reliability and security can vary.

Amd also offers *Digital Ocean*, a cloud service with ROCm powered AMD chips available to use.

# Local or Cloud—How to Decide

While every situation is different, I generally follow the following framework to help me decide where and how to do any tuning or fine-tuning of a model.

You now understand the hardware options and the software stack. The remaining question is: where do I run this? The answer depends on your specific situation, and there’s no universally right choice. Here’s a decision framework to help you think through it.

The first question is simple: Do you have a GPU with 16 GB or more of VRAM? If yes, your local machine can handle QLoRA on models up to 7B (with 16 GB). More VRAM, bigger models. This covers most of what we’ll do in this book. If no, cloud is your primary option.

Then I would ask you: Is this a one-time experiment or ongoing work? If you’re working through this book and plan to fine-tune models as part of your regular work, investing in a local GPU pays for itself within a few months. If this is a one-off exploration, cloud gives you access without commitment, and typically with much lower outlay.

Do you have data sovereignty requirements? Some organizations—particularly in healthcare, finance, and government—have strict rules about where data can be processed. If your training data can’t leave your premises, local training (or a private cloud deployment) is mandatory. This is one of the strongest arguments for local hardware: your data never leaves your building.

Are you training models larger than 13B parameters? For 70B models, even QLoRA needs roughly 40 GB of VRAM—beyond any consumer GPU. You need an A100 (cloud) or a multi-GPU setup. For models up to 13B, a single RTX 4090 with QLoRA is likely enough.

The approach that most practitioners settle into is a hybrid workflow: develop and debug locally (or on free Colab) with a small model and a subset of data, then run the full training job on a cloud GPU when everything is working.

Local is for iteration speed—you can restart a training run in seconds, tweak a hyperparameter, and try again without provisioning anything. Cloud is for the final training run where you need a bigger GPU or longer runtime.

The examples in this book are designed to work across three tiers:

Google Colab Free (T4, 16 GB VRAM)
: 

Qlora on models up to 7B parameters. This is the minimum requirement. If you have access to nothing else, you can still follow every example.

Local RTX 4090 or Colab Pro A100 (24-40 GB VRAM)
: 

Qlora on models up to 13B, or LoRA (without quantization) on 7B models. More flexibility, faster training.

Cloud A100 80 GB
: 

For the adventurous—LoRA on 70B models, full fine-tuning on 7B models, or any experiment where VRAM is the bottleneck.

Ultimately, when you’ve finished with the book, my advice would be to pick the tier that matches your resources and don’t worry about the rest. The techniques are the same regardless of which GPU you’re using; only the model size and training speed change.

# Summary

In this Chapter you walked through setting up your fine-tuning workbench—the hardware, software, and environment decisions that underpin everything else in this book. We explored why GPU memory (VRAM) is the single most important constraint you’ll face, and how techniques like LoRA and QLoRA make fine-tuning accessible on hardware you can actually afford. You surveyed the software stack—from PyTorch and Hugging Face through to experiment tracking—and compared local and cloud options so you can pick the setup that fits your situation and budget.

With your environment verified and a model downloaded, you now have a solid, working foundation. The tools are in place, the math makes sense, and you’re ready to focus on what actually matters: your data, your training strategy, and building expert models that are truly your own.

# About the Author

**Laurence Moroney** is an award-winning researcher and best-selling author. A veteran in the programming and machine learning industry, Laurence has authored over 20 books, including top selling programming books, science fiction novels, and even movie-related comic books. He teaches popular AI specializations on Coursera with Andrew Ng of deeplearning.ai. A recognized speaker all over the world, Laurence is passionate about demystifying AI and ML for software developers.