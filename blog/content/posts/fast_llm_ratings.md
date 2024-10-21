---
title: "How I Sped Up the ChatBot Arena Ratings Calculations from 19 Minutes to 8 Seconds"
date: 2024-10-20
draft: false
tags: ["rating systems", "Bradley-Terry", "optimization", "paired comparison"]
---

# TLDR:
* Vectorized preprocessing
* Deduplication and sample weighting
* Multinomial bootstrapping
* Avoid recomputation
* Exploit sparsity
* Exploit symmetry
* Multiprocessing
* Optimize in-memory data layout

Read the code: [PR: Accelerate Bradley Terry MLE model fitting](https://github.com/lm-sys/FastChat/pull/3523)

# What is ChatBot Arena?
[ChatBot Arena](https://lmarena.ai/) is a website developed by [LMSYS](https://lmsys.org/) where users can put in their own prompts and get replies from two random anonymized LLMs. The user can then vote for which response they prefer or mark it as a tie. By aggregating all of the comparisons, the Arena developers can create a leaderboard of the top LLMs as judged by human preferences.
![alt text](../leaderboard.png)

This provides a great utility to the LLM community in a couple of ways. LLM users can consider the rankings when deciding which LLMs to use in the applications. And perhaps more crucially, LLM developers can get close-to-live feedback of how their models compare to their competitors on a benchmark which cannot be overfit to in the traditional sense. (It has been noted then LLMs trained on ChatBot Arena data may be overfit to the chat task itself, but the conversations voted on are guaranteed not to have been contained in the LLM training data) This utility is used by top labs to A/B test new model versions and get feedback signal before releasing publicly such as in the "im-also-a-good-gpt2-chatbot" which was later [confirmed](https://x.com/LiamFedus/status/1790064963966370209) to be gpt-4o.

# How are the Scores Calculated?
Originally, the scores utilized the [Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system), developed by Árpád Élő in the 1960's to rank chess players based on the outcomes of their matches. However this is not an ideal system to rank LLMs since it was designed to track the changing skill of competitors over time, but the LLM "skill" is more or less constant over time. (It is possible for changes to occur over time due to system prompt changes, quantization or decoding/sampling changes made under the API hood)

### Bradley Terry
Anyway, with the assumption that skill is not changing over time, it's more appropriate to use other methods known in the literature as [paired comparison models](https://encyclopediaofmath.org/wiki/Paired_comparison_model). One of the most common of these is the [Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) (BT) devised in the 1950's (and building on work of Zermelo from the 1920's). In this model each competitor i has a strength parameter \(\\theta_i \\gt 0\) and the probability of competitor a beating competitor b is modeled as:
$$p(a \succ b) = \frac{\theta_a}{\theta_a + \theta_b}$$
So to fit Bradley-Terry model, we want to find the vector \(\theta\) which maximizes the probability of the observed data. This process is called [maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) and is the framework behind much of modern machine learning. (For now we represent a dataset of matches as a set of tuples \((w,j)\) with \(w\) and \(l\) being the ids of the winner and loser):

$$\theta^*= \underset{\theta}{\operatorname{argmax}} \prod_{(w,l) \in \text{matches}} \frac{{\theta_w}}{{\theta_w} + {\theta_l}}$$

This is impractical for several reasons, one is that when you multiply too many probabilities together which are each less than one you get a number closer and closer to 0 which will run into numerical issues when trying to represent it with a floating point number. To get around this we will apply log to the entire expression, which does not change the result since log is monotonic and it will turn the product into a sum. Secondly in order to ensure that all of the \(\theta\)'s are positive, we will apply an exponential transform and optimize parameters \(r_i = \log(\theta_i)\) (so that \(\theta_i = e^{r_i}\)) instead. With these changes the maximizer of the now log likelihood can be written:
$$\mathbf{r}^* = \underset{\mathbf{r}}{\operatorname{argmax}} \sum_{(w,l) \in \text{matches}} \log\left(\frac{e^{r_w}}{e^{r_w} + e^{r_l}}\right)$$
Some of you may have recoginzed this form and in fact it can be written instead with our good friend the logistic sigmoid! (Full derivation in the appendix)
$$\mathbf{r}^* = \underset{\mathbf{r}}{\operatorname{argmax}} \sum_{(w,l) \in \text{matches}} \log(\sigma(r_w - r_l))$$

Next we allow for draws. To do this we will change the encoding of the dataset to be \((a,b,y)\) tuples where \(a\) and \(b\) still represent the competitors, but now we introduce the label/outcome \(\textbf{y}\)  \(y_i \in \{1, 0.5, 0\}\) and represents the outcome or label for a matchup indexed by \(n\). In this setup 1 represents a win for a (loss for b), 0.5 a tie, and 0 a win for b (loss for a). With the assumption gone we can write the objective:
$$\mathbf{r}^* = \underset{\mathbf{r}}{\operatorname{argmax}} \sum_{(a,b,y) \in \text{matches}} \left[y * \log(\sigma(r_a - r_b)) + (1-y) * \log(\sigma(r_b - r_a))\right]$$

### The Logistic Regression Connection
You might have noticed that the final expression in the previous section looked very similar to the [logistic regression loss function](https://en.wikipedia.org/wiki/Logistic_regression#Fit). 

$$\mathbf{w}^* = \underset{\mathbf{r}}{\operatorname{argmin}} - \sum_{i=1}^N \left[y_i * \log(\sigma(x_i^\top w)) + (1-y_i) * \log(1 - \sigma(x_i^\top w))\right]$$
Here \(X\ \in \mathbb{R}^{N \times d}\) is the data matrix and \(w\ \in \mathbb{R}^d\) is a vector of weights. The difference being only in that how the logits (the thing inside the sigmoid) are computed. In BT, the logit is the difference in the ratings, in logistic regression the logit is the dot product between a vector of features and the vector of model weights. Also logistic regression is usually written as minimizing the negative of log likelihood rather than maximizing the log likelihood but the two are equivalent and we can now see an extremely close connection between the Bradley-Terry model and logistic regression. The connection is so close and convenient that the ChatBot arena creators used the scikit-learn logistic regression implementation to calculate the ratings on their site. ([blog post](https://lmsys.org/blog/2023-12-07-leaderboard/#transition-from-online-elo-rating-system-to-bradley-terry-model), [reference notebook](https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH))

To do this you have to do some preprocessing to construct the matrix \(X\) out of the dataset of matchups. The key is to make \(d\) (the dimensionality of the logistic regressio) equal to \(C\), the total number of competitors in the dataset, and to fill the matrix sparsely with only two non-zero entries per row, indicating the two competitors in that metch, with a value of 1 at the index representing competitor a and a -1 for competitor b.

Here's an example of how to represent 2 matches with 4 total competitors (a,b,c,d):
$$\begin{bmatrix}
1 & -1 & 0 & 0 \\\
0 & 0 & 1 & -1
\end{bmatrix}$$
The first row represents a match between a (index 0) and b (index 1), and the second row encodes a match between c (index 2) and d (index 3). The second part of the story is to understand that \(w\) (the logistic regression weights) is the same as \(r\) (the vector of Bradley-Terry ratings). This makes sense when you think about Bradley-Terry from a machine learning perpsective where the ratings are the parameters of the model you are training. In this case \(r = [r_a, r_b, r_c, r_d]\).

Ok now let's put together the matrix representation of matches with the parameter vector and multiply \(X\) by \(w\).
$$\begin{bmatrix}
1 & -1 & 0 & 0 \\\
0 & 0 & 1 & -1
\end{bmatrix} \times \begin{bmatrix}
r_a \\\
r_b \\\
r_c \\\
r_d
\end{bmatrix} = \begin{bmatrix}
r_a - r_b \\\
r_c - r_d \end{bmatrix}$$
And look at that, we've gotten back to the difference in ratings that we expect to put into the sigmoid during the BT model fitting and we got there fully using the logistic regression toolbox.

### Bootstrapping
The final piece of the picture in order to produce the ratings (and error bars) for the ratings displayed on the leaderboard is [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)). This is the process of repeatedly sampling with replacement from a dataset, and performing some computation on each re-sampled dataset. In this case it just samples rows from the original matchup dataset 100 times, and computes the Bradley-Terry ratings on each sample. Since the rows used to compute the ratings are different, the final ratings produced are different in each iteration. At the end they take the median as the main ratings and use the quantiles to set the error bars. 

# How to Make it Super Fast?
In order to make this code much faster, I performed several types of optimizations targeting different aspects of the setup.

## Custom Bradley-Terry Model Implementation
While the above scheme does make use of the highly optimized software in numpy/scipy/scikit-learn, there are still some drawbacks.

### Draws
In the above formulation, if the label \(y\) is 0.5, then the loss and gradients work out to be half of the loss/grad if it was a win, plus half the loss/grad if it was a loss.
$$\mathcal{L}_i = \frac{1}{2} * \log(\sigma(r_a - r_b)) + \frac{1}{2} * \log(\sigma(r_b - r_a))$$
This is a problem for LogisticRegression in scikit-learn which is only works for this application with labels of 0 and 1 (you can do 3 class classification with sklearn logistic regression but it won't correspond to the Bradley-Terry model anymore). To get around this, the implementation LMSYS used was to duplicate all rows of the dataset. If the row originally had the label 0 or 1, then both copies were unmodified, but if it was a draw in the original dataset, then in the duplicated dataset one row has the label set to 1 and the other to 0. Now the total loss is the sum over twice as many rows but the optimal ratings are the same. This is a very clever approach by LMSYS which allowed them to utilize scikit-learn, unfortunately this introduced extra cost in two places:
* the preprocessing to duplicate the data and re-label the draws is expensive (it uses pandas)
* the model fitting is now twice as expensive

### Feature Dimension
In this case \(C\), the number of competitors (models) is 138. That's the total number of models that have been shown to users in ChatBot Arena. This means the the data matrix \(X\)
is N by 138. In order to compute the logit (input to the sigmoid), we end up doing a dot product between two 138 dimensional vectors to compute the difference between two ratings. In my custom Bradley-Terry implementation, instead of \(X\), we store an N by 2 matrix of integer ids representing the models, then to compute the logits we simply need to use the model ids to index into the ratings array and do a substraction. Yes dot products and matrix multiplications are fast these days but it's still not faster than a vectorized subtraction. Here's a demonstration of how the code looks in each case:

```python
# logistic regression method
lr_logits = np.dot(X, ratings) # (N x d) x d matrix vector multiplication

matchup_ratings = ratings[matchups] # (N x 2) index into d dimensional array
bt_logits = matchup_ratings[:, 0] - matchup_ratings[:, 1] # (N x 2) elementwise subtraction
```

## Deduplication and Weighting
Recall from above the the total loss/grad are simply the sums of the loss/grad in each row. The next optimization exploits the fact that while there are only 138 models, there are like 1.7 millions match rows. This means that a lot of the rows are duplicates. If we can do preprocessing to find only the unique matchups, and how many times they occur, we can compute the loss/grad for each unique possibility, and multiply it by how many times that possibility occured. If it occured many times, it will likely be cheaper to compute.
At the simplest level you can think of replacing (val + val + val + val + val) with 5 * val which will take less cycles. To implement this efficiently, I created a single numpy array with 3 columns: ```model_a_id, model_b_id, outcome```. On this array I used ```np.unique(..., return_counts=True, axis=0)```. This efficiently identifies all the unique possibilities row-wise and their counts. I then use the counts essentially as sample weights in the optimization. In practice this brought down the number of total rows used in each optimization step from about 1.7 million to 22 thousand which saved a lot of time.

## Vectorized Preprocessing
In the original code, the \(X\) matrix was constructed using an interesting though ultimately suboptimal method which first using [pivot tables](https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html) to get 3 dataframes of shape  d by d representing the head to head wins, ties, and losses for pairs of models. Then an \(X\) matrix is constructed of shape d * (d - 1) * 2 by d. The number of rows is the total number of possible model combinations. This is already much better than using the full 1.7 million rows, but it is still larger to allocate the memory for the total number of possible combinations rather than the number of observed combinations. The real slowdown here is that the matrix was constructed using a python for loop over the rows and columns of the pivot table which is notoriously slow. [[1](https://stackoverflow.com/questions/48951047/iteration-over-columns-and-rows-in-pandas-dataframe)] [[2](https://stackoverflow.com/questions/62457975/python-pandas-iterate-over-dataframe-rows-with-iterrows-is-slow-can-it-be-repla)] [[3](https://www.youtube.com/watch?v=LkBz5NS-RF8)] [[4](https://blog.dailydoseofds.com/p/why-pandas-dataframe-iteration-is)] [[5](https://www.reddit.com/r/learnpython/comments/7m2txy/improve_speed_of_iterating_over_pandas_dataframe/)] [[6](https://blog.dailydoseofds.com/p/why-is-iteration-ridiculously-slow)] [[7](https://github.com/pandas-dev/pandas/issues/10334)] [[8](https://realpython.com/pandas-iterate-over-rows/)] [[9](https://towardsdatascience.com/how-to-make-your-pandas-loop-71-803-times-faster-805030df4f06)] [[10](https://medium.com/@noah.samuel.harrison/stop-looping-through-rows-in-pandas-dataframes-9375dde3410b)]

We speed this up by a great deal by by only using and not constructing a full \(X\) matrix but only the integer ids. We discussed above how we use ```np.unique``` to get unique matchups, and later we'll discuss how we mapped the model names to integers. So here I'll just show how we very efficiently mapped the "winner" column of the original dataframe to the outcomes numpy array of 0, 0.5, and 1. Create appropriate sized array with 0.5 as defualt for draws, use vectorized masks to override the correct values with 1 or 0.

```python
outcomes = np.full(len(df), 0.5)
outcomes[df["winner"] == "model_a"] = 1.0
outcomes[df["winner"] == "model_b"] = 0.0
```

## Bootstrapping
This section will be large and contain several subsections because it is both compilcated and the methods used gave exceptionally large speedups. The three core areas are parallelization, reducing repeated computation, and a clever method of sampling. In the original code, it does a for loop and in each iteration it does the following:
1. Use ```pd.DataFrame.sample(frac=1.0, replace=True)``` to get a bootstrap sample of the original dataframe
2. Do the preprocessing described above
3. Fit the BT model on the preprocessed data

### Parallelization
This one is relatively straightforward, the original code used a regular python for loop, each iteration does not use 100% of the CPU and they are not dependent on each other so they can be computed in parallel to save time by using all CPU cores at full capacity. I did this using standard python [```multiprocessing.Pool```](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool). I used ```imap_unordered``` for maximal efficiency sine I don't care about the order of the samples.

### Problem: Repeated Computation
As stated above, the existing preprocessing method was pretty slow, and now seeing it inside a for loop that runs 100 or 1000 times should be concerning. This repeated preprocessing was one of the biggest optimization surfaces. The issue is that it's not immediately obvious how to to fix it.

At a high level yeah it's simple, we want to do the preprocessing once at the beginning and then use the preprocessed data in each iteration. We could do this by preprocessing matchups once at the beginning, then doing bootstrap samples on that array, then doing the rest of the preprocessing (deduplication/weighting) on each bootstrap sample. This is still not optimal though since we are still repeating that second part many times.

We may think we can avoid this by doing the full preprocessing, and then doing the bootstrap sampling on the final array of unique rows. I actually did do this and it was fast, but I was dismayed to find that the results were very different from the original method. I realized that this was because when you pick or drop one row from the unique array, that corresponds to picking or dropping *all of the instances of that matchup* from the orignal dataset, which is *not* what bootstrapping is. Each row needs to be picked or dropped independently.

So that's the question that kept me up at night: "how can I pull bootstrap samples directly from the space of unique options?"

### Solution: Multinomial Sampling
The beginnings of the answer came to me while walking around the neighborhood. I realized that in the "sample then process" approach, the process part is counting how many times each unique row was picked. So the question is instead of "sample then count", we should directly sample the counts. For each unique row, we want to pull a random number that represents how many times that row will appear in the final bootstrap dataset. Fortunately that is a well known problem and is decribed exactly by the [multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution). This is perfect since we can do the initial preprocessing once to get the unique rows and their counts, and then use the counts as the sampling weights for the multinomial bootstrap! It's also awesome since the actual "data" used in the optimization step is the same for every bootstrap iteration and can actually be shared in memory rather than duplicated. The **only** thing that is different across the different samples are the counts/weights applied to the matchups.

```python
def compute_bootstrap_bt(
    df,
    num_boot,
):
    matchups, outcomes, models, weights = preprocess_for_bt(df)
    # bootstrap sample the unique outcomes and their counts directly using the multinomial distribution
    rng = np.random.default_rng(seed=0)
    idxs = rng.multinomial(
        n=len(df), pvals=weights / weights.sum(), size=(num_boot)
    )
    # only the distribution over their occurance counts changes between samples (and it can be 0)
    boot_weights = idxs.astype(np.float64) / len(df)
    # the only thing different across samples is the distribution of weights
    bt_fn = partial(fit_bt, matchups, outcomes, n_models=len(models), tol=tol)
    results = np.empty((num_boot, len(models)))
    with mp.Pool(processes=os.cpu_count()) as pool:
        for idx, result in enumerate(pool.imap_unordered(bt_fn, boot_weights)):
            results[idx,:] = result
    return np.median(ratings, axis=0)
```


## Various Micro Optimizations
At every single step I wanted to be as efficient as possible and in the pursuit of my goal I found some little things I didn't know about before and that I'm kind of proud of.

### Mapping Model Names to Indices
In order to create the ```matchups``` array of model indices, I have to map assign an integer to each unique model name and then map all of the values in the original dataframe to their appropriate integer. Early on I used a combination of ```pd.unique```, ```.sort``` and ```DataFrame.map```, but after some research I found [```pd.factorize```](https://pandas.pydata.org/docs/reference/api/pandas.factorize.html) which does that but really fast. (it also flattens the produced indices so you gotta re-stack them)


```python
n_rows = len(df)
model_indices, models = pd.factorize(pd.concat([df["model_a"], df["model_b"]]))
matchups = np.column_stack([model_indices[:n_rows], model_indices[n_rows:]])
```

### Symmetry of Gradients
There is a cool detail about the Bradley-Terry (and Elo) model that exploits the structure of the equation for the loss. Since the logit is \(r_a - r_b\), the gradient of the loss with respect to \(r_a\) and with respect to \(r_b\) will be the same except with a negative sign on the latter due to the chain rule. This means that you only need to compute the gradient once and then flip the sign when applying it to update b. 

### Aggregating the Gradients
While computing the gradients, you get a gradient of shape (N,) which is the gradient for each competitor for each row. But in order to do gradient descent (or [L-BFGS](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) in this case), you need the gradient of shape d (the number of models). I use numpy a lot but it took talking to Claude to get this one, but [```np.add.at```](https://stackoverflow.com/questions/45473896/np-add-at-indexing-with-array) is a really marvelous function that does this. This operation is also known as [```scatter_add```](https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/add.html). This can be combined with a broadcast multiplication of the gradient by ```[1.0, -1.0]``` (exploit the above symmetry) to do a super efficient gradient aggregation.

```python
matchups_grads = (outcomes - probs) * weights
model_grad = np.zeros_like(ratings)
# aggregate gradients at the model level using the indices in matchups
np.add.at(
    model_grad, # array to add into
    matchups,   # indices to add at
    matchups_grads[:, None] * np.array([1.0, -1.0]) # values to add (values added to b are sign flipped)
)
```

### Something Kinda Weird about Pandas and JSON
During this process I worked in a dev repo [faster-arena-elo](https://github.com/cthorrez/faster-arena-elo) and for quick iteration, I did some of the initial preprocessing such as filtering out non-ananymized rows, or truncating to a smaller subsample and writing that to an intermediate parquet file and reading with polars for faster disk loading times on the tests. I noticed that this actually sped up the original baseline by a lot. As I debugged I found that you could get a ~3.5x speedup (19 min -> 5 min) in the original code simply by converting the original pandas dataframe to polars and back, or even by writing it to disk as parquet and then reading it back using only pandas. My best guess here is that the original data being a json, when it is loaded by pandas has a very poor arrangement in memory like it isn't contiguous or causes a lot of cache misses, and that routing through a better representation and back can fix that. (I actually didn't even apply this change in the PR so the entire thing could still speed up more)

```python
# routing through parquet
buffer = io.BytesIO()
df.to_parquet(buffer)
df = pd.read_parquet(buffer)
```

If you have slow pandas code that reads from json and don't have the time to migrate to polars you could get a decent speedup with this little trick.


### Caveat about Timing
An intersting fact to note is that with all of the optimizations applied to the preprocessing and model fitting, the single sample run of Bradley-Terry model fitting is atually **only slightly faster** than the original code! (5.1s -> 2.3s on my laptop) Despite the fact that it is highly optimized, and produces a very efficient representation, the new code does more processing work overall to produce that representation. It's only when we do the bootstrap that we reap all of the benefits of the deduplication, removing repeated computation, and parallelization. 

# Conclusion
So, with all of those optimizations (and some more), the overall runtime of Bradley-Terry model for 100 bootstrap samples was reduced from over 19 minutes, to just 8 seconds.
![alt text](../times.png)

I learned a lot of lessons along the way on how to target various optimization surfaces ranging across data structures and formats, algorithmic complexity, mathematical optimization. My parting thought is to encourage all ml practitioners to implement things from scratch yourself sometimes. This process builds deep understanding if the data and models you are working with which can often be leveraged for massive improvements in speed. An in ML, improvements in speed mean less cost, more experiment iterations, and better experiences for your users.


# Appendix
### Exponential Reparameterized BT with Sigmoid
Recall \(\sigma(x) = \frac{1}{1 + e^{-x}}\)

$$
p(i \succ j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} * \frac{e^{-r_i}}{e^{-r_i}} = \frac{1}{1 + e^{r_j}*e^{-r_i}} = \frac{1}{1 + e^{-(r_i - r_j)}} = \sigma(r_i - r_j)
$$
