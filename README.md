# RL-simulator

## abstract
　It is very difficult for users to find their favorite items from a huge set of items. Therefore, it is necessary to assist the user's decision-making process with a recommendation model.　　
 
 　Z. Sun et al. conducted an exhaustive study of the hyperparameters during model training for deep learning-based recommendation problems, and pointed out that the metrics value at a single time is affected by the hyperparameters [1].
  
　On the other hand, in actual recommendation problems, not only the model accuracy at a specific time but also the long-term effects of recommendation on users and services are of great interest. M. Chen et al. conducted online experiments and discussed the long-term effects of recommendations on users and services [2].
 
　However, since they used online experiments, the long-term effects of different hyper-parameters on users and services are not well known yet. Therefore, in this study, we conducted simulations of multiple off-policy recommendation models and examined the long-term effects on services.
In particular, we compared models based on deep reinforcement learning, and examined the relationship between metrics values and models. For this purpose, we created the simulator shown in the figure on the right and conducted simulations using MovieLens1M as data. We confirmed that the time variation of the metrics values strongly depends on the model. We also confirmed that the difference in the formulation of the measures affects the tendency of recommendation.

## process of simulator
![process drawio](https://user-images.githubusercontent.com/40622501/187359819-c42532b5-5614-4830-a24f-664b664c0a9d.png)

## results
![metrics](https://user-images.githubusercontent.com/40622501/187360348-57c694a1-252a-4743-bf24-3e5a3c1b0328.png)

## references
- [1] Z. Sun et al., “Are we evaluating rigorously? Benchmarking recommendation for reproducible evaluation and fair comparison,” presented at the RecSys ’20: Fourteenth ACM Conference on Recommender Systems, Virtual Event Brazil, Sep. 2020. doi: 10.1145/3383313.3412489. 
- [2] M. Chen et al., “Values of User Exploration in Recommender Systems,” in Fifteenth ACM Conference on Recommender Systems, New York, NY, USA: Association for Computing Machinery, 2021, pp. 85‒95.
