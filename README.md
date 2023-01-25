# RL-simulator

## abstract
It can be challenging for users to locate their desired items amongst an extensive array of options. Thus, it is essential to aid the user's decision-making process through the implementation of a recommendation model. 

 Z. Sun et al. conducted a comprehensive examination of the impact of hyperparameters during training for deep learning-based recommendation problems and posited that the metric values a certain point in time are influenced by the hyperparameters [1]. 
 
  Conversely, in real-world recommendation scenarios, not only is the accuracy of the model at a specific moment crucial but also the long-term effects of recommendations on both users and services are of paramount importance. M. Chen et al. carried out online experiments and delved into the long-term effects of recommendations on users and services [2]. 
  
   However, as they utilized online experiments, the long-term effects of various hyperparameters on both users and services are yet to be fully understood. Therefore, in this study, we simulated multiple off-policy recommendation models and analyzed the long-term effects on services. Specifically, we compared models utilizing deep reinforcement learning and examined the correlation between metric values and models. To achieve this, we constructed the simulator depicted in the illustration on the right and conducted simulations using MovieLens1M as data. We established that the temporal variation of metric values is heavily dependent on the model employed. Additionally, we established that variations in the formulation of measures can alter the inclination of recommendations.

## process of simulator
![process drawio](https://user-images.githubusercontent.com/40622501/187359819-c42532b5-5614-4830-a24f-664b664c0a9d.png)

## results
![metrics](https://user-images.githubusercontent.com/40622501/187360348-57c694a1-252a-4743-bf24-3e5a3c1b0328.png)

## references
- [1] Z. Sun et al., “Are we evaluating rigorously? Benchmarking recommendation for reproducible evaluation and fair comparison,” presented at the RecSys ’20: Fourteenth ACM Conference on Recommender Systems, Virtual Event Brazil, Sep. 2020. doi: 10.1145/3383313.3412489. 
- [2] M. Chen et al., “Values of User Exploration in Recommender Systems,” in Fifteenth ACM Conference on Recommender Systems, New York, NY, USA: Association for Computing Machinery, 2021, pp. 85‒95.
