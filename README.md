# DAN-SNR
A Deep Attentive Network for Social-Aware Next Point-of-Interest Recommendation

Next (or successive) point-of-interest (POI) recommendation has attracted increasing attention in recent years. Most of the previous studies attempted to incorporate the spatiotemporal information and sequential patterns of user check-ins into recommendation models to predict the target user’s next move. However, none of these approaches utilized the social influence of each user’s friends. In this study, we discuss a new topic of next POI recommendation and present a deep attentive network for social-aware next POI recommendation called DAN-SNR. In particular, the DAN-SNR makes use of the self-attention mechanism instead of the architecture of recurrent neural networks to model sequential influence and social influence in a unified manner. Moreover, we design and implement two parallel channels to capture short-term user preference and long-term user preference as well as social influence, respectively. By leveraging multi-head self-attention, the DAN-SNR can model long-range dependencies between any two historical check-ins efficiently and weigh their contributions to the next destination adaptively. Also, we carried out a comprehensive evaluation using large-scale real-world datasets collected from two popular location-based social networks, namely Gowalla and Brightkite. Experimental results indicate that the DAN-SNR outperforms seven competitive baseline approaches regarding recommendation performance and is of high efficiency among six neural-network- and attention-based methods.

Next, we introduce how to run our model for provided example data or your own data.

# Environment

Python 3.7

TensorFlow 1.2.0

Numpy 1.15.0

# Usage
As an illustration, we provide the data and running command for Gowalla and Brightkite.

# Input data
userlocation.csv：includes user ID, POI ID, latitude, longitude, checkin time.

locations.csv: includes POI ID, latitude, longitude, city ID

position_embedding.csv: includes the embeddings of all POIs  

user_embedding.csv: includes the embeddings of all users  

network.csv: includes all friendships of all users (from usrID, to userID)

# Contact
Liwei Huang, dr_huanglw@163.com

# Citation
This work has been published on ACM transactions on Internet technology. If you use DAN-SNR in your research, please cite our paper:

Liwei Huang, Yutao Ma, Yanbo Liu, and Keqing He. 2020. DAN-SNR: A Deep Attentive Network for Social aware Next Point-of-interest Recommendation. ACM Trans. Internet Technol. 21, 1, Article 2 (December 2020), 27 pages.
https://doi.org/10.1145/3430504
