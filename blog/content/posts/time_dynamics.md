---
title: "Time Dynamics in Online Rating Systems"
date: 2024-11-10
draft: true
tags: ["rating systems", "Glicko", "TrueSkill"]
---

#### Note
I'm writing this post mostly to solidify my own understanding of the topic in order to design and run the correct experiments.

## Background
The core idea behind rating systems is very simple:
* win -> rating go up
  * win against stronger opponent -> rating go up more
  * win against weaker opponent -> rating go up less
* lose -> rating go down
  * lose against stronger opponent -> rating go down less
  * lose against weaker opponent -> rating go down more

This is the basis of Elo [[1](https://en.wikipedia.org/wiki/Elo_rating_system)] and most other ratings systems and is described more technically as the "skills outcome model" in [[2](https://arxiv.org/abs/2104.14012)].

However, when other rating systems like Glicko [[3](http://www.glicko.net/glicko.html)] and TrueSkill [[4](https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/)], introduced the idea of measuring the variance or uncertainty in skill in addition to the skill rating itself, it came along with the question of how we deal with the change in varaince over time. This is a departure from the model used in Elo, where the updates are based only on the outcomes, now there is a portion of the update that depends on *when* the games are played and how frequently.

## Glicko
The Glicko system maintains a "rating deviation" (RD) for each competitor. This is the standard deviation (square root of variance) of the skill rating variable, and represents the uncertaintly about the current rating. If the RD is high, we have high uncertainty (low certainty) about that rating estimate. If the RD is low, we have low uncertainty (high certainty). When we see a competitor for the first time, they have high uncertainty since we know nothing about them, so we start the RD out with a high value. As we observe their results over time, the *outcome based* updates will lower their RD as we increase our certainty with evidence. 

However, if there is a long period of time in which they do not play, intuitively our uncertainty grows in that time even though there has been no outcomes to update their RDs with. Glicko handles this by having a hyperparameter $c$ which represents the increase in RD per unit time of not observing a result.


### Sources
[1] Wikipedia contributors. *Elo rating system*. Wikipedia, The Free Encyclopedia. Available: [https://en.wikipedia.org/wiki/Elo_rating_system](https://en.wikipedia.org/wiki/Elo_rating_system)

[2] Szczecinski, L., & Tihon, R. (2021). Simplified Kalman filter for online rating: one-fits-all approach. *arXiv preprint arXiv:2104.14012*. [https://arxiv.org/abs/2104.14012](https://arxiv.org/abs/2104.14012)

[3] Glickman, M. E. (1999). Parameter estimation in large dynamic paired comparison experiments. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 48(3), 377–394. Oxford University Press. [http://www.glicko.net/research/glicko.pdf](http://www.glicko.net/research/glicko.pdf)

[4] Herbrich, R., Minka, T., & Graepel, T. (2006). TrueSkill™: a Bayesian skill rating system. *Advances in Neural Information Processing Systems*, 19. [https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/](https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/)





